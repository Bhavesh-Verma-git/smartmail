"""
src/ui/app.py
-------------
Flask web application for the SmartMail Functional Testing UI.
Directly interfaces with the SmartMail backend without fabricating any data.
"""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from src.contacts.column_mapper import (
    FIELD_DESCRIPTIONS,
    STANDARD_FIELDS,
    auto_map_columns,
    validate_user_mapping,
)
from src.contacts.csv_importer import import_contacts_from_csv
from src.contacts.spreadsheet_importer import (
    SpreadsheetError,
    inspect_spreadsheet,
    parse_spreadsheet_contacts,
)
from src.cv.generator import generate_tailored_cv
from src.cv.models import ProjectSelectionResult
from src.cv.selector import select_projects
from src.outreach.models import DraftStatus, EmailDraft, SendResult, WorkflowMode
from src.outreach.send_log import SendLog
from src.profile.loader import load_profile
from src.ui.cv_validator import MAX_CV_SIZE_BYTES, format_file_size, validate_cv_stream
from src.ui.storage import (
    approve_draft,
    clear_registered_cv,
    get_batch_record,
    get_dashboard_stats,
    get_draft,
    get_launch_readiness_checklist,
    get_registered_cv_metadata,
    get_session,
    list_all_drafts,
    list_batch_records,
    normalize_evidence_summary,
    register_uploaded_cv,
    reject_draft,
    save_batch_record,
    save_draft_record,
    save_session,
    store_and_register_cv,
    update_draft,
)
from src.workflow.engine import OutreachWorkflow


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask web application."""
    template_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )
    app.config["SECRET_KEY"] = "smartmail-local-testing-secret-key-2026"
    app.config["SEND_LOG_PATH"] = "data/send_log.db"
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    if test_config:
        app.config.update(test_config)

    @app.template_filter("normalize_evidence")
    def jinja_normalize_evidence(evidence_data: Any) -> list[dict[str, str]]:
        return normalize_evidence_summary(evidence_data)

    def _get_send_log() -> SendLog:
        return SendLog(db_path=app.config.get("SEND_LOG_PATH", "data/send_log.db"))

    @app.context_processor
    def inject_global_template_defaults() -> dict[str, Any]:
        return {
            "standard_fields": list(STANDARD_FIELDS),
            "field_descriptions": FIELD_DESCRIPTIONS,
        }

    # -----------------------------------------------------------------------
    # Route: Dashboard
    # -----------------------------------------------------------------------
    @app.route("/")
    def dashboard() -> str:
        stats = get_dashboard_stats(send_log=_get_send_log())
        return render_template("dashboard.html", active_tab="dashboard", stats=stats)

    # -----------------------------------------------------------------------
    # Route: Contact Import Page (4-Step Wizard)
    # -----------------------------------------------------------------------
    @app.route("/import", methods=["GET"])
    def import_contacts_page() -> str:
        cv_meta = get_registered_cv_metadata()
        default_campaign_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
        return render_template(
            "import.html",
            active_tab="import",
            registered_cv=cv_meta or {},
            current_mode="uploaded_cv",
            default_campaign_id=default_campaign_id,
            standard_fields=list(STANDARD_FIELDS),
            field_descriptions=FIELD_DESCRIPTIONS,
        )

    # -----------------------------------------------------------------------
    # Route: Pre-Upload CSV Validation Preview (AJAX - Backwards Compatible)
    # -----------------------------------------------------------------------
    @app.route("/import/preview", methods=["POST"])
    def preview_csv_upload() -> Response:
        file = request.files.get("csv_file")
        if not file or not file.filename:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            from src.contacts.email_parser import _FREE_PROVIDERS
            import_result = import_contacts_from_csv(tmp_path)
            contacts_data = []
            for c in import_result.contacts:
                is_free = (
                    c.email_result.registered_domain in _FREE_PROVIDERS
                    or c.email_result.email_domain in _FREE_PROVIDERS
                )
                contacts_data.append({
                    "row_number": c.row_number,
                    "email": c.email_result.normalized_email,
                    "company_name": c.company_name,
                    "contact_name": c.contact_name,
                    "website": c.website,
                    "role_or_job_url": c.role_or_job_url,
                    "source_url": c.source_url,
                    "is_free_provider": is_free,
                })

            errors_data = []
            for err in import_result.errors:
                errors_data.append({
                    "row_number": err.row_number,
                    "raw_row": err.raw_row,
                    "reason": err.reason,
                })

            return jsonify({
                "success": True,
                "valid_count": len(contacts_data),
                "error_count": len(errors_data),
                "contacts": contacts_data,
                "errors": errors_data,
            })
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Stage 2: Inspect Spreadsheet (CSV, XLSX, XLS)
    # -----------------------------------------------------------------------
    @app.route("/contacts/inspect", methods=["POST"])
    def inspect_contacts_spreadsheet() -> Response:
        file = request.files.get("contacts_file") or request.files.get("contact_file")
        if not file or not file.filename:
            return jsonify({"success": False, "error": "No spreadsheet file selected."}), 400

        suffix = Path(file.filename).suffix.lower()
        if suffix not in (".csv", ".xlsx", ".xls"):
            return jsonify({
                "success": False,
                "error": f"Unsupported format '{suffix}'. Please upload a .csv, .xlsx, or .xls file."
            }), 400

        # Save to output/batches_input with unique token
        batches_dir = Path("output/batches_input")
        batches_dir.mkdir(parents=True, exist_ok=True)
        file_token = uuid.uuid4().hex[:12]
        safe_name = Path(file.filename).name.replace(" ", "_")
        saved_path = batches_dir / f"{file_token}_{safe_name}"
        file.save(str(saved_path))

        try:
            sheet_param = request.form.get("sheet_name")
            inspection = inspect_spreadsheet(saved_path, sheet_name=sheet_param)
            return jsonify({
                "success": True,
                "file_token": file_token,
                "filename": file.filename,
                "file_type": inspection.file_type,
                "sheet_names": inspection.sheet_names,
                "selected_sheet": inspection.selected_sheet,
                "headers": inspection.headers,
                "total_rows_approx": inspection.total_rows_approx,
                "proposed_mapping": inspection.proposed_mapping.mapping,
                "email_column": inspection.proposed_mapping.email_column,
                "email_confident": inspection.proposed_mapping.email_confident,
                "mapping_notes": inspection.proposed_mapping.mapping_notes,
                "standard_fields": list(STANDARD_FIELDS),
                "field_descriptions": FIELD_DESCRIPTIONS,
            })
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    # -----------------------------------------------------------------------
    # Stage 2: Pre-Flight Column Mapping & Validation Preview
    # -----------------------------------------------------------------------
    @app.route("/contacts/preview", methods=["POST"])
    def preview_contacts_mapping() -> Response:
        data = request.get_json(silent=True) or request.form.to_dict()
        file_token = data.get("file_token", "").strip()
        sheet_name = data.get("sheet_name")
        mapping_raw = data.get("mapping")

        if isinstance(mapping_raw, str):
            try:
                mapping = json.loads(mapping_raw)
            except Exception:
                mapping = None
        elif isinstance(mapping_raw, dict):
            mapping = mapping_raw
        else:
            mapping = None

        batches_dir = Path("output/batches_input")
        matching = list(batches_dir.glob(f"{file_token}_*"))
        if not matching:
            return jsonify({"success": False, "error": f"Spreadsheet token '{file_token}' expired or not found."}), 404

        file_path = matching[0]
        try:
            parse_res = parse_spreadsheet_contacts(file_path, mapping=mapping, sheet_name=sheet_name)
            return jsonify({
                "success": True,
                "summary": {
                    "total_source_rows": parse_res.summary.total_source_rows,
                    "total_data_rows": parse_res.summary.total_data_rows,
                    "valid_email_count": parse_res.summary.valid_email_count,
                    "invalid_syntax_count": parse_res.summary.invalid_syntax_count,
                    "missing_email_count": parse_res.summary.missing_email_count,
                    "duplicate_email_count": parse_res.summary.duplicate_email_count,
                    "missing_company_count": parse_res.summary.missing_company_count,
                    "missing_name_count": parse_res.summary.missing_name_count,
                    "free_provider_count": parse_res.summary.free_provider_count,
                    "duplicate_details": [
                        {
                            "email": d.email,
                            "first_row": d.first_row,
                            "duplicate_rows": d.duplicate_rows,
                        }
                        for d in parse_res.summary.duplicate_details
                    ],
                },
                "applied_mapping": parse_res.applied_mapping,
                "rows": [
                    {
                        "row_number": r.row_number,
                        "email": r.email,
                        "contact_name": r.contact_name,
                        "company_name": r.company_name,
                        "website": r.website,
                        "role_or_job_url": r.role_or_job_url,
                        "status": r.status,
                        "is_valid": r.is_valid,
                        "reason": r.reason,
                        "is_free_provider": r.is_free_provider,
                        "is_missing_company": r.is_missing_company,
                        "is_missing_name": r.is_missing_name,
                    }
                    for r in parse_res.rows
                ],
            })
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    # -----------------------------------------------------------------------
    # Stage 2: Download Validation CSV Report
    # -----------------------------------------------------------------------
    @app.route("/contacts/download-report", methods=["POST"])
    def download_validation_report() -> Response:
        data = request.get_json(silent=True) or request.form.to_dict()
        file_token = data.get("file_token", "").strip()
        sheet_name = data.get("sheet_name")
        mapping_raw = data.get("mapping")

        if isinstance(mapping_raw, str):
            try:
                mapping = json.loads(mapping_raw)
            except Exception:
                mapping = None
        elif isinstance(mapping_raw, dict):
            mapping = mapping_raw
        else:
            mapping = None

        batches_dir = Path("output/batches_input")
        matching = list(batches_dir.glob(f"{file_token}_*"))
        if not matching:
            return Response("Spreadsheet file not found", status=404)

        file_path = matching[0]
        try:
            parse_res = parse_spreadsheet_contacts(file_path, mapping=mapping, sheet_name=sheet_name)
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Source Row", "Email", "Name", "Company", "Website", "Status", "Explanation"])
            for r in parse_res.rows:
                writer.writerow([r.row_number, r.email, r.contact_name, r.company_name, r.website, r.status, r.reason])

            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename=validation_report_{file_token[:8]}.csv"},
            )
        except Exception as exc:
            return Response(f"Validation error: {exc}", status=400)

    # -----------------------------------------------------------------------
    # Route: Process Batch Run
    # -----------------------------------------------------------------------
    @app.route("/import/process", methods=["POST"])
    def process_batch_run() -> Any:
        file_token = request.form.get("file_token", "").strip()
        file = request.files.get("csv_file")

        if not file_token and (not file or not file.filename):
            flash("Please upload or select a contact spreadsheet to process.", "warning")
            return redirect(url_for("import_contacts_page"))

        target_role = request.form.get("target_role", "AI/ML Intern").strip()
        mode_str = request.form.get("workflow_mode", "uploaded_cv").strip()
        campaign_id = request.form.get("campaign_id", "").strip() or f"camp_{uuid.uuid4().hex[:8]}"
        job_description = request.form.get("job_description", "").strip()

        mode = WorkflowMode.TAILORED_CV if mode_str == "tailored_cv" else WorkflowMode.UPLOADED_CV

        # Check uploaded CV availability for Uploaded CV mode
        cv_meta = get_registered_cv_metadata()
        uploaded_cv_path = cv_meta.get("path") if cv_meta.get("exists") else None

        send_log = _get_send_log()
        try:
            workflow = OutreachWorkflow(
                default_mode=mode,
                uploaded_cv_path=uploaded_cv_path,
                email_sender_provider=None,
                dry_run=True,  # Mandatory dry-run default
                send_log=send_log,
                campaign_id=campaign_id,
            )

            batch_id = f"batch_{uuid.uuid4().hex[:10]}"

            if file_token:
                # Stage 2: Process from spreadsheet token with custom mapping
                batches_dir = Path("output/batches_input")
                matching = list(batches_dir.glob(f"{file_token}_*"))
                if not matching:
                    flash(f"Spreadsheet session expired. Please re-upload your contact file.", "warning")
                    return redirect(url_for("import_contacts_page"))
                contact_file_path = matching[0]
                sheet_name = request.form.get("sheet_name")
                mapping_str = request.form.get("mapping")
                mapping = json.loads(mapping_str) if mapping_str else None

                parse_res = parse_spreadsheet_contacts(contact_file_path, mapping=mapping, sheet_name=sheet_name)
                batch_res = workflow.process_import_result(
                    import_result=parse_res,
                    job_description=job_description,
                    target_role=target_role,
                )
            else:
                # Legacy / direct CSV upload fallback
                output_dir = Path("output/batches_input")
                output_dir.mkdir(parents=True, exist_ok=True)
                csv_save_path = output_dir / f"{batch_id}.csv"
                file.save(str(csv_save_path))

                batch_res = workflow.process_csv(
                    csv_path=csv_save_path,
                    job_description=job_description,
                    target_role=target_role,
                )

            # Persist drafts and record outcomes
            results_list = []
            for i, wf_res in enumerate(batch_res.results):
                draft_id = wf_res.draft.draft_id if wf_res.draft else None
                rec_res = wf_res.recipient_resolution
                classification_dict = wf_res.classification.to_dict() if wf_res.classification else {}

                # Format evidence items safely for JSON
                evidence_list = []
                if wf_res.research_result and getattr(wf_res.research_result, "evidence", None):
                    for ev in wf_res.research_result.evidence:
                        evidence_list.append({
                            "source_url": getattr(ev, "source_url", ""),
                            "source_type": getattr(ev, "source_type", ""),
                            "title": getattr(ev, "title", ""),
                            "retrieval_status": getattr(ev, "retrieval_status", ""),
                            "extracted_text": getattr(ev, "extracted_text", "")[:400],
                            "claim_supported": getattr(ev, "claim_supported", ""),
                            "contradictions": getattr(ev, "contradictions", []),
                            "supports": getattr(ev, "supports", []),
                        })

                if wf_res.draft:
                    v_pipe = wf_res.verification
                    v_details = {}
                    if v_pipe:
                        v_details = {
                            "email_syntax_valid": True,
                            "company_identity_status": getattr(v_pipe.verification, "company_status", "REVIEW"),
                            "company_identity_note": "; ".join(getattr(v_pipe.verification, "verification_notes", [])) or "Evaluated by company research",
                            "domain_relationship_status": "SUPPORTED" if getattr(v_pipe.verification, "overall_status", "") != "BLOCKED" else "REVIEW",
                            "domain_relationship_note": "Domain checked against company identity",
                            "relevance_status": getattr(v_pipe.match, "relevance_status", "REVIEW"),
                            "matched_areas": getattr(v_pipe.match, "matched_areas", []),
                            "role_status": getattr(v_pipe.match, "case_type", "GENERAL_INQUIRY"),
                            "role_status_note": getattr(v_pipe.match, "explanation", ""),
                            "source_urls": getattr(v_pipe.verification, "evidence_sources", []),
                            "snippets": [],
                        }

                    save_draft_record(
                        draft=wf_res.draft,
                        campaign_id=campaign_id,
                        verification_details=v_details,
                        classification=classification_dict,
                        research_evidence=evidence_list,
                    )

                results_list.append({
                    "row_number": wf_res.row_number if wf_res.row_number is not None else (i + 1),
                    "status": wf_res.status,
                    "contact_email": wf_res.contact_email,
                    "company_name": wf_res.company_name,
                    "resolved_company": getattr(wf_res.research_result, "resolved_company_name", "") or wf_res.company_name,
                    "supplied_company": getattr(wf_res.contact, "company_name", "") if wf_res.contact else "",
                    "supplied_name": getattr(wf_res.contact, "contact_name", "") if wf_res.contact else "",
                    "inferred_name": getattr(rec_res, "inferred_name", "") if rec_res else "",
                    "verified_name": getattr(rec_res, "verified_name", "") if rec_res else "",
                    "recipient_name": getattr(rec_res, "resolved_name", "") if rec_res else (getattr(wf_res.contact, "contact_name", "") if wf_res.contact else ""),
                    "recipient_status": getattr(rec_res, "status", "UNKNOWN") if rec_res else "UNKNOWN",
                    "name_verification_status": getattr(rec_res, "name_verification_status", "UNVERIFIED") if rec_res else "UNVERIFIED",
                    "is_generic_inbox": getattr(rec_res, "is_generic", False) if rec_res else False,
                    "ai_ml_relevant": wf_res.classification.ai_ml_relevant if wf_res.classification else False,
                    "classification": wf_res.classification.primary_outcome if wf_res.classification else wf_res.status,
                    "reason_code": wf_res.classification.reason_code if wf_res.classification else "",
                    "reason_summary": wf_res.classification.reason_summary if wf_res.classification else wf_res.message,
                    "matched_areas": wf_res.classification.matched_areas if wf_res.classification else [],
                    "evidence_sources": wf_res.classification.evidence_sources if wf_res.classification else [],
                    "evidence_items": evidence_list,
                    "contradictions": wf_res.classification.contradictions if wf_res.classification else [],
                    "message": wf_res.message,
                    "error": wf_res.error,
                    "draft_id": draft_id,
                    "generation_status": getattr(wf_res.draft, "generation_status", "DRAFT_GENERATED") if wf_res.draft else "GENERATION_FAILED",
                    "personalization_explanation": getattr(wf_res.draft, "personalization_explanation", "") if wf_res.draft else "",
                    "matched_skills": getattr(wf_res.draft, "matched_skills", []) if wf_res.draft else [],
                    "matched_projects": getattr(wf_res.draft, "matched_projects", []) if wf_res.draft else [],
                    "original_fields": getattr(wf_res.contact, "raw_values", {}) if wf_res.contact else {},
                })

            # Save batch record
            batch_record = {
                "batch_id": batch_id,
                "campaign_id": campaign_id,
                "target_role": target_role,
                "mode": mode.value,
                "total_rows": batch_res.total_rows,
                "processed_count": batch_res.processed_count,
                "drafted_count": batch_res.drafted_count,
                "review_required_count": batch_res.review_required_count,
                "blocked_count": batch_res.blocked_count,
                "failed_count": batch_res.failed_count,
                "confident_count": batch_res.confident_count,
                "review_name_count": batch_res.review_name_count,
                "not_ai_relevant_count": batch_res.not_ai_relevant_count,
                "eliminated_count": batch_res.eliminated_count,
                "other_review_count": batch_res.other_review_count,
                "draft_generated_count": getattr(batch_res, "draft_generated_count", 0),
                "needs_review_count": getattr(batch_res, "needs_review_count", 0),
                "generation_failed_count": getattr(batch_res, "generation_failed_count", 0),
                "errors": batch_res.errors,
                "results": results_list,
            }
            save_batch_record(batch_id, batch_record)

            flash(f"Batch {batch_id} processed: {batch_res.drafted_count} drafts generated.", "success")
            return redirect(url_for("view_batch_result", batch_id=batch_id))

        except Exception as exc:
            flash(f"Batch processing error: {exc}", "danger")
            return redirect(url_for("import_contacts_page"))

    # -----------------------------------------------------------------------
    # Route: View Batch Results
    # -----------------------------------------------------------------------
    @app.route("/batch/<batch_id>")
    def view_batch_result(batch_id: str) -> Any:
        batch = get_batch_record(batch_id)
        if not batch:
            flash(f"Batch record '{batch_id}' not found.", "warning")
            return redirect(url_for("dashboard"))
        return render_template("batch.html", active_tab="import", batch=batch)

    # -----------------------------------------------------------------------
    # Route: Download Error Reports (CSV & JSON)
    # -----------------------------------------------------------------------
    @app.route("/batch/<batch_id>/errors/csv")
    def download_batch_errors_csv(batch_id: str) -> Response:
        batch = get_batch_record(batch_id)
        if not batch:
            return Response("Batch not found", status=404)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["row_number", "email", "reason", "error"])

        for err in batch.get("errors", []):
            email = err.get("email", "")
            if not email and isinstance(err.get("raw_row"), dict):
                email = err["raw_row"].get("email", "")
            writer.writerow([
                err.get("row_number", ""),
                email,
                err.get("reason", ""),
                err.get("error", ""),
            ])

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={batch_id}_errors.csv"},
        )

    @app.route("/batch/<batch_id>/errors/json")
    def download_batch_errors_json(batch_id: str) -> Response:
        batch = get_batch_record(batch_id)
        if not batch:
            return Response(json.dumps({"error": "Batch not found"}), mimetype="application/json", status=404)

        errors_payload = {
            "batch_id": batch_id,
            "campaign_id": batch.get("campaign_id", ""),
            "failed_count": batch.get("failed_count", 0),
            "errors": batch.get("errors", []),
        }
        return Response(
            json.dumps(errors_payload, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment; filename={batch_id}_errors.json"},
        )

    # -----------------------------------------------------------------------
    # Route: Review Queue
    # -----------------------------------------------------------------------
    @app.route("/review")
    def review_queue() -> str:
        status_filter = request.args.get("status")
        all_drafts = list_all_drafts()
        filtered_drafts = list_all_drafts(status_filter=status_filter)

        needs_review_count = sum(
            1 for d in all_drafts
            if d.get("review_required") and not d.get("approved") and d.get("status") not in ("APPROVED", "SENT", "REJECTED")
        )

        return render_template(
            "review.html",
            active_tab="review",
            drafts=filtered_drafts,
            current_filter=status_filter,
            total_count=len(all_drafts),
            needs_review_count=needs_review_count,
        )

    # -----------------------------------------------------------------------
    # Route: Draft Detail & Editor
    # -----------------------------------------------------------------------
    @app.route("/drafts/<draft_id>")
    def view_draft(draft_id: str) -> Any:
        draft = get_draft(draft_id)
        if not draft:
            flash(f"Draft '{draft_id}' not found.", "warning")
            return redirect(url_for("review_queue"))

        v_details = draft.get("verification", {})
        att_path = draft.get("attachment_path", "")
        cv_exists = bool(att_path and Path(att_path).exists())
        evidence_items = normalize_evidence_summary(draft.get("evidence_summary"))

        return render_template(
            "draft.html",
            active_tab="review",
            draft=draft,
            v_details=v_details,
            cv_exists=cv_exists,
            evidence_items=evidence_items,
        )

    @app.route("/drafts/<draft_id>/edit", methods=["POST"])
    def edit_draft_route(draft_id: str) -> Any:
        subject = request.form.get("subject")
        body = request.form.get("body")
        recipient = request.form.get("recipient")
        recipient_name = request.form.get("recipient_name")

        try:
            update_draft(
                draft_id=draft_id,
                subject=subject,
                body=body,
                recipient=recipient,
                recipient_name=recipient_name,
                reset_approval=True,  # Safeguard: editing resets approval!
            )
            flash("Draft updated. Approval status reset to enforce re-review.", "info")
        except Exception as exc:
            flash(f"Error saving draft edits: {exc}", "danger")

        return redirect(url_for("view_draft", draft_id=draft_id))

    @app.route("/drafts/<draft_id>/approve", methods=["POST"])
    def approve_draft_route(draft_id: str) -> Any:
        try:
            approve_draft(draft_id=draft_id, send_log=_get_send_log())
            flash(f"Draft {draft_id} explicitly approved for sending.", "success")
        except Exception as exc:
            flash(f"Error approving draft: {exc}", "danger")

        return redirect(url_for("view_draft", draft_id=draft_id))

    @app.route("/drafts/<draft_id>/reject", methods=["POST"])
    def reject_draft_route(draft_id: str) -> Any:
        reason = request.form.get("reason", "Marked rejected by reviewer.")
        try:
            reject_draft(draft_id=draft_id, reason=reason)
            flash(f"Draft {draft_id} rejected and placed on hold.", "warning")
        except Exception as exc:
            flash(f"Error rejecting draft: {exc}", "danger")

        return redirect(url_for("view_draft", draft_id=draft_id))

    @app.route("/drafts/<draft_id>/send", methods=["POST"])
    def send_draft_route(draft_id: str) -> Any:
        draft = get_draft(draft_id)
        if not draft:
            flash(f"Draft {draft_id} not found.", "danger")
            return redirect(url_for("review_queue"))

        send_log = _get_send_log()
        mode = WorkflowMode(draft.get("mode", "uploaded_cv"))

        # Initialize workflow with dry_run=True (Safety default)
        workflow = OutreachWorkflow(
            default_mode=mode,
            uploaded_cv_path=draft.get("attachment_path") if mode == WorkflowMode.UPLOADED_CV else None,
            email_sender_provider=app.config.get("EMAIL_SENDER_PROVIDER"),
            dry_run=True,
            send_log=send_log,
            campaign_id=draft.get("campaign_id", ""),
        )

        # Populate draft into review manager
        email_draft = EmailDraft(
            draft_id=draft["draft_id"],
            recipient=draft["recipient"],
            recipient_name=draft.get("recipient_name", ""),
            company_name=draft.get("company_name", ""),
            subject=draft["subject"],
            body=draft["body"],
            attachment_path=draft["attachment_path"],
            mode=mode,
            status=DraftStatus(draft.get("status", "DRAFT")),
            approved=draft.get("approved", False),
            review_required=draft.get("review_required", False),
            review_flags=draft.get("review_flags", []),
        )
        workflow.review_manager.register_draft(email_draft)

        send_res: SendResult = workflow.send_draft(draft_id)

        if send_res.success:
            # Update local disk status to SENT
            draft["status"] = "SENT"
            draft_file = Path("output/.drafts") / f"{draft_id}.json"
            if draft_file.exists():
                draft_file.write_text(json.dumps(draft, indent=2), encoding="utf-8")
            flash(f"Send outcome: {send_res.message}", "success")
        else:
            flash(f"Send blocked ({send_res.error}): {send_res.message}", "warning")

        return redirect(url_for("view_draft", draft_id=draft_id))

    # -----------------------------------------------------------------------
    # Route: CV Mode Settings
    # -----------------------------------------------------------------------
    @app.route("/cv")
    def cv_settings() -> str:
        cv_meta = get_registered_cv_metadata()
        master_profile = {}
        candidate_name = "Candidate"
        try:
            master_profile = load_profile("data/master_profile.json")
            candidate_name = master_profile.get("personal_info", {}).get("full_name", "Candidate")
        except Exception:
            pass

        return render_template(
            "cv.html",
            active_tab="cv",
            cv_meta=cv_meta,
            candidate_name=candidate_name,
            master_profile=master_profile,
            tailored_result=None,
        )

    @app.route("/cv/upload", methods=["POST"])
    def upload_cv_file_route() -> Any:
        file = request.files.get("cv_file")
        is_ajax = request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest" or "application/json" in request.headers.get("Accept", "")

        if not file or not file.filename:
            if is_ajax:
                return jsonify({"success": False, "error": "No file was selected."}), 400
            flash("Please choose a PDF file to upload.", "warning")
            return redirect(url_for("import_contacts_page"))

        try:
            file_bytes = file.read()
            meta = store_and_register_cv(file_bytes=file_bytes, original_filename=file.filename)
            if is_ajax:
                return jsonify({"success": True, "cv": meta})
            flash(f"Successfully registered CV: {meta['filename']} ({meta['size_formatted']})", "success")
            return redirect(url_for("import_contacts_page"))
        except Exception as exc:
            if is_ajax:
                return jsonify({"success": False, "error": str(exc)}), 400
            flash(f"CV validation error: {exc}", "danger")
            return redirect(url_for("import_contacts_page"))

    @app.route("/cv/clear", methods=["POST"])
    def clear_cv_route() -> Any:
        clear_registered_cv()
        is_ajax = request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest"
        if is_ajax:
            return jsonify({"success": True})
        flash("Registered CV has been removed.", "info")
        return redirect(url_for("import_contacts_page"))

    @app.route("/cv/test-tailor", methods=["POST"])
    def test_cv_tailoring_route() -> str:
        target_company = request.form.get("target_company", "Acme AI Labs").strip()
        target_role = request.form.get("target_role", "AI/ML Intern").strip()
        job_description = request.form.get("job_description", "").strip()

        cv_meta = get_registered_cv_metadata()
        master_profile = load_profile("data/master_profile.json")
        candidate_name = master_profile.get("personal_info", {}).get("full_name", "Candidate")

        # Run project selection
        selection = select_projects(
            profile=master_profile,
            job_description=job_description,
            target_role=target_role,
        )

        selected_projects = []
        for p in selection.selected_projects:
            title = p.get("title", p.get("name", "Selected Project")) if isinstance(p, dict) else getattr(p, "title", "Selected Project")
            score = p.get("score", 100) if isinstance(p, dict) else getattr(p, "score", 100)
            reason = (
                p.get("ranking_reason", p.get("summary", "Selected by relevance algorithm"))
                if isinstance(p, dict)
                else getattr(p, "reason", "Selected by relevance algorithm")
            )
            selected_projects.append({
                "name": title,
                "score": score,
                "reason": reason,
            })

        # Run CV generation
        tailored_res = generate_tailored_cv(
            profile=master_profile,
            selection_result=selection,
            job_description=job_description,
            target_role=target_role,
            target_company=target_company,
        )

        pdf_filename = Path(tailored_res.pdf_path).name if tailored_res.pdf_path else ""

        result_view = {
            "selected_projects": selected_projects,
            "content": tailored_res.content,
            "pdf_path": tailored_res.pdf_path,
            "pdf_filename": pdf_filename,
            "pdf_error": tailored_res.pdf_error,
        }

        return render_template(
            "cv.html",
            active_tab="cv",
            cv_meta=cv_meta,
            candidate_name=candidate_name,
            master_profile=master_profile,
            tailored_result=result_view,
        )

    # -----------------------------------------------------------------------
    # Route: Send Log
    # -----------------------------------------------------------------------
    @app.route("/send-log")
    def view_send_log() -> str:
        sl = _get_send_log()
        entries = sl.list_all(limit=500)
        return render_template("send_log.html", active_tab="send_log", entries=entries)

    # -----------------------------------------------------------------------
    # Route: Launch Readiness Checklist
    # -----------------------------------------------------------------------
    @app.route("/checklist")
    def launch_checklist() -> str:
        checklist = get_launch_readiness_checklist()
        return render_template("checklist.html", active_tab="checklist", checklist=checklist)

    # -----------------------------------------------------------------------
    # Route: Serve CV Files (Downloaded safely)
    # -----------------------------------------------------------------------
    @app.route("/download/cv/<filename>")
    def download_cv_file(filename: str) -> Any:
        safe_name = Path(filename).name
        # Look in output/, output/tailored_cvs/, templates/
        candidate_dirs = [
            Path("output"),
            Path("output/tailored_cvs"),
            Path("templates"),
        ]
        for cdir in candidate_dirs:
            target = cdir / safe_name
            if target.exists() and target.is_file():
                return send_file(str(target.resolve()), as_attachment=False)

        flash(f"CV file '{safe_name}' not found on server.", "warning")
        return redirect(url_for("cv_settings"))

    return app


if __name__ == "__main__":
    app = create_app()
    print("=" * 65)
    print(" SmartMail Local Functional Testing UI running at:")
    print(" http://127.0.0.1:5000")
    print(" (Bound to local loopback; dry-run safe mode is default)")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=False)
