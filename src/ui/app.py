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
import re
import secrets
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
from src.outreach.quality_gate import QUALITY_GATE_CHECKS_METADATA
from src.outreach.send_log import SendLog
from src.profile.loader import load_profile
from src.ui.cv_validator import MAX_CV_SIZE_BYTES, format_file_size, validate_cv_stream
from src.ui.storage import (
    _atomic_write_json,
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
    regenerate_draft_content,
    delete_draft,
    delete_drafts_for_batch,
    clear_all_drafts,
)
from src.workflow.batch_runner import BackgroundBatchRunner
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
    # Environment-based secret key loading with production validation
    env_secret = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY")
    flask_env = os.getenv("FLASK_ENV", "development").lower()
    is_production = flask_env in ("production", "prod")

    if is_production:
        if not env_secret or env_secret in ("smartmail-local-testing-secret-key-2026", "dev", "secret", "change-me"):
            raise RuntimeError(
                "Insecure configuration: FLASK_SECRET_KEY must be set to a strong, unique secret in production environments."
            )
        secret_key = env_secret
    else:
        # Development / test mode: load from env or generate a secure random secret for the process
        secret_key = env_secret if env_secret else secrets.token_hex(32)

    app.config["SECRET_KEY"] = secret_key
    app.config["SEND_LOG_PATH"] = "data/send_log.db"
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max body size
    app.config.setdefault("WTF_CSRF_ENABLED", True)
    if is_production:
        app.config["SESSION_COOKIE_SECURE"] = True

    if test_config:
        app.config.update(test_config)

    @app.before_request
    def check_csrf() -> Any:
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            if not app.config.get("WTF_CSRF_ENABLED", True):
                return None
            from flask import session
            expected = session.get("csrf_token")
            token = (
                request.form.get("csrf_token")
                or request.headers.get("X-CSRFToken")
                or request.headers.get("X-CSRF-Token")
            )
            if not token and request.is_json:
                data = request.get_json(silent=True)
                if isinstance(data, dict):
                    token = data.get("csrf_token")

            if not expected or not token or not secrets.compare_digest(str(token), str(expected)):
                is_ajax = (
                    request.is_json
                    or request.headers.get("X-Requested-With") == "XMLHttpRequest"
                    or "application/json" in request.headers.get("Accept", "")
                )
                if not expected:
                    err_msg = "CSRF validation failed: session expired or missing. Please refresh the page and try again."
                else:
                    err_msg = "CSRF validation failed: missing or invalid CSRF token. Please refresh the page and try again."

                if is_ajax:
                    return jsonify({"success": False, "error": err_msg}), 400
                return err_msg, 400
        return None

    @app.errorhandler(413)
    def request_entity_too_large(error: Any) -> Any:
        is_ajax = (
            request.is_json
            or request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or "application/json" in request.headers.get("Accept", "")
        )
        if is_ajax:
            return jsonify({"success": False, "error": "Uploaded file exceeds maximum allowed size (16 MB)."}), 413
        flash("The uploaded file exceeds the 16 MB size limit.", "danger")
        return redirect(request.referrer or url_for("dashboard")), 413

    @app.template_filter("normalize_evidence")
    def jinja_normalize_evidence(evidence_data: Any) -> list[dict[str, str]]:
        return normalize_evidence_summary(evidence_data)

    def _get_send_log() -> SendLog:
        return SendLog(db_path=app.config.get("SEND_LOG_PATH", "data/send_log.db"))

    @app.context_processor
    def inject_global_template_defaults() -> dict[str, Any]:
        from flask import session
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_hex(32)
        return {
            "standard_fields": list(STANDARD_FIELDS),
            "field_descriptions": FIELD_DESCRIPTIONS,
            "csrf_token": session["csrf_token"],
        }

    # -----------------------------------------------------------------------
    # Route: Dashboard (Overview)
    # -----------------------------------------------------------------------
    @app.route("/")
    @app.route("/overview")
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
            app.logger.warning("CSV preview error: %s", exc)
            return jsonify({"success": False, "error": "Failed to parse CSV preview. Please verify file format."}), 500
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
            try:
                if saved_path.exists():
                    saved_path.unlink()
            except Exception:
                pass
            app.logger.warning("Spreadsheet inspection failed: %s", exc)
            return jsonify({"success": False, "error": "Failed to inspect spreadsheet file. Ensure the file is not corrupted."}), 400

    # -----------------------------------------------------------------------
    # Stage 2: Pre-Flight Column Mapping & Validation Preview
    # -----------------------------------------------------------------------
    @app.route("/contacts/preview", methods=["POST"])
    def preview_contacts_mapping() -> Response:
        data = request.get_json(silent=True) or request.form.to_dict()
        file_token = data.get("file_token", "").strip()
        if not file_token or not re.match(r"^[a-zA-Z0-9_-]{1,64}$", file_token):
            return jsonify({"success": False, "error": "Invalid file token format."}), 400
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

        batches_dir = Path("output/batches_input").resolve()
        matching = [p for p in batches_dir.glob(f"{file_token}_*") if p.resolve().is_relative_to(batches_dir)]
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
            app.logger.warning("Spreadsheet preview mapping failed: %s", exc)
            return jsonify({"success": False, "error": "Failed to preview contact mapping."}), 400

    # -----------------------------------------------------------------------
    # Stage 2: Download Validation CSV Report
    # -----------------------------------------------------------------------
    @app.route("/contacts/download-report", methods=["POST"])
    def download_validation_report() -> Response:
        data = request.get_json(silent=True) or request.form.to_dict()
        file_token = data.get("file_token", "").strip()
        if not file_token or not re.match(r"^[a-zA-Z0-9_-]{1,64}$", file_token):
            return Response("Invalid file token format", status=400)
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

        batches_dir = Path("output/batches_input").resolve()
        matching = [p for p in batches_dir.glob(f"{file_token}_*") if p.resolve().is_relative_to(batches_dir)]
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
            app.logger.warning("Download validation report failed: %s", exc)
            return Response("Failed to generate validation report.", status=400)

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
        stage = "initializing workflow"
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
                stage = "validating session token"
                if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", file_token):
                    flash("Invalid spreadsheet token format. Please re-upload your file.", "danger")
                    return redirect(url_for("import_contacts_page"))
                batches_dir = Path("output/batches_input").resolve()
                matching = [p for p in batches_dir.glob(f"{file_token}_*") if p.resolve().is_relative_to(batches_dir)]
                if not matching:
                    flash("Spreadsheet session expired. Please re-upload your contact file.", "warning")
                    return redirect(url_for("import_contacts_page"))
                contact_file_path = matching[0]
                sheet_name = request.form.get("sheet_name")
                mapping_str = request.form.get("mapping")
                mapping = json.loads(mapping_str) if mapping_str else None

                stage = "parsing spreadsheet and applying column mapping"
                parse_res = parse_spreadsheet_contacts(contact_file_path, mapping=mapping, sheet_name=sheet_name)
            else:
                # Legacy / direct CSV upload fallback
                stage = "saving uploaded CSV"
                output_dir = Path("output/batches_input")
                output_dir.mkdir(parents=True, exist_ok=True)
                csv_save_path = output_dir / f"{batch_id}.csv"
                file.save(str(csv_save_path))

                stage = "parsing uploaded CSV"
                parse_res = import_contacts_from_csv(csv_save_path)

            # Start background batch execution with bounded concurrency
            stage = "scheduling background batch runner"
            runner = BackgroundBatchRunner.get_instance()
            wait_for_completion = bool(app.config.get("TESTING", False))
            runner.start_batch(
                batch_id=batch_id,
                import_result=parse_res,
                workflow=workflow,
                target_role=target_role,
                job_description=job_description,
                campaign_id=campaign_id,
                mode=mode,
                max_workers=4,
                wait_for_completion=wait_for_completion,
            )

            flash(f"Batch {batch_id} initiated. Processing in background.", "info")
            return redirect(url_for("view_batch_result", batch_id=batch_id))

        except Exception as exc:
            app.logger.exception("Batch processing failure at stage '%s' for campaign '%s': %s", stage, campaign_id, exc)
            flash(
                f"Batch initiation failed during {stage}: {type(exc).__name__} ({str(exc)[:100]}). "
                "Please review your inputs or column mapping and try again.",
                "danger",
            )
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

    @app.route("/batch/<batch_id>/status")
    def batch_status_json(batch_id: str) -> Any:
        """JSON progress endpoint for live UI updates during background processing."""
        batch = get_batch_record(batch_id)
        if not batch:
            return jsonify({"error": f"Batch '{batch_id}' not found."}), 404
        status = batch.get("status", "COMPLETED")
        total = batch.get("total_rows", 0)
        proc = batch.get("processed_count", 0)
        if status == "ERROR":
            pct = int((proc / max(1, total)) * 100) if total > 0 else 0
        else:
            pct = int((proc / max(1, total)) * 100) if total > 0 else 100
        return jsonify({
            "batch_id": batch_id,
            "status": status,
            "error": batch.get("error", None),
            "total_rows": total,
            "processed_count": proc,
            "drafted_count": batch.get("drafted_count", 0),
            "review_required_count": batch.get("review_required_count", 0),
            "blocked_count": batch.get("blocked_count", 0),
            "failed_count": batch.get("failed_count", 0),
            "percent": pct,
        })

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
        approved_count = sum(1 for d in all_drafts if d.get("status") == "APPROVED")
        sent_count = sum(1 for d in all_drafts if d.get("status") in ("SENT", "SIMULATED_SENT"))

        return render_template(
            "review.html",
            active_tab="review",
            drafts=filtered_drafts,
            current_filter=status_filter,
            total_count=len(all_drafts),
            needs_review_count=needs_review_count,
            approved_count=approved_count,
            sent_count=sent_count,
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
        sender_email = os.getenv("SMARTMAIL_TEST_SENDER_EMAIL") or os.getenv("GMAIL_SENDER_ADDRESS") or "test-sender@example.com"

        return render_template(
            "draft.html",
            active_tab="review",
            draft=draft,
            v_details=v_details,
            cv_exists=cv_exists,
            evidence_items=evidence_items,
            sender_email=sender_email,
            quality_gate_metadata=QUALITY_GATE_CHECKS_METADATA,
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
                send_log=_get_send_log(),
            )
            flash("Draft updated. Approval status reset to enforce re-review.", "info")
        except Exception as exc:
            app.logger.warning("Error saving draft edits: %s", exc)
            flash("An error occurred while saving draft edits. Please try again.", "danger")

        return redirect(url_for("view_draft", draft_id=draft_id))

    @app.route("/drafts/<draft_id>/regenerate", methods=["POST"])
    def regenerate_draft_route(draft_id: str) -> Any:
        try:
            regenerate_draft_content(draft_id, send_log=_get_send_log())
            flash("Draft successfully regenerated based on verified company research.", "success")
        except Exception as exc:
            app.logger.warning("Error regenerating draft %s: %s", draft_id, exc)
            flash(f"Could not regenerate draft: {exc}", "danger")

        return redirect(url_for("view_draft", draft_id=draft_id))

    @app.route("/drafts/<draft_id>/approve", methods=["POST"])
    def approve_draft_route(draft_id: str) -> Any:
        try:
            approve_draft(draft_id=draft_id, send_log=_get_send_log())
            flash(f"Draft {draft_id} explicitly approved for sending.", "success")
        except Exception as exc:
            app.logger.warning("Error approving draft: %s", exc)
            flash("An error occurred while approving the draft. Please try again.", "danger")

        return redirect(url_for("view_draft", draft_id=draft_id))

    @app.route("/drafts/<draft_id>/reject", methods=["POST"])
    def reject_draft_route(draft_id: str) -> Any:
        reason = request.form.get("reason", "Marked rejected by reviewer.")
        try:
            reject_draft(draft_id=draft_id, reason=reason, send_log=_get_send_log())
            flash(f"Draft {draft_id} rejected and placed on hold.", "warning")
        except Exception as exc:
            app.logger.warning("Error rejecting draft: %s", exc)
            flash("An error occurred while rejecting the draft. Please try again.", "danger")

        return redirect(url_for("view_draft", draft_id=draft_id))

    @app.route("/drafts/<draft_id>/delete", methods=["POST"])
    def delete_draft_route(draft_id: str) -> Any:
        try:
            success = delete_draft(draft_id=draft_id, send_log=_get_send_log())
            if success:
                flash(f"Draft {draft_id} permanently deleted.", "info")
            else:
                flash(f"Draft {draft_id} could not be deleted or was not found.", "warning")
        except Exception as exc:
            app.logger.warning("Error deleting draft: %s", exc)
            flash(f"Error deleting draft: {exc}", "danger")
        return redirect(url_for("review_queue"))

    @app.route("/batches/<batch_id>/delete-drafts", methods=["POST"])
    def delete_batch_drafts_route(batch_id: str) -> Any:
        try:
            res = delete_drafts_for_batch(batch_id=batch_id, send_log=_get_send_log())
            flash(f"Deleted {res.get('deleted_count', 0)} drafts from batch '{batch_id}'.", "info")
        except Exception as exc:
            app.logger.warning("Error deleting batch drafts: %s", exc)
            flash(f"Error deleting batch drafts: {exc}", "danger")
        return redirect(url_for("view_batch_result", batch_id=batch_id))

    @app.route("/drafts/cleanup", methods=["POST"])
    def cleanup_drafts_route() -> Any:
        scope = request.form.get("scope", "unapproved").strip().lower()
        batch_id = request.form.get("batch_id", "").strip()
        try:
            if scope == "batch" and batch_id:
                res = delete_drafts_for_batch(batch_id, send_log=_get_send_log())
                cnt = res.get("deleted_count", 0)
                flash(f"Batch cleanup complete: successfully deleted {cnt} draft(s) for campaign {batch_id}.", "success")
            else:
                if scope not in ("unapproved", "rejected", "all"):
                    scope = "unapproved"
                res = clear_all_drafts(scope=scope, send_log=_get_send_log())
                cnt = res.get("deleted_count", 0)
                scope_label = "all" if scope == "all" else ("unapproved" if scope == "unapproved" else "rejected")
                flash(f"Cleanup complete: successfully deleted {cnt} {scope_label} draft(s).", "success")
        except Exception as exc:
            app.logger.warning("Error during draft cleanup: %s", exc)
            flash(f"Error during draft cleanup: {exc}", "danger")
        return redirect(request.referrer or url_for("dashboard"))

    @app.route("/drafts/<draft_id>/send", methods=["POST"])
    def send_draft_route(draft_id: str) -> Any:
        draft = get_draft(draft_id)
        if not draft:
            flash(f"Draft {draft_id} not found.", "danger")
            return redirect(url_for("review_queue"))

        send_log = _get_send_log()
        mode = WorkflowMode(draft.get("mode", "uploaded_cv"))

        # Initialize workflow with dry_run=True (Safety default unless explicitly configured)
        is_dry_run = app.config.get("DRY_RUN", True)
        workflow = OutreachWorkflow(
            default_mode=mode,
            uploaded_cv_path=draft.get("attachment_path") if mode == WorkflowMode.UPLOADED_CV else None,
            email_sender_provider=app.config.get("EMAIL_SENDER_PROVIDER"),
            dry_run=is_dry_run,
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

        draft_file = Path("output/.drafts") / f"{draft_id}.json"
        if send_res.success:
            if is_dry_run:
                draft["status"] = DraftStatus.SIMULATED_SENT.value
            else:
                draft["status"] = DraftStatus.SENT.value
            if draft_file.exists():
                _atomic_write_json(draft_file, draft)
            flash(f"Send outcome: {send_res.message}", "success")
        elif send_res.error == "UNCERTAIN_SEND":
            draft["status"] = DraftStatus.UNCERTAIN.value
            if draft_file.exists():
                _atomic_write_json(draft_file, draft)
            flash(f"Send outcome uncertain ({send_res.error}): {send_res.message}", "warning")
        elif send_res.error in ("PROVIDER_ERROR", "NETWORK_ERROR", "TIMEOUT") or (not send_res.success and send_res.error not in (
            "APPROVAL_REQUIRED",
            "DUPLICATE_RECIPIENT",
            "ALREADY_SENT",
            "LIVE_SENDING_DISABLED",
            "GMAIL_NOT_CONFIGURED",
            "INVALID_RECIPIENT",
            "EMPTY_SUBJECT",
            "EMPTY_BODY",
            "MISSING_ATTACHMENT",
            "ATTACHMENT_NOT_FOUND",
            "INVALID_ATTACHMENT",
            "SEND_IN_PROGRESS",
            "CONTENTION_ERROR",
        )):
            draft["status"] = DraftStatus.FAILED.value
            if draft_file.exists():
                _atomic_write_json(draft_file, draft)
            flash(f"Send failed ({send_res.error}): {send_res.message}", "danger")
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
            meta = store_and_register_cv(file_bytes=file.stream, original_filename=file.filename)
            if is_ajax:
                return jsonify({"success": True, "cv": meta})
            flash(f"Successfully registered CV: {meta['filename']} ({meta['size_formatted']})", "success")
            return redirect(url_for("import_contacts_page"))
        except Exception as exc:
            app.logger.warning("CV upload validation error: %s", exc)
            if is_ajax:
                return jsonify({"success": False, "error": "CV validation failed. Please upload a valid PDF up to 10 MB."}), 400
            flash("CV validation failed. Please upload a valid PDF up to 10 MB.", "danger")
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
        if not safe_name.lower().endswith(".pdf"):
            flash("Access denied: only PDF files may be downloaded.", "danger")
            return redirect(url_for("cv_settings"))
        # Look strictly in output/uploaded_cvs, output/tailored_cvs, templates
        candidate_dirs = [
            Path("output/uploaded_cvs").resolve(),
            Path("output/tailored_cvs").resolve(),
            Path("templates").resolve(),
        ]
        for cdir in candidate_dirs:
            target = (cdir / safe_name).resolve()
            if target.is_relative_to(cdir) and target.exists() and target.is_file():
                return send_file(str(target), as_attachment=False)

        # Fallback check against registered CV path if safe_name matches filename or stored_filename
        cv_meta = get_registered_cv_metadata()
        if cv_meta.get("exists") and cv_meta.get("path"):
            reg_path = Path(cv_meta["path"]).resolve()
            if (safe_name == cv_meta.get("filename") or safe_name == cv_meta.get("stored_filename")) and reg_path.exists():
                return send_file(str(reg_path), as_attachment=False)

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
