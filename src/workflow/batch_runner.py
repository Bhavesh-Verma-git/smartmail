"""
src/workflow/batch_runner.py
----------------------------
Background batch execution engine with bounded concurrency, incremental persistence,
and live progress tracking for the SmartMail web UI.

Requirements satisfied:
1. Non-blocking Flask execution: launches batch in background, returns batch_id immediately.
2. Standard-library ThreadPoolExecutor with configurable bounded worker pool (default 4).
3. Incremental disk persistence: persists each draft and updates batch state on completion.
4. Preserves CSV row order in the UI even when worker threads complete out of order.
5. Live progress reporting: completed, processing, review-required, blocked, failed counts.
6. Batch state recovery after page refresh via atomic JSON persistence.
7. Idempotent execution and duplicate submission prevention.
8. Robust partial failure isolation: individual contact errors do not terminate the batch.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.outreach.models import DraftStatus, WorkflowMode
from src.research.cache import CachedResearchProvider, ResearchCache
from src.ui.storage import (
    _atomic_write_json,
    get_batch_record,
    save_batch_record,
    save_draft_record,
)
from src.verification.models import BLOCKED
from src.workflow.engine import OutreachWorkflow, WorkflowResult

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_contacts_and_errors(import_result: Any) -> tuple[list[Any], list[Any]]:
    """Defensively extract contacts and errors from ImportResult, SpreadsheetParseResult, or mocks."""
    raw_valid = getattr(import_result, "valid_contacts", None)
    raw_contacts = getattr(import_result, "contacts", None)

    if isinstance(raw_valid, (list, tuple)) and raw_valid:
        contacts = raw_valid
    elif isinstance(raw_contacts, (list, tuple)) and raw_contacts:
        contacts = raw_contacts
    elif isinstance(raw_valid, (list, tuple)):
        contacts = raw_valid
    elif isinstance(raw_contacts, (list, tuple)):
        contacts = raw_contacts
    else:
        contacts = raw_valid or raw_contacts or []

    errors = getattr(import_result, "errors", [])
    if not isinstance(errors, (list, tuple)):
        errors = []
    return list(contacts), list(errors)


def _format_evidence_list(wf_res: WorkflowResult) -> list[dict[str, Any]]:
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
    return evidence_list


def _format_result_entry(i: int, wf_res: WorkflowResult) -> dict[str, Any]:
    draft_id = wf_res.draft.draft_id if wf_res.draft else None
    rec_res = wf_res.recipient_resolution
    evidence_list = _format_evidence_list(wf_res)

    row_num = wf_res.row_number if wf_res.row_number is not None else (i + 1)

    return {
        "row_number": row_num,
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
    }


class BackgroundBatchRunner:
    """
    Manages background batch runs without blocking Flask web requests.
    Thread-safe coordinator with incremental progress persistence.
    """

    _instance: BackgroundBatchRunner | None = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_batches: set[str] = set()

    @classmethod
    def get_instance(cls) -> BackgroundBatchRunner:
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def is_batch_active(self, batch_id: str) -> bool:
        with self._lock:
            return batch_id in self._active_batches

    def start_batch(
        self,
        batch_id: str,
        import_result: Any,
        workflow: OutreachWorkflow,
        target_role: str = "AI/ML Intern",
        job_description: str = "",
        campaign_id: str = "",
        mode: WorkflowMode = WorkflowMode.UPLOADED_CV,
        max_workers: int = 4,
        research_provider: Any = None,
        custom_output_dir: str | Path | None = None,
        wait_for_completion: bool = False,
    ) -> str:
        """
        Initialize batch record on disk and launch background execution.
        Returns batch_id immediately (or waits if wait_for_completion=True in tests).
        """
        with self._lock:
            if batch_id in self._active_batches:
                logger.warning("Batch %s is already running.", batch_id)
                return batch_id
            self._active_batches.add(batch_id)

        contacts, errors = _extract_contacts_and_errors(import_result)
        total_rows = len(contacts) + len(errors)
        errors_list: list[dict[str, Any]] = []
        for err in errors:
            errors_list.append({
                "row_number": err.row_number,
                "raw_row": getattr(err, "raw_row", {}),
                "reason": getattr(err, "reason", "Import validation error"),
            })

        # Initial batch record (marked PROCESSING)
        initial_record: dict[str, Any] = {
            "batch_id": batch_id,
            "campaign_id": campaign_id,
            "target_role": target_role,
            "mode": mode.value if hasattr(mode, "value") else str(mode),
            "status": "PROCESSING",
            "total_rows": total_rows,
            "processed_count": 0,
            "drafted_count": 0,
            "review_required_count": 0,
            "blocked_count": 0,
            "failed_count": len(errors_list),
            "confident_count": 0,
            "review_name_count": 0,
            "not_ai_relevant_count": 0,
            "eliminated_count": 0,
            "other_review_count": 0,
            "draft_generated_count": 0,
            "needs_review_count": 0,
            "generation_failed_count": 0,
            "errors": errors_list,
            "results": [],
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        save_batch_record(batch_id, initial_record)

        # Launch background thread for non-blocking processing
        thread = threading.Thread(
            target=self._execute_batch_background,
            args=(
                batch_id,
                import_result,
                workflow,
                target_role,
                job_description,
                campaign_id,
                mode,
                max_workers,
                research_provider,
                custom_output_dir,
                errors_list,
            ),
            name=f"BatchRunner-{batch_id}",
            daemon=True,
        )
        thread.start()
        if wait_for_completion:
            thread.join(timeout=60.0)
        return batch_id

    def _execute_batch_background(
        self,
        batch_id: str,
        import_result: Any,
        workflow: OutreachWorkflow,
        target_role: str,
        job_description: str,
        campaign_id: str,
        mode: WorkflowMode,
        max_workers: int,
        research_provider: Any,
        custom_output_dir: str | Path | None,
        initial_errors: list[dict[str, Any]],
    ) -> None:
        """
        Worker body executing in background thread.
        Applies bounded ThreadPoolExecutor, incremental persistence, and row ordering.
        """
        try:
            # Wrap research provider with thread-safe cache
            caching_provider = research_provider
            if not isinstance(research_provider, CachedResearchProvider):
                inner = research_provider if research_provider is not None else None
                if inner is None:
                    from src.research.providers import get_default_research_provider
                    inner = get_default_research_provider()
                caching_provider = CachedResearchProvider(inner, cache=workflow.research_cache)

            completed_items: list[tuple[int, WorkflowResult]] = []
            errors_list = list(initial_errors)
            persisted_results_map: dict[int, dict[str, Any]] = {}
            results_lock = threading.Lock()

            contacts, _ = _extract_contacts_and_errors(import_result)
            total_batch_rows = len(contacts) + len(initial_errors)

            # Contact processing task
            def _process_single(idx: int, contact: Any) -> WorkflowResult:
                recipient = contact.email_result.normalized_email

                # Check duplicate send against SQLite send_log
                if workflow.send_log.was_successfully_sent(recipient, campaign_id) or (
                    workflow.sender.dry_run and workflow.send_log.was_simulated(recipient, campaign_id)
                ):
                    return WorkflowResult(
                        mode=mode,
                        contact_email=recipient,
                        company_name=getattr(contact, "company_name", "") or "",
                        status=BLOCKED,
                        message=f"Duplicate send prevented: Email already sent to {recipient}.",
                        row_number=getattr(contact, "row_number", idx + 1),
                        contact=contact,
                    )

                try:
                    wf_res = workflow.process_contact(
                        contact=contact,
                        job_description=job_description,
                        target_role=target_role,
                        research_provider=caching_provider,
                        custom_output_dir=custom_output_dir,
                    )
                    return wf_res
                except Exception as exc:
                    logger.warning("Error processing contact %s: %s", recipient, exc)
                    return WorkflowResult(
                        mode=mode,
                        contact_email=recipient,
                        company_name=getattr(contact, "company_name", "") or "",
                        status="ERROR",
                        message=f"Error processing contact: {exc}",
                        error=str(exc),
                        row_number=getattr(contact, "row_number", idx + 1),
                        contact=contact,
                    )

            # Bounded ThreadPool execution
            worker_count = max(1, min(max_workers, 16))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_info = {
                    executor.submit(_process_single, i, contact): (i, contact)
                    for i, contact in enumerate(contacts)
                }

                for future in as_completed(future_to_info):
                    i, contact = future_to_info[future]
                    try:
                        wf_res = future.result()
                    except Exception as fatal_exc:
                        logger.error("Unhandled exception for contact %s: %s", contact, fatal_exc)
                        wf_res = WorkflowResult(
                            mode=mode,
                            contact_email=getattr(contact.email_result, "normalized_email", "unknown"),
                            company_name=getattr(contact, "company_name", "") or "",
                            status="ERROR",
                            message=f"Unexpected fatal error: {fatal_exc}",
                            error=str(fatal_exc),
                            row_number=getattr(contact, "row_number", i + 1),
                            contact=contact,
                        )

                    # Persist draft to disk if generated
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
                            classification=wf_res.classification.to_dict() if wf_res.classification else {},
                            research_evidence=_format_evidence_list(wf_res),
                        )

                    # Record errors if failed
                    if wf_res.status == "ERROR":
                        with results_lock:
                            errors_list.append({
                                "row_number": wf_res.row_number,
                                "email": wf_res.contact_email,
                                "error": wf_res.error or wf_res.message,
                            })

                    entry = _format_result_entry(i, wf_res)
                    with results_lock:
                        completed_items.append((i, wf_res))
                        persisted_results_map[i] = entry

                        # Incremental update of batch state on disk
                        self._write_incremental_state(
                            batch_id=batch_id,
                            campaign_id=campaign_id,
                            target_role=target_role,
                            mode=mode,
                            total_rows=total_batch_rows,
                            completed_items=completed_items,
                            errors_list=errors_list,
                            persisted_results_map=persisted_results_map,
                            status="PROCESSING",
                        )

            # Final write with COMPLETED status and strictly sorted row order
            with results_lock:
                self._write_incremental_state(
                    batch_id=batch_id,
                    campaign_id=campaign_id,
                    target_role=target_role,
                    mode=mode,
                    total_rows=total_batch_rows,
                    completed_items=completed_items,
                    errors_list=errors_list,
                    persisted_results_map=persisted_results_map,
                    status="COMPLETED",
                )

        except Exception as exc:
            logger.exception("Fatal batch runner crash for batch %s: %s", batch_id, exc)
            try:
                curr = get_batch_record(batch_id) or {}
                curr["status"] = "ERROR"
                curr["error"] = str(exc)
                curr["updated_at"] = _now_iso()
                save_batch_record(batch_id, curr)
            except Exception:
                pass
        finally:
            with self._lock:
                self._active_batches.discard(batch_id)

    def _write_incremental_state(
        self,
        batch_id: str,
        campaign_id: str,
        target_role: str,
        mode: WorkflowMode,
        total_rows: int,
        completed_items: list[tuple[int, WorkflowResult]],
        errors_list: list[dict[str, Any]],
        persisted_results_map: dict[int, dict[str, Any]],
        status: str,
    ) -> None:
        """Helper to write incremental or final batch record with sorted row order."""
        drafted_count = 0
        review_required_count = 0
        blocked_count = 0
        failed_count = len(errors_list)
        confident_count = 0
        review_name_count = 0
        not_ai_relevant_count = 0
        eliminated_count = 0
        other_review_count = 0
        draft_generated_count = 0
        needs_review_count = 0
        generation_failed_count = 0

        for _, wf_res in completed_items:
            if wf_res.classification is not None:
                outcome = wf_res.classification.primary_outcome
                if outcome == "CONFIDENT":
                    confident_count += 1
                elif outcome == "REVIEW_NAME":
                    review_name_count += 1
                elif outcome == "NOT_AI_RELEVANT":
                    not_ai_relevant_count += 1
                elif outcome == "ELIMINATED":
                    eliminated_count += 1
                else:
                    other_review_count += 1

            if wf_res.status == BLOCKED:
                blocked_count += 1
            elif wf_res.status in ("REVIEW", "PROCEED"):
                if wf_res.draft is not None:
                    drafted_count += 1
                    gen_st = getattr(wf_res.draft, "generation_status", "DRAFT_GENERATED")
                    if gen_st == "DRAFT_GENERATED":
                        draft_generated_count += 1
                    elif gen_st == "NEEDS_REVIEW":
                        needs_review_count += 1
                    else:
                        generation_failed_count += 1
                if wf_res.status == "REVIEW":
                    review_required_count += 1
            elif wf_res.status == "ERROR":
                generation_failed_count += 1

        # Preserve CSV row order by sorting by original index
        sorted_indices = sorted(persisted_results_map.keys())
        sorted_results = [persisted_results_map[k] for k in sorted_indices]

        record: dict[str, Any] = {
            "batch_id": batch_id,
            "campaign_id": campaign_id,
            "target_role": target_role,
            "mode": mode.value if hasattr(mode, "value") else str(mode),
            "status": status,
            "total_rows": total_rows,
            "processed_count": len(completed_items),
            "drafted_count": drafted_count,
            "review_required_count": review_required_count,
            "blocked_count": blocked_count,
            "failed_count": failed_count,
            "confident_count": confident_count,
            "review_name_count": review_name_count,
            "not_ai_relevant_count": not_ai_relevant_count,
            "eliminated_count": eliminated_count,
            "other_review_count": other_review_count,
            "draft_generated_count": draft_generated_count,
            "needs_review_count": needs_review_count,
            "generation_failed_count": generation_failed_count,
            "errors": errors_list,
            "results": sorted_results,
            "updated_at": _now_iso(),
        }
        save_batch_record(batch_id, record)
