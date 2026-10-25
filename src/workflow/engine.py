"""
src/workflow/engine.py
----------------------
end-to-end workflow orchestrator for the uploaded-CV-only workflow.
"""
from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.contacts.csv_importer import ContactRecord
from src.outreach.drafting import draft_outreach_email
from src.outreach.mailer import EmailReviewManager, EmailSender
from src.outreach.models import DraftStatus, EmailDraft, SendResult, WorkflowMode
from src.outreach.send_log import SendLog
from src.profile.loader import load_profile
from src.profile.validator import assert_valid_profile
from src.research.cache import CachedResearchProvider, ResearchCache
from src.research.checks import _is_http_url
from src.research.models import ResearchInput
from src.research.researcher import build_research_input, research_contact
from src.verification.classifier import ContactClassification, classify_contact
from src.verification.models import BLOCKED, PipelineResult, PROCEED
from src.verification.pipeline import run_verification


@dataclass
class WorkflowResult:
    """Outcome of processing a contact through the outreach workflow."""
    mode: WorkflowMode
    contact_email: str
    company_name: str
    status: str  # "PROCEED", "REVIEW", "BLOCKED", "ERROR"
    draft: EmailDraft | None = None
    verification: PipelineResult | None = None
    message: str = ""
    error: str | None = None
    # Stage 3 additions
    row_number: int | None = None
    contact: Any | None = None
    research_result: Any | None = None
    classification: ContactClassification | None = None
    recipient_resolution: Any | None = None
    cross_campaign_warning: str = ""


@dataclass
class BatchResult:
    """Aggregated outcome of processing a batch of contacts via process_csv."""
    total_rows: int = 0
    processed_count: int = 0
    drafted_count: int = 0
    review_required_count: int = 0
    blocked_count: int = 0
    failed_count: int = 0
    results: list[WorkflowResult] = field(default_factory=list)
    review_queue: list[WorkflowResult] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    # Stage 3 classification counts
    confident_count: int = 0
    review_name_count: int = 0
    not_ai_relevant_count: int = 0
    eliminated_count: int = 0
    other_review_count: int = 0
    # Stage 4 draft generation counts
    draft_generated_count: int = 0
    needs_review_count: int = 0
    generation_failed_count: int = 0

    def summary(self) -> str:
        return (
            f"Batch Processing Summary:\n"
            f"  Total input rows:      {self.total_rows}\n"
            f"  Processed contacts:    {self.processed_count}\n"
            f"  Drafts created:        {self.drafted_count}\n"
            f"  Review required:       {self.review_required_count}\n"
            f"  Blocked contacts:      {self.blocked_count}\n"
            f"  Failed / Invalid rows: {self.failed_count}\n"
            f"  Review queue size:     {len(self.review_queue)}"
        )


class OutreachWorkflow:
    """
    Unified workflow controller orchestrating research, verification,
    uploaded-CV handling, drafting, and review.
    """

    def __init__(
        self,
        profile_path: str | Path = "data/master_profile.json",
        default_mode: WorkflowMode = WorkflowMode.UPLOADED_CV,
        uploaded_cv_path: str | Path | None = None,
        email_sender_provider: Any = None,
        dry_run: bool = True,
        send_log: SendLog | None = None,
        campaign_id: str = "",
    ) -> None:
        self.profile_path = Path(profile_path)
        self.profile: dict[str, Any] = {}
        if self.profile_path.exists():
            self.profile = load_profile(self.profile_path)
            # Ensure the master profile is valid
            assert_valid_profile(self.profile)

        self._lock = threading.RLock()
        self.research_cache: ResearchCache = ResearchCache()
        if default_mode != WorkflowMode.UPLOADED_CV:
            raise ValueError("Tailored-CV mode was removed. Only uploaded-CV mode is supported.")
        self._mode = default_mode
        self._uploaded_cv_path: Path | None = None
        self._uploaded_cv_hash: str | None = None
        self._uploaded_cv_original_bytes: bytes | None = None
        self._cached_cv_text: str | None = None

        if uploaded_cv_path is not None:
            self.set_uploaded_cv(uploaded_cv_path)

        self.review_manager = EmailReviewManager()
        self.sender = EmailSender(
            review_manager=self.review_manager,
            provider=email_sender_provider,
            dry_run=dry_run,
        )
        # Persistent send log — default to SQLite; tests may pass in-memory instance
        self.send_log: SendLog = send_log or SendLog()
        self.sender.send_log = self.send_log
        self.campaign_id = campaign_id


    # -----------------------------------------------------------------------
    # Mode Management
    # -----------------------------------------------------------------------

    @property
    def mode(self) -> WorkflowMode:
        """Current workflow mode."""
        return self._mode

    def set_mode(self, mode: WorkflowMode) -> None:
        """Set the workflow mode. Only the uploaded-CV mode is supported.

        The tailored-CV mode was removed; passing it raises instead of
        silently falling back, so a legacy request can never be mistaken
        for an uploaded-CV run.
        """
        if mode != WorkflowMode.UPLOADED_CV:
            raise ValueError(
                "Tailored-CV mode was removed. Only uploaded-CV mode is supported; "
                "legacy tailored drafts are blocked, never converted."
            )
        with self._lock:
            self._mode = mode

    # -----------------------------------------------------------------------
    # Uploaded CV Safeguards
    # -----------------------------------------------------------------------

    def set_uploaded_cv(self, file_path: str | Path) -> None:
        """
        Register the existing CV file for Uploaded CV Mode.
        Computes SHA-256 hash and caches bytes to guarantee byte-for-byte preservation.
        Pre-caches text representation thread-safely.
        """
        with self._lock:
            p = Path(file_path).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Uploaded CV file not found at: {file_path}")
            
            file_bytes = p.read_bytes()
            self._uploaded_cv_path = p
            self._uploaded_cv_original_bytes = file_bytes
            self._uploaded_cv_hash = hashlib.sha256(file_bytes).hexdigest()
            self._cached_cv_text = self._extract_cv_text_from_path(p)

    @property
    def uploaded_cv_path(self) -> Path | None:
        """Path to the registered uploaded CV."""
        return self._uploaded_cv_path

    def verify_uploaded_cv_unmodified(self) -> bool:
        """
        Assert that the uploaded CV file remains byte-for-byte identical to its original state.
        """
        if self._uploaded_cv_path is None or self._uploaded_cv_original_bytes is None:
            return True
        current_bytes = self._uploaded_cv_path.read_bytes()
        current_hash = hashlib.sha256(current_bytes).hexdigest()
        return (
            current_bytes == self._uploaded_cv_original_bytes
            and current_hash == self._uploaded_cv_hash
        )

    def _extract_cv_text_from_path(self, p: Path) -> str:
        text = ""
        try:
            if p.suffix.lower() == ".pdf":
                import pypdf
                reader = pypdf.PdfReader(str(p))
                pages_text = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(pages_text)
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        return text

    def get_uploaded_cv_text(self) -> str:
        """
        Extract text from the registered uploaded CV for dynamic skill matching.
        Does not alter the file. Caches text in memory thread-safely.
        """
        with self._lock:
            if not self._uploaded_cv_path or not self._uploaded_cv_path.exists():
                return ""
            if self._cached_cv_text is not None:
                return self._cached_cv_text

            self._cached_cv_text = self._extract_cv_text_from_path(self._uploaded_cv_path)
            return self._cached_cv_text

    # -----------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------

    def process_contact(
        self,
        contact: ContactRecord,
        job_description: str = "",
        target_role: str = "",
        research_provider: Any = None,
        custom_output_dir: str | Path | None = None,
    ) -> WorkflowResult:
        """
        Execute the workflow for a single contact record.
        """
        company_name = (contact.company_name or "").strip()
        recipient_email = contact.email_result.normalized_email
        contact_name = contact.contact_name
        website = contact.website
        # Determine job URL vs role title
        contact_role_raw = contact.role_or_job_url.strip() if contact.role_or_job_url else ""
        target_role_raw = target_role.strip() if target_role else ""

        # Extract genuine HTTP/HTTPS job URL if either field provides one
        job_url = ""
        if _is_http_url(contact_role_raw):
            job_url = contact_role_raw
        elif _is_http_url(target_role_raw):
            job_url = target_role_raw

        # Extract plain-text role title context (e.g. "AI/ML Intern")
        effective_role = ""
        if target_role_raw and not _is_http_url(target_role_raw):
            effective_role = target_role_raw
        elif contact_role_raw and not _is_http_url(contact_role_raw):
            effective_role = contact_role_raw

        # 1. Company Research (Stage 3)
        res_input = build_research_input(contact)
        # Ensure role_or_job_url only contains a genuine HTTP/HTTPS URL
        res_input.role_or_job_url = job_url
        if research_provider is not None and hasattr(research_provider, "research"):
            research_result = research_provider.research(res_input)
        else:
            research_result = research_contact(res_input, provider=research_provider)

        # Resolve company name from domain research if not explicitly supplied in contact record
        if not company_name and getattr(research_result, "resolved_company_name", ""):
            company_name = research_result.resolved_company_name

        # 2. Company & Contact Verification + Role Relevance (Stage 4)
        cv_text = self.get_uploaded_cv_text()
        verification_result = run_verification(
            research_result=research_result,
            research_input=res_input,
            profile=self.profile,
            cv_text=cv_text,
        )

        overall_status = verification_result.verification.overall_status

        # 3. Stage 3 Contact Classification
        rec_res = getattr(research_result, "recipient_resolution", None)
        contact_classification = classify_contact(
            research_result=research_result,
            verification_result=verification_result,
            recipient_resolution=rec_res,
            row_number=getattr(contact, "row_number", None),
            original_email=contact.email,
        )

        # SAFEGUARD: BLOCKED verification stops the workflow immediately.
        # Relevance NEVER overrides a BLOCKED verification.
        if overall_status == BLOCKED:
            reasons = "; ".join(verification_result.verification.blocking_reasons)
            return WorkflowResult(
                mode=self._mode,
                contact_email=recipient_email,
                company_name=company_name,
                status=BLOCKED,
                verification=verification_result,
                message=f"Workflow halted: Verification failed and marked as BLOCKED ({reasons}).",
                row_number=getattr(contact, "row_number", None),
                contact=contact,
                research_result=research_result,
                classification=contact_classification,
                recipient_resolution=rec_res,
            )

        # 3. Uploaded-CV handling (the only supported mode)
        attachment_path = ""

        if self._mode == WorkflowMode.UPLOADED_CV:
            # UPLOADED CV MODE RULES:
            # 1. Must have an uploaded CV registered.
            if not self._uploaded_cv_path or not self._uploaded_cv_path.exists():
                raise RuntimeError(
                    "Uploaded CV Mode requires an existing CV file. Call set_uploaded_cv() first."
                )
            # 2. Verify file remains byte-for-byte unmodified.
            if not self.verify_uploaded_cv_unmodified():
                raise RuntimeError("Uploaded CV was unexpectedly altered!")

            attachment_path = str(self._uploaded_cv_path.resolve())

        # 4. Draft Outreach Email (Stage 4 Personalized Drafting)
        draft = draft_outreach_email(
            profile=self.profile,
            recipient_email=recipient_email,
            company_name=company_name,
            contact_name=contact_name,
            target_role=effective_role,
            job_description=job_description,
            verification_pipeline_result=verification_result,
            mode=self._mode,
            attachment_path=attachment_path,
            contact_notes=contact.notes,
            classification=contact_classification,
            recipient_resolution=rec_res,
            research_result=research_result,
            cv_text=cv_text,
        )

        # Register draft for human review + persist to send log
        self.review_manager.register_draft(draft)
        self.send_log.record_draft(
            draft_id=draft.draft_id,
            recipient=recipient_email,
            campaign_id=self.campaign_id,
        )

        # 5. Final check on uploaded CV preservation (explicit error, not
        # a bare assert, so it still fires under `python -O`).
        if self._uploaded_cv_path and not self.verify_uploaded_cv_unmodified():
            raise RuntimeError("Uploaded CV modified during workflow execution!")

        return WorkflowResult(
            mode=self._mode,
            contact_email=recipient_email,
            company_name=company_name,
            status=overall_status,
            draft=draft,
            verification=verification_result,
            message="Outreach draft generated and awaiting human review.",
            row_number=getattr(contact, "row_number", None),
            contact=contact,
            research_result=research_result,
            classification=contact_classification,
            recipient_resolution=rec_res,
        )

    # -----------------------------------------------------------------------
    # Review & Sending Delegates
    # -----------------------------------------------------------------------

    def get_draft(self, draft_id: str) -> EmailDraft:
        """Fetch draft for inspection."""
        return self.review_manager.get_draft(draft_id)

    def edit_draft(
        self,
        draft_id: str,
        recipient: str | None = None,
        recipient_name: str | None = None,
        subject: str | None = None,
        body: str | None = None,
        attachment_path: str | None = None,
    ) -> EmailDraft:
        """Edit draft fields before approval."""
        return self.review_manager.edit_draft(
            draft_id=draft_id,
            recipient=recipient,
            recipient_name=recipient_name,
            subject=subject,
            body=body,
            attachment_path=attachment_path,
        )

    def approve_draft(self, draft_id: str) -> EmailDraft:
        """Approve draft for sending and record in persistent log."""
        draft = self.review_manager.approve_draft(draft_id)
        self.send_log.record_approved(
            draft_id=draft_id,
            recipient=draft.recipient,
            campaign_id=self.campaign_id,
        )
        return draft

    def send_draft(self, draft_id: str) -> SendResult:
        """Attempt to send an approved draft through safeguards and atomic reservation."""
        draft = self.review_manager.get_draft(draft_id)

        # Future-live gates (dry-run unaffected): live transmission must
        # be deliberately enabled and not kill-switched, and must stay
        # under the conservative daily cap. These run BEFORE claiming so a
        # blocked live attempt never reserves the draft.
        if not self.sender.dry_run:
            from src.outreach.compliance import daily_send_cap, live_sending_enabled
            from datetime import datetime, timezone as _tz
            # The live gate targets the REAL Gmail provider only. Injected
            # providers are test doubles (the mailer documents them as
            # test-only); blocking them would break the mock live-send
            # path tests rely on. With no provider at all, the mailer
            # itself reports GMAIL_NOT_CONFIGURED / LIVE_SENDING_DISABLED
            # honestly and never transmits, so it is left to the mailer.
            _prov = getattr(self.sender, "provider", None)
            _is_real_gmail = (
                _prov is not None
                and type(_prov).__name__ == "GmailProvider"
            )
            if _is_real_gmail and not live_sending_enabled():
                return SendResult(
                    success=False, draft_id=draft_id, recipient=draft.recipient,
                    message="Live email transmission is disabled in SmartMail.",
                    error="LIVE_SENDING_DISABLED",
                )
            cap = daily_send_cap()
            today = datetime.now(_tz.utc).strftime("%Y-%m-%d")
            if self.send_log.count_success_on_day(today) >= cap:
                return SendResult(
                    success=False, draft_id=draft_id, recipient=draft.recipient,
                    message=f"Daily live-send cap reached ({cap} confirmed sends today).",
                    error="SEND_CAP_REACHED",
                )

        # Atomic reservation / claim against persistent SQLite send_log
        claimed, claim_reason = self.send_log.claim_for_sending(
            draft_id=draft_id,
            recipient=draft.recipient,
            campaign_id=self.campaign_id,
            dry_run=self.sender.dry_run,
        )
        if not claimed:
            msg_map = {
                "DUPLICATE_RECIPIENT": f"Send blocked: An email has already been sent to {draft.recipient} (persistent send log).",
                "ALREADY_SENT": f"Send blocked: An email has already been sent to {draft.recipient} (persistent send log).",
                "UNCERTAIN_SEND": f"Send blocked: A previous send to {draft.recipient} had an uncertain outcome. Manual reconciliation required before resending.",
                "SEND_IN_PROGRESS": f"Send blocked: A send attempt to {draft.recipient} is currently in progress.",
                "SUPPRESSED": f"Send blocked: {draft.recipient} is on the suppression/opt-out list.",
                "CONTENTION_ERROR": f"Send blocked: Database lock contention while attempting to reserve draft. Please retry.",
            }
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message=msg_map.get(claim_reason, f"Send blocked: {claim_reason}."),
                error=claim_reason,
            )

        result = self.sender.send(draft_id)
        if result.success:
            if self.sender.dry_run:
                self.send_log.record_simulated(
                    draft_id=draft_id,
                    recipient=draft.recipient,
                    campaign_id=self.campaign_id,
                )
            else:
                self.send_log.record_success(
                    draft_id=draft_id,
                    recipient=draft.recipient,
                    campaign_id=self.campaign_id,
                )
        elif result.error == "APPROVAL_INVALIDATED":
            # The draft changed after approval: drop the claim and leave
            # the draft unapproved for re-review (mailer already cleared
            # its in-memory approval and hash).
            self.send_log.release_claim(draft_id=draft_id, status="DRAFT")
        elif result.error == "UNCERTAIN_SEND":
            self.send_log.record_uncertain(
                draft_id=draft_id,
                recipient=draft.recipient,
                campaign_id=self.campaign_id,
            )
        elif result.error in (
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
            "SUPPRESSED",
            "SEND_CAP_REACHED",
        ):
            status_restore = "APPROVED" if draft.approved else "DRAFT"
            self.send_log.release_claim(draft_id=draft_id, status=status_restore)
        else:
            self.send_log.record_failed(
                draft_id=draft_id,
                recipient=draft.recipient,
                campaign_id=self.campaign_id,
            )
        return result

    # -----------------------------------------------------------------------
    # Batch Processing
    # -----------------------------------------------------------------------

    def process_csv(
        self,
        csv_path: str | Path,
        job_description: str = "",
        target_role: str = "",
        research_provider: Any = None,
        custom_output_dir: str | Path | None = None,
        dry_run: bool | None = None,
        max_workers: int = 4,
        **kwargs: Any,
    ) -> BatchResult:
        """
        Process an arbitrary CSV file of contacts in batch.
        Delegates to process_import_result.
        """
        from src.contacts.csv_importer import import_contacts_from_csv

        if dry_run is not None:
            self.sender.dry_run = dry_run

        import_result = import_contacts_from_csv(csv_path)
        return self.process_import_result(
            import_result=import_result,
            job_description=job_description,
            target_role=target_role,
            research_provider=research_provider,
            custom_output_dir=custom_output_dir,
            dry_run=dry_run,
            max_workers=max_workers,
            **kwargs,
        )

    def process_import_result(
        self,
        import_result: Any,
        job_description: str = "",
        target_role: str = "",
        research_provider: Any = None,
        custom_output_dir: str | Path | None = None,
        dry_run: bool | None = None,
        max_workers: int = 4,
        **kwargs: Any,
    ) -> BatchResult:
        """
        Process an arbitrary ImportResult (from CSV, XLSX, or XLS) in batch.

        - Records malformed/duplicate rows in failed_count and errors.
        - Uses bounded concurrency with ThreadPoolExecutor (default 4 workers).
        - Caches public company research safely via CachedResearchProvider.
        - Preserves strict CSV row order in the output.
        - Continues processing subsequent contacts if one contact fails.
        - Enforces duplicate prevention across persistent send_log.
        - Gathers review queue and separate counts.
        - Never automatically sends emails.
        """
        if dry_run is not None:
            self.sender.dry_run = dry_run

        raw_valid = getattr(import_result, "valid_contacts", None)
        raw_contacts = getattr(import_result, "contacts", None)
        if isinstance(raw_valid, (list, tuple)) and raw_valid:
            contacts = list(raw_valid)
        elif isinstance(raw_contacts, (list, tuple)) and raw_contacts:
            contacts = list(raw_contacts)
        elif isinstance(raw_valid, (list, tuple)):
            contacts = list(raw_valid)
        elif isinstance(raw_contacts, (list, tuple)):
            contacts = list(raw_contacts)
        else:
            contacts = list(raw_valid or raw_contacts or [])

        raw_errors = getattr(import_result, "errors", [])
        errors = list(raw_errors) if isinstance(raw_errors, (list, tuple)) else []

        batch = BatchResult(
            total_rows=len(contacts) + len(errors)
        )

        # Record import errors (syntax-invalid, duplicate within CSV, missing email)
        for err in errors:
            batch.failed_count += 1
            batch.errors.append({
                "row_number": err.row_number,
                "raw_row": getattr(err, "raw_row", {}),
                "reason": getattr(err, "reason", "Import validation error"),
            })

        caching_provider = research_provider
        if not isinstance(research_provider, CachedResearchProvider):
            inner = research_provider if research_provider is not None else None
            if inner is None:
                from src.research.providers import get_default_research_provider
                inner = get_default_research_provider()
            caching_provider = CachedResearchProvider(inner, cache=self.research_cache)

        def _process_one_contact(idx: int, contact: Any) -> WorkflowResult:
            recipient = contact.email_result.normalized_email

            # Check if this recipient was already sent an email in this campaign
            if self.send_log.was_successfully_sent(recipient, self.campaign_id) or (
                self.sender.dry_run and self.send_log.was_simulated(recipient, self.campaign_id)
            ):
                return WorkflowResult(
                    mode=self._mode,
                    contact_email=recipient,
                    company_name=getattr(contact, "company_name", "") or "",
                    status=BLOCKED,
                    message=f"Duplicate send prevented: Email already sent to {recipient}.",
                    row_number=getattr(contact, "row_number", idx + 1),
                    contact=contact,
                )

            # Suppression blocks drafting entirely for this contact.
            if self.send_log.is_suppressed(recipient):
                return WorkflowResult(
                    mode=self._mode,
                    contact_email=recipient,
                    company_name=getattr(contact, "company_name", "") or "",
                    status=BLOCKED,
                    message=f"Send prevented: {recipient} is on the suppression/opt-out list.",
                    row_number=getattr(contact, "row_number", idx + 1),
                    contact=contact,
                )

            try:
                wf_res = self.process_contact(
                    contact=contact,
                    job_description=job_description,
                    target_role=target_role,
                    research_provider=caching_provider,
                    custom_output_dir=custom_output_dir,
                )
                if wf_res.draft and self.send_log.has_any_campaign_send(recipient):
                    wf_res.cross_campaign_warning = (
                        f"Recipient {recipient} was already emailed under a different "
                        "campaign. Duplicate protection is campaign-scoped; review "
                        "before sending again."
                    )
                return wf_res
            except Exception as exc:
                return WorkflowResult(
                    mode=self._mode,
                    contact_email=recipient,
                    company_name=getattr(contact, "company_name", "") or "",
                    status="ERROR",
                    message=f"Error processing contact: {exc}",
                    error=str(exc),
                    row_number=getattr(contact, "row_number", idx + 1),
                    contact=contact,
                )

        completed_items: list[tuple[int, WorkflowResult]] = []
        worker_count = max(1, min(max_workers, 16))

        if worker_count <= 1 or len(contacts) <= 1:
            for idx, contact in enumerate(contacts):
                completed_items.append((idx, _process_one_contact(idx, contact)))
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_idx = {
                    executor.submit(_process_one_contact, i, c): i
                    for i, c in enumerate(contacts)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        res = future.result()
                    except Exception as fatal_exc:
                        contact_ref = contacts[idx]
                        res = WorkflowResult(
                            mode=self._mode,
                            contact_email=getattr(contact_ref.email_result, "normalized_email", "unknown"),
                            company_name=getattr(contact_ref, "company_name", "") or "",
                            status="ERROR",
                            message=f"Unexpected error: {fatal_exc}",
                            error=str(fatal_exc),
                            row_number=getattr(contact_ref, "row_number", idx + 1),
                            contact=contact_ref,
                        )
                    completed_items.append((idx, res))

        # Preserve exact CSV row order
        completed_items.sort(key=lambda x: x[0])

        for _, wf_res in completed_items:
            batch.processed_count += 1
            batch.results.append(wf_res)

            # Track Stage 3 classification outcomes
            if wf_res.classification is not None:
                outcome = wf_res.classification.primary_outcome
                if outcome == "CONFIDENT":
                    batch.confident_count += 1
                elif outcome == "REVIEW_NAME":
                    batch.review_name_count += 1
                elif outcome == "NOT_AI_RELEVANT":
                    batch.not_ai_relevant_count += 1
                elif outcome == "ELIMINATED":
                    batch.eliminated_count += 1
                else:
                    batch.other_review_count += 1

            if wf_res.status == BLOCKED:
                batch.blocked_count += 1
            elif wf_res.status in ("REVIEW", "PROCEED"):
                if wf_res.draft is not None:
                    batch.drafted_count += 1
                    gen_st = getattr(wf_res.draft, "generation_status", "DRAFT_GENERATED")
                    if gen_st == "DRAFT_GENERATED":
                        batch.draft_generated_count += 1
                    elif gen_st == "NEEDS_REVIEW":
                        batch.needs_review_count += 1
                    else:
                        batch.generation_failed_count += 1
                if wf_res.status == "REVIEW":
                    batch.review_required_count += 1
                    batch.review_queue.append(wf_res)
            elif wf_res.status == "ERROR":
                batch.failed_count += 1
                batch.generation_failed_count += 1
                batch.errors.append({
                    "row_number": wf_res.row_number,
                    "email": wf_res.contact_email,
                    "error": wf_res.error or wf_res.message,
                })

        return batch
