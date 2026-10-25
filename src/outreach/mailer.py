"""
src/outreach/mailer.py
----------------------
Review, approval, duplicate-send prevention, and sending safeguards.

Safeguards:
1. Emails are NEVER sent automatically; drafts require explicit approval.
2. All draft fields (recipient, subject, body, attachment) are reviewable and editable.
3. Duplicate sends to the same recipient address are strictly blocked.
4. If Gmail integration is not configured, explains required setup instead of pretending.
5. In test or simulation mode, real network transmissions are forbidden.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from src.contacts.email_parser import parse_email
from src.outreach.approval import approval_hash_for, approval_matches
from src.outreach.models import DraftStatus, EmailDraft, SendResult


class DuplicateSendError(Exception):
    """Raised when an attempt is made to send a duplicate email to the same recipient."""
    pass


class ApprovalRequiredError(Exception):
    """Raised when an attempt is made to send an unapproved draft."""
    pass


class EmailReviewManager:
    """
    Manages drafts, human review, editing, and approval tracking.
    Thread-safe implementation supporting concurrent batch processing.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._drafts: dict[str, EmailDraft] = {}
        self._sent_recipients: set[str] = set()

    def register_draft(self, draft: EmailDraft) -> EmailDraft:
        """Register a newly generated draft for review."""
        with self._lock:
            self._drafts[draft.draft_id] = draft
            return draft

    def get_draft(self, draft_id: str) -> EmailDraft:
        """Retrieve a draft by ID."""
        with self._lock:
            if draft_id not in self._drafts:
                raise KeyError(f"Draft '{draft_id}' not found.")
            return self._drafts[draft_id]

    def list_drafts(self) -> list[EmailDraft]:
        """List all registered drafts."""
        with self._lock:
            return list(self._drafts.values())

    def edit_draft(
        self,
        draft_id: str,
        recipient: str | None = None,
        recipient_name: str | None = None,
        subject: str | None = None,
        body: str | None = None,
        attachment_path: str | None = None,
    ) -> EmailDraft:
        """
        Allow the user to review and edit any draft field before approval.
        Modifying an approved draft resets its approval status to ensure re-review.
        """
        with self._lock:
            draft = self.get_draft(draft_id)
            if draft.status in (DraftStatus.SENT, DraftStatus.SIMULATED_SENT, DraftStatus.UNCERTAIN):
                raise ValueError(
                    f"Draft '{draft_id}' is already {draft.status.value} and cannot be edited."
                )
            if recipient is not None:
                draft.recipient = recipient.strip()
            if recipient_name is not None:
                draft.recipient_name = recipient_name.strip()
            if subject is not None:
                draft.subject = subject.strip()
            if body is not None:
                draft.body = body
            if attachment_path is not None:
                draft.attachment_path = attachment_path.strip()

            # Reset approval if edited, including its binding hash.
            draft.approved = False
            draft.approval_hash = ""
            draft.status = DraftStatus.DRAFT
            return draft

    def approve_draft(self, draft_id: str) -> EmailDraft:
        """
        Grant explicit human approval for sending, bound to the exact
        draft content via approval_hash (see src/outreach/approval.py).
        """
        with self._lock:
            draft = self.get_draft(draft_id)
            draft.approved = True
            draft.status = DraftStatus.APPROVED
            draft.approval_hash = approval_hash_for(
                draft.recipient, draft.subject, draft.body, draft.attachment_path,
            )
            return draft

    def reject_draft(self, draft_id: str, reason: str = "") -> EmailDraft:
        """Mark a draft as rejected."""
        with self._lock:
            draft = self.get_draft(draft_id)
            draft.approved = False
            draft.status = DraftStatus.REJECTED
            return draft

    def is_duplicate(self, recipient_email: str) -> bool:
        """Check if an email has already been sent to this normalized address."""
        with self._lock:
            return recipient_email.strip().lower() in self._sent_recipients

    def record_sent(self, recipient_email: str) -> None:
        """Record that an email was sent to a recipient."""
        with self._lock:
            self._sent_recipients.add(recipient_email.strip().lower())

    def clear_sent_registry(self) -> None:
        """Clear sent history (primarily for test isolation)."""
        with self._lock:
            self._sent_recipients.clear()


def get_sender_address(profile: dict[str, Any] | None = None) -> str:
    """
    Get configured sender email account address for draft previews and test sending context.
    Separates sender account from recipient email addresses.
    Precedence:
    1. SMARTMAIL_TEST_SENDER_EMAIL environment variable (safe local test account)
    2. GMAIL_SENDER_ADDRESS environment variable
    3. profile['personal_info']['email'] if available
    4. Safe fallback test address 'test-sender@example.com'
    """
    env_sender = os.getenv("SMARTMAIL_TEST_SENDER_EMAIL") or os.getenv("GMAIL_SENDER_ADDRESS")
    if env_sender and env_sender.strip():
        return env_sender.strip()
    if profile and isinstance(profile, dict):
        prof_email = profile.get("personal_info", {}).get("email", "")
        if prof_email and prof_email.strip():
            return prof_email.strip()
    return "test-sender@example.com"


class EmailSender:
    """
    Handles final sending validation and provider dispatch.
    Ensures explicit approval, duplicate protection, and honest configuration reporting.
    """

    def __init__(
        self,
        review_manager: EmailReviewManager,
        provider: Any = None,
        dry_run: bool = True,
        sender_address: str | None = None,
        send_log: Any = None,
    ) -> None:
        self.review_manager = review_manager
        self.provider = provider
        self.dry_run = dry_run
        self.sender_address = sender_address or get_sender_address()
        # Optional persistent send log for suppression checks. The engine
        # always wires this; direct mailer use without it skips the check
        # (the engine claim path still enforces suppression).
        self.send_log = send_log

    def is_gmail_configured(self) -> bool:
        """Check if Gmail OAuth credentials and token paths are configured."""
        creds_path = os.getenv("GMAIL_CREDENTIALS_PATH")
        token_path = os.getenv("GMAIL_TOKEN_PATH")
        if not creds_path or not token_path:
            return False
        return os.path.exists(creds_path) and os.path.exists(token_path)

    def send(self, draft_id: str) -> SendResult:
        """
        Send an email draft following all strict safeguards.

        Safeguard order:
        1. Draft field validation (recipient, subject, body, attachment presence and existence).
        2. Duplicate send prevention.
        3. Explicit human approval verification.
        4. DRY-RUN SAFETY: If dry_run is True, network transmission is prevented
           regardless of whether an external provider is injected. Status is marked
           as SIMULATED_SENT.
        5. Live transmission safety: Live transmission remains strictly disabled
           unless an explicit provider is configured and dry_run=False.
        """
        draft = self.review_manager.get_draft(draft_id)

        # Safeguard 1: Field validation
        recipient_clean = (draft.recipient or "").strip()
        parsed_email = parse_email(recipient_clean)
        if not parsed_email.is_syntax_valid:
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message="Send blocked: Recipient email address is missing or syntactically invalid.",
                error="INVALID_RECIPIENT",
            )
        # Suppression check (when a persistent log is wired in): an opted-out
        # recipient is blocked even in dry-run mode.
        if self.send_log is not None and self.send_log.is_suppressed(recipient_clean):
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message=f"Send blocked: {draft.recipient} is on the suppression/opt-out list.",
                error="SUPPRESSED",
            )
        if not (draft.subject or "").strip():
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message="Send blocked: Email subject cannot be empty.",
                error="EMPTY_SUBJECT",
            )
        if not (draft.body or "").strip():
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message="Send blocked: Email body cannot be empty.",
                error="EMPTY_BODY",
            )
        if not draft.attachment_path or not str(draft.attachment_path).strip():
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message="Send blocked: An attachment path is required before sending.",
                error="MISSING_ATTACHMENT",
            )
        att_str = str(draft.attachment_path).strip()
        att_path = Path(att_str)
        if not att_path.exists():
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message=f"Send blocked: Specified attachment file does not exist on disk: {draft.attachment_path}",
                error="ATTACHMENT_NOT_FOUND",
            )
        if not att_path.is_file():
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message=f"Send blocked: Specified attachment is not a regular file: {draft.attachment_path}",
                error="INVALID_ATTACHMENT",
            )
        if att_path.suffix.lower() != ".pdf":
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message=f"Send blocked: Attachment must be a PDF file (.pdf extension required): {draft.attachment_path}",
                error="INVALID_ATTACHMENT",
            )

        # Verify readability and PDF magic header
        try:
            with open(att_path, "rb") as f:
                header = f.read(5)
                if header != b"%PDF-":
                    return SendResult(
                        success=False,
                        draft_id=draft_id,
                        recipient=draft.recipient,
                        message=f"Send blocked: Attachment is not a valid PDF (missing %PDF- header): {draft.attachment_path}",
                        error="INVALID_ATTACHMENT",
                    )
        except (OSError, PermissionError) as err:
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message=f"Send blocked: Attachment file is unreadable: {err}",
                error="INVALID_ATTACHMENT",
            )

        # Verify parseable PDF structure and page count
        try:
            import pypdf
            reader = pypdf.PdfReader(str(att_path))
            if len(reader.pages) == 0:
                return SendResult(
                    success=False,
                    draft_id=draft_id,
                    recipient=draft.recipient,
                    message="Send blocked: Attachment PDF contains no pages.",
                    error="INVALID_ATTACHMENT",
                )
            if reader.is_encrypted:
                return SendResult(
                    success=False,
                    draft_id=draft_id,
                    recipient=draft.recipient,
                    message="Send blocked: Attachment PDF is encrypted.",
                    error="INVALID_ATTACHMENT",
                )
        except Exception as err:
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message=f"Send blocked: Attachment is corrupted or not a parseable PDF: {err}",
                error="INVALID_ATTACHMENT",
            )

        # Safeguard 2: Duplicate send prevention
        if self.review_manager.is_duplicate(draft.recipient):
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message=f"Send blocked: An email has already been sent to {draft.recipient}.",
                error="DUPLICATE_RECIPIENT",
            )

        # Safeguard 3: Explicit human approval required
        if not draft.approved or draft.status != DraftStatus.APPROVED:
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message="Send blocked: Explicit user approval is required before sending.",
                error="APPROVAL_REQUIRED",
            )

        # Safeguard 3b: Approval binding. If this draft was approved under
        # the hash scheme, the hash must still match the current content.
        # A mismatch means the draft was edited after approval: the old
        # approval is invalid and the draft needs re-review. Drafts
        # approved before this feature (empty hash) are grandfathered in
        # for this check only; new approvals always carry a hash.
        if draft.approval_hash and not approval_matches(
            draft.approval_hash, draft.recipient, draft.subject, draft.body,
            draft.attachment_path,
        ):
            draft.approved = False
            draft.approval_hash = ""
            draft.status = DraftStatus.DRAFT
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message="Send blocked: Draft was edited after approval; approval invalidated. Re-review and re-approve the exact draft.",
                error="APPROVAL_INVALIDATED",
            )

        # Safeguard 4: DRY RUN SAFETY CHECK (MUST precede any provider call)
        # Prevents real network transmission regardless of whether a provider is injected.
        if self.dry_run:
            self.review_manager.record_sent(draft.recipient)
            draft.status = DraftStatus.SIMULATED_SENT
            return SendResult(
                success=True,
                draft_id=draft_id,
                recipient=draft.recipient,
                message="Simulated send succeeded (DRY RUN mode — network transmission prevented).",
            )

        # Safeguard 5: Non-dry-run provider execution (strictly for tests/mock providers)
        # Bounded retries: transient failures (timeout/connection) are
        # retried at most twice (3 attempts total). Permanent failures are
        # never retried. A timeout on the final attempt is UNCERTAIN -- the
        # provider may have accepted the email -- and must be reconciled
        # by a human, never blindly retried. A DB claim is NOT a guarantee
        # of exactly-once delivery.
        if self.provider is not None:
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    self.provider.send_email(
                        recipient=draft.recipient,
                        subject=draft.subject,
                        body=draft.body,
                        attachment_path=draft.attachment_path,
                    )
                    self.review_manager.record_sent(draft.recipient)
                    draft.status = DraftStatus.SENT
                    return SendResult(
                        success=True,
                        draft_id=draft_id,
                        recipient=draft.recipient,
                        message="Email sent successfully via injected provider.",
                    )
                except Exception as e:
                    last_exc = e
                    err_lower = str(e).lower()
                    is_transient = any(kw in err_lower for kw in ("timeout", "timed out", "connection reset", "broken pipe", "connection error", "temporarily"))
                    if not is_transient or attempt == 2:
                        err_code = "UNCERTAIN_SEND" if is_transient else "PROVIDER_ERROR"
                        return SendResult(
                            success=False,
                            draft_id=draft_id,
                            recipient=draft.recipient,
                            message=f"Provider failed to send email: {e}",
                            error=err_code,
                        )
                    continue
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message=f"Provider failed to send email: {last_exc}",
                error="PROVIDER_ERROR",
            )

        # Check Gmail configuration honestly if in non-dry-run mode without custom provider
        if not self.is_gmail_configured():
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message=(
                    "Gmail integration is not configured. Setup required: "
                    "Set GMAIL_CREDENTIALS_PATH and GMAIL_TOKEN_PATH environment variables "
                    "pointing to valid Google OAuth client secrets and token files."
                ),
                error="GMAIL_NOT_CONFIGURED",
            )

        # Live Gmail sending remains strictly disabled
        return SendResult(
            success=False,
            draft_id=draft_id,
            recipient=draft.recipient,
            message="Live email transmission is disabled in SmartMail.",
            error="LIVE_SENDING_DISABLED",
        )
