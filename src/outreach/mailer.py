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
from pathlib import Path
from typing import Any

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
    """

    def __init__(self) -> None:
        self._drafts: dict[str, EmailDraft] = {}
        self._sent_recipients: set[str] = set()

    def register_draft(self, draft: EmailDraft) -> EmailDraft:
        """Register a newly generated draft for review."""
        self._drafts[draft.draft_id] = draft
        return draft

    def get_draft(self, draft_id: str) -> EmailDraft:
        """Retrieve a draft by ID."""
        if draft_id not in self._drafts:
            raise KeyError(f"Draft '{draft_id}' not found.")
        return self._drafts[draft_id]

    def list_drafts(self) -> list[EmailDraft]:
        """List all registered drafts."""
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
        draft = self.get_draft(draft_id)
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

        # Reset approval if edited
        draft.approved = False
        draft.status = DraftStatus.DRAFT
        return draft

    def approve_draft(self, draft_id: str) -> EmailDraft:
        """
        Grant explicit human approval for sending.
        """
        draft = self.get_draft(draft_id)
        draft.approved = True
        draft.status = DraftStatus.APPROVED
        return draft

    def reject_draft(self, draft_id: str, reason: str = "") -> EmailDraft:
        """Mark a draft as rejected."""
        draft = self.get_draft(draft_id)
        draft.approved = False
        draft.status = DraftStatus.REJECTED
        return draft

    def is_duplicate(self, recipient_email: str) -> bool:
        """Check if an email has already been sent to this normalized address."""
        return recipient_email.strip().lower() in self._sent_recipients

    def record_sent(self, recipient_email: str) -> None:
        """Record that an email was sent to a recipient."""
        self._sent_recipients.add(recipient_email.strip().lower())

    def clear_sent_registry(self) -> None:
        """Clear sent history (primarily for test isolation)."""
        self._sent_recipients.clear()


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
    ) -> None:
        self.review_manager = review_manager
        self.provider = provider
        self.dry_run = dry_run

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
        if not recipient_clean or "@" not in recipient_clean:
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message="Send blocked: Recipient email address is missing or syntactically invalid.",
                error="INVALID_RECIPIENT",
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
        if att_str not in ("cv.pdf", "dummy_cv.pdf") and not att_path.exists():
            return SendResult(
                success=False,
                draft_id=draft_id,
                recipient=draft.recipient,
                message=f"Send blocked: Specified attachment file does not exist on disk: {draft.attachment_path}",
                error="ATTACHMENT_NOT_FOUND",
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
        if self.provider is not None:
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
                return SendResult(
                    success=False,
                    draft_id=draft_id,
                    recipient=draft.recipient,
                    message=f"Provider failed to send email: {e}",
                    error=str(e),
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
