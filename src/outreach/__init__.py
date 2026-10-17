"""
src/outreach
------------
Outreach email drafting, review, and sending package.
"""
from src.outreach.drafting import build_subject_line, draft_outreach_email
from src.outreach.mailer import (
    ApprovalRequiredError,
    DuplicateSendError,
    EmailReviewManager,
    EmailSender,
)
from src.outreach.models import DraftStatus, EmailDraft, SendResult, WorkflowMode

__all__ = [
    "WorkflowMode",
    "DraftStatus",
    "EmailDraft",
    "SendResult",
    "build_subject_line",
    "draft_outreach_email",
    "EmailReviewManager",
    "EmailSender",
    "ApprovalRequiredError",
    "DuplicateSendError",
]
