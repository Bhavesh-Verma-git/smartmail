"""
src/outreach/models.py
----------------------
Data models and status enums for email drafting, review, and sending safeguards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class WorkflowMode(str, Enum):
    """Selectable workflow modes in the application."""
    UPLOADED_CV = "uploaded_cv"   # Default and primary mode
    TAILORED_CV = "tailored_cv"   # Optional secondary mode


class DraftStatus(str, Enum):
    """Lifecycle status of an outreach draft."""
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    SENT = "SENT"
    SIMULATED_SENT = "SIMULATED_SENT"
    REJECTED = "REJECTED"


class GenerationStatus(str, Enum):
    """Stage 4 draft generation outcome."""
    DRAFT_GENERATED = "DRAFT_GENERATED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    GENERATION_FAILED = "GENERATION_FAILED"


@dataclass
class EmailDraft:
    """
    Structured outreach draft for human review and approval.

    Attributes:
        draft_id: Unique identifier for the draft.
        recipient: Email address of the recipient.
        recipient_name: Optional name of the recipient / contact.
        company_name: Name of the target company.
        subject: Email subject line.
        body: Plain-text email body.
        attachment_path: File path of the attached CV (original or tailored).
        mode: The WorkflowMode used to generate this draft.
        status: Current status (DRAFT, APPROVED, SENT, REJECTED).
        approved: True only if explicit user approval was granted.
        review_required: True if company research, domain, or role matching returned REVIEW.
        review_flags: Warnings or notes requiring human attention.
        tailored_cv_path: Path to the generated CV if tailored mode was used.
        created_at: ISO timestamp of creation.
        generation_status: Outcome of generation (DRAFT_GENERATED, NEEDS_REVIEW, GENERATION_FAILED).
        personalization_explanation: Concise explanation of evidence connecting candidate to company.
        evidence_summary: Extracted summary of company AI/ML work.
        matched_skills: Candidate skills matched to company work.
        matched_projects: Specific CV projects highlighted in the email.
        recipient_type: Categorization of recipient (individual, generic_inbox, unknown).
    """
    draft_id: str
    recipient: str
    recipient_name: str
    company_name: str
    subject: str
    body: str
    attachment_path: str
    mode: WorkflowMode
    status: DraftStatus = DraftStatus.DRAFT
    approved: bool = False
    review_required: bool = False
    review_flags: list[str] = field(default_factory=list)
    tailored_cv_path: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    generation_status: str = "DRAFT_GENERATED"
    personalization_explanation: str = ""
    evidence_summary: str = ""
    matched_skills: list[str] = field(default_factory=list)
    matched_projects: list[str] = field(default_factory=list)
    recipient_type: str = "individual"



@dataclass
class SendResult:
    """
    Result of an attempt to send an email.

    Attributes:
        success: Whether the send succeeded.
        draft_id: The ID of the targeted draft.
        recipient: Recipient email address.
        message: Human-readable message or explanation.
        error: Specific error details if unsuccessful.
        timestamp: ISO timestamp of the attempt.
    """
    success: bool
    draft_id: str
    recipient: str
    message: str
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
