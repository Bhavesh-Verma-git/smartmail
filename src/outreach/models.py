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
    # LEGACY ONLY: value kept so stored drafts from the removed tailored-CV
    # mode can be detected and blocked (never silently converted).
    # It is NOT a supported workflow mode; OutreachWorkflow rejects it.
    TAILORED_CV = "tailored_cv"


class DraftStatus(str, Enum):
    """Lifecycle status of an outreach draft."""
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    SENT = "SENT"
    SIMULATED_SENT = "SIMULATED_SENT"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"


class GenerationStatus(str, Enum):
    """Stage 4 draft generation outcome."""
    DRAFT_GENERATED = "DRAFT_GENERATED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    GENERATION_FAILED = "GENERATION_FAILED"


@dataclass
class PersonalizationRecord:
    """
    Internal audit record detailing why an outreach email was personalized.
    """
    company_finding: str = ""
    evidence_source_url: str = ""
    evidence_source_title: str = ""
    source_excerpt: str = ""
    candidate_experience: str = ""
    connection_reasoning: str = ""
    technical_domain: str = ""
    supporting_cv_evidence: str = ""
    limitations_or_assumptions: list[str] = field(default_factory=list)
    contact_discrepancies: list[dict[str, Any]] = field(default_factory=list)
    unresolved_caveats: list[str] = field(default_factory=list)
    quality_gate_passed: bool = True
    quality_gate_checks: dict[str, bool] = field(default_factory=dict)
    quality_gate_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonalizationRecord:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            company_finding=data.get("company_finding", ""),
            evidence_source_url=data.get("evidence_source_url", ""),
            evidence_source_title=data.get("evidence_source_title", ""),
            source_excerpt=data.get("source_excerpt", ""),
            candidate_experience=data.get("candidate_experience", ""),
            connection_reasoning=data.get("connection_reasoning", ""),
            technical_domain=data.get("technical_domain", ""),
            supporting_cv_evidence=data.get("supporting_cv_evidence", ""),
            limitations_or_assumptions=data.get("limitations_or_assumptions", []),
            contact_discrepancies=data.get("contact_discrepancies", []),
            unresolved_caveats=data.get("unresolved_caveats", []),
            quality_gate_passed=data.get("quality_gate_passed", True),
            quality_gate_checks=data.get("quality_gate_checks", {}),
            quality_gate_notes=data.get("quality_gate_notes", []),
        )


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
        attachment_path: File path of the attached CV (the uploaded original).
        mode: The WorkflowMode used to generate this draft.
        status: Current status (DRAFT, APPROVED, SENT, REJECTED).
        approved: True only if explicit user approval was granted.
        review_required: True if company research, domain, or role matching returned REVIEW.
        review_flags: Warnings or notes requiring human attention.
        created_at: ISO timestamp of creation.
        generation_status: Outcome of generation (DRAFT_GENERATED, NEEDS_REVIEW, GENERATION_FAILED).
        personalization_explanation: Concise explanation of evidence connecting candidate to company.
        evidence_summary: Extracted summary of company AI/ML work.
        matched_skills: Candidate skills matched to company work.
        matched_projects: Specific CV projects highlighted in the email.
        recipient_type: Categorization of recipient (individual, generic_inbox, unknown).
        personalization_record: Structured reasoning and quality gate audit details.
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
    # Hash binding this approval to the exact reviewed draft (recipient,
    # subject, body, attachment identity). Empty for pre-feature approvals.
    approval_hash: str = ""
    review_required: bool = False
    review_flags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    generation_status: str = "DRAFT_GENERATED"
    personalization_explanation: str = ""
    evidence_summary: str = ""
    matched_skills: list[str] = field(default_factory=list)
    matched_projects: list[str] = field(default_factory=list)
    recipient_type: str = "individual"
    personalization_record: PersonalizationRecord | None = None



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
