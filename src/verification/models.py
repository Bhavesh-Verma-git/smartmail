"""
src/verification/models.py
---------------------------
Data classes and status constants for the Stage 4 verification and
matching layer.

## Status vocabulary

### Company / contact verification statuses

VERIFIED
    The available evidence is internally consistent and sufficiently
    supports proceeding.  This does NOT mean the company or contact has
    been independently authenticated — only that no contradictions were
    found and the evidence meets the minimum criteria for the workflow.

REVIEW
    Evidence is incomplete, ambiguous, or there is a concern that does
    not meet the threshold for automatic blocking.  A human must decide
    whether to proceed.  This is the default when certainty is low.

BLOCKED
    A serious contradiction or clearly invalid condition was detected
    that warrants stopping the workflow automatically.
    Note: A human override mechanism is not yet implemented in this stage;
    BLOCKED statuses currently stop the workflow completely and cannot be
    bypassed.

### Overall workflow status

PROCEED
    Both company and contact pass verification and the profile match is
    RELEVANT or REVIEW.  The system can continue to the next stage.

REVIEW  (shared constant)
    One or more fields needs human attention before the next stage.

BLOCKED (shared constant)
    A blocking condition exists.  The workflow stops here.

### Profile relevance statuses

RELEVANT
    The company's evidence-backed technical areas overlap with the
    candidate's profile.  Proceeding is appropriate.

NOT_A_MATCH
    The available evidence indicates the company's work is unlikely to
    be relevant to an AI/ML internship based on the profile.

REVIEW  (shared constant)
    Insufficient evidence to determine relevance, or the overlap is
    ambiguous.

### Outreach case type

SPECIFIC_INTERNSHIP
    A role or job URL was supplied and the role_status is FOUND.
    The outreach should reference the specific opportunity.

GENERAL_INQUIRY
    No specific listing was found (role_status is NOT_FOUND or UNKNOWN).
    The outreach should be a general HR internship inquiry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

# Shared across company_status, contact_status, overall_status
VERIFIED = "VERIFIED"
BLOCKED = "BLOCKED"

# profile relevance
RELEVANT = "RELEVANT"
NOT_A_MATCH = "NOT_A_MATCH"

# overall_status — note: REVIEW and BLOCKED are shared with the above
PROCEED = "PROCEED"

# Imported from research models for convenience — do not redefine here.
# REVIEW is defined in research.models; import it where needed.

# Case type
SPECIFIC_INTERNSHIP = "SPECIFIC_INTERNSHIP"
GENERAL_INQUIRY = "GENERAL_INQUIRY"


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """
    Structured output from the verification layer.

    Each field is independent — a good contact_status does not compensate
    for a poor company_status, and vice versa.

    Fields
    ------
    company_status      VERIFIED | REVIEW | BLOCKED
    contact_status      VERIFIED | REVIEW | BLOCKED
    overall_status      PROCEED | REVIEW | BLOCKED
    blocking_reasons    Non-empty only when overall_status is BLOCKED.
                        Each entry is a specific, human-readable reason.
    review_flags        Concerns that do not rise to BLOCKED level but
                        require human attention before proceeding.
    verification_notes  Additional context for the human reviewer.
    evidence_sources    Non-empty, deduplicated research source URLs that
                        bear on the company/contact verification decision.
    """
    company_status: str = "REVIEW"
    contact_status: str = "REVIEW"
    overall_status: str = "REVIEW"
    blocking_reasons: list[str] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)
    verification_notes: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)

    @property
    def source_urls(self) -> list[str]:
        """Convenience alias for evidence_sources."""
        return self.evidence_sources


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    Structured output from the profile matching layer.

    Fields
    ------
    relevance_status    RELEVANT | REVIEW | NOT_A_MATCH
    case_type           SPECIFIC_INTERNSHIP | GENERAL_INQUIRY
    matched_areas       Technical areas from research evidence that appear
                        in the candidate's profile.  Evidence-backed only.
    profile_evidence    Which profile elements support the match, e.g.
                        project names, skill categories.  No invented items.
    explanation         1-2 sentence human-readable explanation drawing
                        on actual profile facts and research evidence.
    match_notes         Additional context, caveats, or open questions.
    evidence_sources    Non-empty, deduplicated research source URLs that
                        support the matched technical areas or role listing.
    """
    relevance_status: str = "REVIEW"
    case_type: str = GENERAL_INQUIRY
    matched_areas: list[str] = field(default_factory=list)
    profile_evidence: list[str] = field(default_factory=list)
    explanation: str = ""
    match_notes: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)

    @property
    def source_urls(self) -> list[str]:
        """Convenience alias for evidence_sources."""
        return self.evidence_sources


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Combined output of run_verification().

    Holds the research inputs that drove the result alongside the
    verification and matching outputs so that every conclusion is
    traceable to its source.

    Fields
    ------
    verification    VerificationResult from the verification layer.
    match           MatchResult from the profile matching layer.
    summary         One-sentence plain-English summary for quick review.
    """
    verification: VerificationResult = field(
        default_factory=VerificationResult
    )
    match: MatchResult = field(default_factory=MatchResult)
    summary: str = ""
