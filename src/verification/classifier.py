"""
src/verification/classifier.py
------------------------------
Evidence-based classification system for SmartMail Stage 3.
Assigns clear primary outcomes (CONFIDENT, REVIEW_NAME, ELIMINATED, NOT_AI_RELEVANT)
and explicit reason codes based strictly on external facts and verifiable evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.contacts.recipient_resolver import (
    NAME_GENERIC_INBOX,
    NAME_INFERRED,
    NAME_UNKNOWN,
    NAME_VERIFIED,
    RecipientResolution,
)
from src.research.models import (
    NOT_APPLICABLE,
    RETRIEVAL_BLOCKED,
    RETRIEVAL_EMPTY,
    RETRIEVAL_ERROR,
    RETRIEVAL_JS_ONLY,
    RETRIEVAL_NETWORK_ERROR,
    RETRIEVAL_OK,
    REVIEW as RES_REVIEW,
    SUPPORTED,
    UNKNOWN,
    EvidenceItem,
    ResearchResult,
)
from src.verification.models import (
    BLOCKED,
    NOT_A_MATCH,
    PROCEED,
    RELEVANT,
    MatchResult,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# Primary Classification Outcomes
# ---------------------------------------------------------------------------
CLASSIFICATION_CONFIDENT = "CONFIDENT"
CLASSIFICATION_REVIEW_NAME = "REVIEW_NAME"
CLASSIFICATION_NOT_AI_RELEVANT = "NOT_AI_RELEVANT"
CLASSIFICATION_ELIMINATED = "ELIMINATED"

# Convenient aliases
CONFIDENT = CLASSIFICATION_CONFIDENT
REVIEW_NAME = CLASSIFICATION_REVIEW_NAME
NOT_AI_RELEVANT = CLASSIFICATION_NOT_AI_RELEVANT
ELIMINATED = CLASSIFICATION_ELIMINATED

# Explicit Reason Codes
REASON_CONFIDENT = "CONFIDENT"
REASON_INVALID_EMAIL = "INVALID_EMAIL"
REASON_DOMAIN_MISMATCH = "DOMAIN_MISMATCH"
REASON_COMPANY_UNVERIFIED = "COMPANY_UNVERIFIED"
REASON_RECIPIENT_UNVERIFIED = "RECIPIENT_UNVERIFIED"
REASON_RESEARCH_FAILED = "RESEARCH_FAILED"
REASON_CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
REASON_NOT_AI_RELEVANT = "NOT_AI_RELEVANT"
REASON_AI_RELEVANCE_UNCERTAIN = "AI_RELEVANCE_UNCERTAIN"


@dataclass
class ContactClassification:
    """
    Comprehensive classification result for a single contact record.
    Advisory only -- never authorizes automated live sending.
    """
    primary_outcome: str             # CONFIDENT | REVIEW_NAME | NOT_AI_RELEVANT | ELIMINATED
    reason_code: str                 # Specific reason code from above
    explanation: str                 # Short human-readable justification drawing on evidence
    ai_ml_relevance: str             # RELEVANT | REVIEW | NOT_AI_RELEVANT
    recipient_resolution: RecipientResolution
    matched_areas: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    is_outreach_ready: bool = False  # Advisory flag

    @property
    def ai_ml_relevant(self) -> bool:
        return self.ai_ml_relevance == "RELEVANT"

    @property
    def reason_summary(self) -> str:
        return self.explanation

    @property
    def recipient_name(self) -> str:
        if self.recipient_resolution:
            return self.recipient_resolution.resolved_name
        return ""

    @property
    def name_verification_status(self) -> str:
        if self.recipient_resolution:
            return self.recipient_resolution.name_verification_status
        return "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_outcome": self.primary_outcome,
            "reason_code": self.reason_code,
            "explanation": self.explanation,
            "reason_summary": self.explanation,
            "ai_ml_relevance": self.ai_ml_relevance,
            "ai_ml_relevant": self.ai_ml_relevant,
            "recipient_name": self.recipient_name,
            "name_verification_status": self.name_verification_status,
            "is_generic_inbox": getattr(self.recipient_resolution, "is_generic_inbox", False) if self.recipient_resolution else False,
            "matched_areas": self.matched_areas,
            "evidence_sources": self.evidence_sources,
            "limitations": self.limitations,
            "contradictions": self.contradictions,
            "is_outreach_ready": self.is_outreach_ready,
        }


def classify_contact(
    research_result: ResearchResult,
    verification_result: VerificationResult | Any,
    match_result: MatchResult | None = None,
    recipient_resolution: RecipientResolution | None = None,
    is_email_syntax_valid: bool = True,
    row_number: int | None = None,
    original_email: str = "",
) -> ContactClassification:
    """
    Determine the primary outcome and reason code using explicit evidence requirements.
    
    Order of Evaluation:
    1. Check for email syntax validity.
    2. Check for domain mismatch or contradictory evidence.
    3. Check for technical research failures (network errors, blocked pages).
    4. Check company identity establishment.
    5. Evaluate AI/ML technical relevance.
    6. Evaluate recipient identity (personal verified vs generic inbox vs unverified).
    """
    # Unpack PipelineResult if passed
    if hasattr(verification_result, "verification") and hasattr(verification_result, "match"):
        actual_verification = verification_result.verification
        if match_result is None:
            match_result = verification_result.match
    else:
        actual_verification = verification_result

    if match_result is None:
        match_result = MatchResult(
            relevance_status="REVIEW",
            case_type="GENERAL_INQUIRY",
            matched_areas=research_result.technical_areas,
            profile_evidence=[],
            explanation="",
        )

    if recipient_resolution is None:
        recipient_resolution = getattr(research_result, "recipient_resolution", None)
    if recipient_resolution is None:
        from src.contacts.recipient_resolver import resolve_recipient
        recipient_resolution = resolve_recipient(original_email)

    evidence = research_result.evidence
    contradictions = list(research_result.contradictions)
    limitations = list(research_result.limitations)
    matched_areas = list(match_result.matched_areas)
    evidence_sources = list(getattr(actual_verification, "evidence_sources", []))

    # 1. Email syntax check
    if not is_email_syntax_valid:
        return ContactClassification(
            primary_outcome=CLASSIFICATION_ELIMINATED,
            reason_code=REASON_INVALID_EMAIL,
            explanation="Email address syntax is malformed or invalid.",
            ai_ml_relevance="NOT_AI_RELEVANT",
            recipient_resolution=recipient_resolution,
            limitations=["Invalid email syntax."],
            is_outreach_ready=False,
        )

    # 2. Domain mismatch & contradictions
    domain_status = research_result.domain_relationship_status
    if actual_verification.overall_status == BLOCKED:
        reasons_str = "; ".join(actual_verification.blocking_reasons)
        if domain_status == NOT_APPLICABLE or "free provider" in reasons_str.lower():
            code = REASON_COMPANY_UNVERIFIED
        elif "conflict" in reasons_str.lower() or "mismatch" in reasons_str.lower():
            code = REASON_DOMAIN_MISMATCH if "domain" in reasons_str.lower() else REASON_CONTRADICTORY_EVIDENCE
        else:
            code = REASON_CONTRADICTORY_EVIDENCE
        
        return ContactClassification(
            primary_outcome=CLASSIFICATION_ELIMINATED,
            reason_code=code,
            explanation=f"Conflicting or blocked evidence: {reasons_str}",
            ai_ml_relevance="NOT_AI_RELEVANT",
            recipient_resolution=recipient_resolution,
            limitations=limitations,
            contradictions=contradictions,
            is_outreach_ready=False,
        )

    # 3. Check for technical research failures
    # Important: Technical failures must NOT be silently marked as ELIMINATED!
    retrieval_failures = [
        e for e in evidence
        if e.retrieval_status in (RETRIEVAL_ERROR, RETRIEVAL_NETWORK_ERROR, RETRIEVAL_BLOCKED)
    ]
    ok_external_pages = [
        e for e in evidence
        if e.retrieval_status == RETRIEVAL_OK
        and e.source_type not in ("user_supplied", "notes")
        and bool((e.extracted_text or "").strip() or (e.title or "").strip())
    ]

    if retrieval_failures and not ok_external_pages and research_result.company_identity_status in (UNKNOWN, RES_REVIEW):
        fail_reasons = "; ".join(e.retrieval_note or e.claim_supported for e in retrieval_failures if e.retrieval_note or e.claim_supported)
        return ContactClassification(
            primary_outcome=CLASSIFICATION_REVIEW_NAME,
            reason_code=REASON_RESEARCH_FAILED,
            explanation=f"Automated web research encountered technical errors ({fail_reasons or 'unreachable'}). Manual review required.",
            ai_ml_relevance="REVIEW",
            recipient_resolution=recipient_resolution,
            evidence_sources=[e.source_url for e in retrieval_failures if e.source_url],
            limitations=limitations,
            is_outreach_ready=False,
        )

    # 4. Check company identity establishment
    # Free email provider with no company identity
    if domain_status == NOT_APPLICABLE and research_result.company_identity_status in (UNKNOWN, RES_REVIEW) and not ok_external_pages:
        return ContactClassification(
            primary_outcome=CLASSIFICATION_ELIMINATED,
            reason_code=REASON_COMPANY_UNVERIFIED,
            explanation="Contact uses a free email provider (@gmail/@yahoo) with no verifiable company website or external evidence.",
            ai_ml_relevance="NOT_AI_RELEVANT",
            recipient_resolution=recipient_resolution,
            limitations=limitations,
            is_outreach_ready=False,
        )

    # Company identity completely unverified / unknown
    if research_result.company_identity_status == UNKNOWN:
        return ContactClassification(
            primary_outcome=CLASSIFICATION_ELIMINATED,
            reason_code=REASON_COMPANY_UNVERIFIED,
            explanation="No credible web evidence could be retrieved to establish the company identity.",
            ai_ml_relevance="NOT_AI_RELEVANT",
            recipient_resolution=recipient_resolution,
            limitations=limitations,
            is_outreach_ready=False,
        )

    # 5. Evaluate AI/ML technical relevance
    # Important distinction: A confirmed company that does not do AI/ML is NOT_AI_RELEVANT,
    # not an identity failure!
    if match_result.relevance_status == NOT_A_MATCH or (
        ok_external_pages and not research_result.technical_areas and not matched_areas
    ):
        return ContactClassification(
            primary_outcome=CLASSIFICATION_NOT_AI_RELEVANT,
            reason_code=REASON_NOT_AI_RELEVANT,
            explanation="Retrieved official company evidence does not establish meaningful AI/ML products, research, or engineering work.",
            ai_ml_relevance="NOT_AI_RELEVANT",
            recipient_resolution=recipient_resolution,
            evidence_sources=evidence_sources,
            limitations=limitations,
            is_outreach_ready=False,
        )

    # If relevance is ambiguous or unverified
    if match_result.relevance_status == "REVIEW" and not matched_areas:
        return ContactClassification(
            primary_outcome=CLASSIFICATION_REVIEW_NAME,
            reason_code=REASON_AI_RELEVANCE_UNCERTAIN,
            explanation="External evidence does not clearly confirm whether company engages in AI/ML work. Human review needed.",
            ai_ml_relevance="REVIEW",
            recipient_resolution=recipient_resolution,
            evidence_sources=evidence_sources,
            limitations=limitations,
            is_outreach_ready=False,
        )

    # 6. Evaluate recipient identity for companies with confirmed AI/ML relevance
    # Company is verified and AI/ML work is supported!
    # Check recipient:
    if recipient_resolution.name_verification_status == NAME_VERIFIED:
        # Fully verified person at relevant company
        return ContactClassification(
            primary_outcome=CLASSIFICATION_CONFIDENT,
            reason_code=REASON_CONFIDENT,
            explanation=f"Company identity verified with confirmed AI/ML work ({', '.join(matched_areas)}). Recipient identity independently verified.",
            ai_ml_relevance="RELEVANT",
            recipient_resolution=recipient_resolution,
            matched_areas=matched_areas,
            evidence_sources=evidence_sources,
            limitations=limitations,
            is_outreach_ready=True,
        )
    elif recipient_resolution.is_generic_inbox:
        # Functional inbox (hr@, careers@, info@) at relevant company
        role_label = recipient_resolution.inferred_role_or_title or "functional inbox"
        return ContactClassification(
            primary_outcome=CLASSIFICATION_REVIEW_NAME,
            reason_code=REASON_RECIPIENT_UNVERIFIED,
            explanation=f"Company identity and AI/ML work supported ({', '.join(matched_areas)}). Contact is a {role_label}; review before outreach.",
            ai_ml_relevance="RELEVANT",
            recipient_resolution=recipient_resolution,
            matched_areas=matched_areas,
            evidence_sources=evidence_sources,
            limitations=limitations + ["Functional organizational mailbox; individual recipient name unknown."],
            is_outreach_ready=False,
        )
    elif recipient_resolution.name_verification_status == NAME_INFERRED:
        # Inferred name from username or supplied in spreadsheet, but no independent public profile
        return ContactClassification(
            primary_outcome=CLASSIFICATION_REVIEW_NAME,
            reason_code=REASON_RECIPIENT_UNVERIFIED,
            explanation=f"Company identity and AI/ML work supported ({', '.join(matched_areas)}). Recipient name '{recipient_resolution.inferred_name}' is unverified against public sources.",
            ai_ml_relevance="RELEVANT",
            recipient_resolution=recipient_resolution,
            matched_areas=matched_areas,
            evidence_sources=evidence_sources,
            limitations=limitations + ["Recipient name inferred from email/spreadsheet; unverified by external public profile."],
            is_outreach_ready=False,
        )
    else:
        # Unknown recipient name
        return ContactClassification(
            primary_outcome=CLASSIFICATION_REVIEW_NAME,
            reason_code=REASON_RECIPIENT_UNVERIFIED,
            explanation=f"Company identity and AI/ML work supported ({', '.join(matched_areas)}), but recipient individual is unknown.",
            ai_ml_relevance="RELEVANT",
            recipient_resolution=recipient_resolution,
            matched_areas=matched_areas,
            evidence_sources=evidence_sources,
            limitations=limitations + ["Recipient individual is unknown."],
            is_outreach_ready=False,
        )
