"""
src/verification/verifier.py
-----------------------------
Company and contact verification.

## Public API

    verify(research_result, research_input) -> VerificationResult

## How it works

The verifier reads the status fields produced by Stage 3 research and applies
explicit, documented rules to determine:

  company_status  — how well the company evidence holds together
  contact_status  — how well the contact email / domain evidence holds together
  overall_status  — whether the workflow should PROCEED, wait for REVIEW,
                    or stop (BLOCKED)

## Verification rules

### company_status = VERIFIED
    ALL of the following must be true:
    - company_identity_status is SUPPORTED
    - domain_relationship_status is SUPPORTED or legitimately NOT_APPLICABLE
      (domain_relationship_status == REVIEW strictly yields company_status = REVIEW)

### company_status = BLOCKED
    company_identity_status is REVIEW  AND
    domain_relationship_status is REVIEW or UNKNOWN
    (both dimensions weak simultaneously — no corroboration available)

### company_status = REVIEW
    Everything else — at least one dimension is weak/unresolved, but not both.

### contact_status = VERIFIED
    domain_relationship_status is SUPPORTED.

### contact_status = BLOCKED
    This verifier does not block on contact alone.  A bad contact with
    a good company still proceeds to REVIEW so a human can evaluate.

### contact_status = REVIEW
    domain_relationship_status is NOT_APPLICABLE, REVIEW, or UNKNOWN.
    A contact with a free email provider (NOT_APPLICABLE) requires human review
    because a free provider cannot confirm the individual's affiliation with the company.

### overall_status = PROCEED
    company_status == VERIFIED and contact_status == VERIFIED

### overall_status = BLOCKED
    company_status == BLOCKED

### overall_status = REVIEW
    Everything else.

## What verification does NOT claim

- It does not claim the company is a legitimate employer.
- It does not claim the contact is an authorised recruiter.
- It does not claim the email address exists or can receive mail.
- It does not claim an internship is currently open.
- A good profile match does not override an unresolved verification.

## Separation of concerns

Role status (FOUND / NOT_FOUND / UNKNOWN) and application status are
NOT used in verification.  They belong to the matching layer.
"""

from __future__ import annotations

from src.research.models import (
    ResearchInput,
    ResearchResult,
    SUPPORTED,
    REVIEW as RES_REVIEW,
    UNKNOWN,
    NOT_APPLICABLE,
    RETRIEVAL_OK,
)
from src.verification.models import (
    VerificationResult,
    VERIFIED,
    BLOCKED,
)

# Convenience alias so this module can use "REVIEW" without shadowing the
# research constant.
_REVIEW = "REVIEW"


def verify(
    research_result: ResearchResult,
    research_input: ResearchInput,
) -> VerificationResult:
    """
    Apply verification rules to the research result.

    Parameters
    ----------
    research_result : ResearchResult from Stage 3.
    research_input  : ResearchInput that drove the research.

    Returns
    -------
    VerificationResult with company_status, contact_status, overall_status,
    blocking_reasons, review_flags, verification_notes, and evidence_sources.
    """
    blocking_reasons: list[str] = []
    review_flags: list[str] = []
    notes: list[str] = []

    identity = research_result.company_identity_status
    domain = research_result.domain_relationship_status

    # -----------------------------------------------------------------------
    # Company status
    # -----------------------------------------------------------------------
    company_status = _assess_company(
        identity, domain, research_input,
        blocking_reasons, review_flags, notes,
    )

    # -----------------------------------------------------------------------
    # Contact status
    # -----------------------------------------------------------------------
    contact_status = _assess_contact(
        domain, research_input,
        review_flags, notes,
    )

    # -----------------------------------------------------------------------
    # Overall status
    # -----------------------------------------------------------------------
    if blocking_reasons:
        overall_status = BLOCKED
    elif company_status == VERIFIED and contact_status == VERIFIED:
        overall_status = "PROCEED"
    else:
        overall_status = _REVIEW

    # -----------------------------------------------------------------------
    # Evidence sources
    # -----------------------------------------------------------------------
    sources = _extract_verification_sources(research_result, research_input)

    return VerificationResult(
        company_status=company_status,
        contact_status=contact_status,
        overall_status=overall_status,
        blocking_reasons=blocking_reasons,
        review_flags=review_flags,
        verification_notes=notes,
        evidence_sources=sources,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_verification_sources(
    research_result: ResearchResult,
    inp: ResearchInput,
) -> list[str]:
    """
    Extract relevant research source URLs that actually bear on company/contact
    verification decisions. Excludes empty URLs and duplicates while preserving order.
    """
    sources: list[str] = []
    seen: set[str] = set()

    for e in research_result.evidence:
        url = (getattr(e, "source_url", "") or "").strip()
        if not url:
            continue
        # Only successfully retrieved evidence can support a verification decision
        retrieval_status = getattr(e, "retrieval_status", "")
        if retrieval_status != RETRIEVAL_OK:
            continue

        bears_on_verification = False
        supports = getattr(e, "supports", [])
        if supports:
            if any(c in supports for c in ("company_identity", "domain_relationship")):
                bears_on_verification = True
        elif getattr(e, "source_type", "") in ("official_website", "source_url"):
            bears_on_verification = True
        elif inp.company_name and inp.company_name.lower() in (
            (getattr(e, "extracted_text", "") or "") + " " + (getattr(e, "title", "") or "")
        ).lower():
            bears_on_verification = True
        elif not supports and getattr(e, "source_type", "") != "job_listing":
            bears_on_verification = True

        if bears_on_verification and url not in seen:
            seen.add(url)
            sources.append(url)

    return sources


def _assess_company(
    identity: str,
    domain: str,
    inp: ResearchInput,
    blocking_reasons: list[str],
    review_flags: list[str],
    notes: list[str],
) -> str:
    """
    Determine company_status and populate blocking_reasons / review_flags.

    Returns
    -------
    "VERIFIED" | "REVIEW" | "BLOCKED"
    """
    # BLOCKED: no company name was supplied on a corporate-domain contact.
    # Domain discovery alone cannot resolve company identity when the company
    # name is unknown — we have no name to validate retrieved content against.
    # Exception: free-provider emails (domain=NOT_APPLICABLE) lack a corporate
    # domain to begin with, so missing company name stays REVIEW for human
    # triage rather than being silently discarded.
    if not inp.company_name and domain != NOT_APPLICABLE:
        blocking_reasons.append(
            "Company identity and domain relationship are both unresolved. "
            "No company name was supplied so identity cannot be established. "
            f"identity_status={identity}, domain_relationship_status={domain}."
        )
        return BLOCKED

    # BLOCKED: both identity and domain are weak simultaneously
    if identity in (RES_REVIEW, UNKNOWN) and domain in (RES_REVIEW, UNKNOWN):
        blocking_reasons.append(
            "Company identity and domain relationship are both unresolved. "
            "No dimension of the company evidence is sufficiently supported. "
            f"identity_status={identity}, domain_relationship_status={domain}."
        )
        return BLOCKED

    # Domain is REVIEW: company status must be REVIEW (even if identity is SUPPORTED).
    # Insufficient or conflicting domain evidence cannot yield VERIFIED.
    if domain == RES_REVIEW:
        review_flags.append(
            "Company identity evidence was retrieved, but the email domain "
            "relationship requires review. Company status cannot be VERIFIED "
            "while the domain relationship is under review."
        )
        if identity == SUPPORTED:
            notes.append(
                "Company identity is supported, but domain relationship status "
                "is REVIEW. Human review required before declaring verified."
            )
        return _REVIEW

    # VERIFIED: identity supported AND domain is SUPPORTED or legitimately NOT_APPLICABLE
    if identity == SUPPORTED and domain in (SUPPORTED, NOT_APPLICABLE):
        notes.append(
            "Company identity is SUPPORTED based on retrieved evidence. "
            "Check the limitations list for source quality details."
        )
        return VERIFIED

    # REVIEW: identity is unknown but domain is supported (or vice versa)
    if identity == UNKNOWN and domain == SUPPORTED:
        review_flags.append(
            "Domain relationship is supported but company identity could not "
            "be verified (no pages retrieved). Manually confirm the company."
        )
        return _REVIEW

    if identity == SUPPORTED and domain == UNKNOWN:
        review_flags.append(
            "Company identity is supported but no website domain comparison "
            "could be made. Manually check the email domain."
        )
        return _REVIEW

    # Catch-all REVIEW
    review_flags.append(
        f"Company verification is incomplete: identity={identity}, "
        f"domain={domain}. Human review required."
    )
    return _REVIEW


def _assess_contact(
    domain: str,
    inp: ResearchInput,
    review_flags: list[str],
    notes: list[str],
) -> str:
    """
    Determine contact_status.

    Rules
    -----
    VERIFIED : domain is SUPPORTED.
    REVIEW   : domain is NOT_APPLICABLE (free provider), REVIEW, or UNKNOWN.

    Contact is never BLOCKED by this verifier — a bad contact with a valid
    company goes to REVIEW for human evaluation.

    Returns
    -------
    "VERIFIED" | "REVIEW"
    """
    if domain == SUPPORTED:
        notes.append(
            "Email domain matches the company website domain. "
            "A valid email format or domain match does not prove that "
            "the individual is an authorized recruiter or that the address can receive mail."
        )
        return VERIFIED

    if domain == NOT_APPLICABLE:
        review_flags.append(
            f"Email is from a free provider ('{inp.registered_domain}'). "
            "The domain cannot be used to identify the employer or confirm contact affiliation. "
            "Human review is required before outreach."
        )
        notes.append(
            "Free email provider detected: contact relationship to company is unconfirmed."
        )
        return _REVIEW

    if domain == RES_REVIEW:
        review_flags.append(
            f"Email domain '{inp.email_domain}' needs review — it does not "
            "straightforwardly match the company website domain."
        )
        return _REVIEW

    # UNKNOWN domain
    review_flags.append(
        "Email domain relationship is unknown — no website was available for "
        "comparison. Cannot assess whether the contact email is consistent "
        "with the claimed company."
    )
    return _REVIEW
