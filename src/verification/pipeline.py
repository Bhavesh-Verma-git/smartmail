"""
src/verification/pipeline.py
-----------------------------
Top-level verification pipeline for Stage 4.

## Public API

    run_verification(research_result, research_input, profile) -> PipelineResult

## What it does

1. Calls verify() to assess company and contact evidence.
2. Calls match_profile() to assess profile relevance.
3. Combines results into a PipelineResult.
4. Generates a one-sentence summary for quick human review.

## What it does NOT do

- Does not make HTTP requests (that is Stage 3's job).
- Does not generate CVs, draft emails, or access Gmail.
- Does not override a BLOCKED status because of a good profile match.
- Does not invent evidence.

## Separation of concerns

Verification and matching are independent.  A company that fails
verification but happens to be technically relevant is still BLOCKED.
A technically irrelevant company with clean verification is REVIEW,
not PROCEED — because proceeding with irrelevant outreach wastes effort.
"""

from __future__ import annotations

from src.research.models import ResearchInput, ResearchResult
from src.verification.matcher import match_profile
from src.verification.models import (
    PipelineResult,
    VerificationResult,
    MatchResult,
    VERIFIED,
    BLOCKED,
    RELEVANT,
    NOT_A_MATCH,
    PROCEED,
)
from src.verification.verifier import verify

_REVIEW = "REVIEW"


def run_verification(
    research_result: ResearchResult,
    research_input: ResearchInput,
    profile: dict,
    cv_text: str | None = None,
) -> PipelineResult:
    """
    Run the full Stage 4 verification and matching pipeline.

    Parameters
    ----------
    research_result : ResearchResult from Stage 3 research_contact().
    research_input  : ResearchInput that drove the Stage 3 research.
    profile         : dict loaded from master_profile.json via profile.loader.
    cv_text         : Optional raw text extracted from uploaded CV.

    Returns
    -------
    PipelineResult combining VerificationResult and MatchResult.
    """
    verification = verify(research_result, research_input)
    match = match_profile(research_result, profile, cv_text=cv_text)

    # A good profile match does NOT promote a BLOCKED verification.
    # A NOT_A_MATCH demotes a PROCEED to REVIEW (no point sending irrelevant outreach).
    overall = _combine_statuses(verification, match)
    if overall != verification.overall_status:
        verification.overall_status = overall

    summary = _build_summary(verification, match, research_input)

    return PipelineResult(
        verification=verification,
        match=match,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _combine_statuses(
    verification: VerificationResult,
    match: MatchResult,
) -> str:
    """
    Combine verification and match statuses into an overall workflow status.

    Rules (in priority order):
    1. BLOCKED always stays BLOCKED — profile match cannot override it.
    2. If verification is PROCEED but match is NOT_A_MATCH -> REVIEW
       (technically irrelevant companies should not proceed automatically).
    3. Otherwise, keep the verification overall_status.
    """
    if verification.overall_status == BLOCKED:
        return BLOCKED

    if verification.overall_status == PROCEED and match.relevance_status == NOT_A_MATCH:
        return _REVIEW

    return verification.overall_status


def _build_summary(
    verification: VerificationResult,
    match: MatchResult,
    inp: ResearchInput,
) -> str:
    """
    Build a one-sentence plain-English summary for quick human review.
    """
    company = inp.company_name or "Unknown company"
    overall = verification.overall_status
    relevance = match.relevance_status
    case = match.case_type

    if overall == BLOCKED:
        reason = (
            verification.blocking_reasons[0]
            if verification.blocking_reasons
            else "verification failed"
        )
        return (
            f"{company}: BLOCKED — {reason}"
        )

    if overall == PROCEED:
        case_label = (
            "specific internship found"
            if case == "SPECIFIC_INTERNSHIP"
            else "general inquiry"
        )
        return (
            f"{company}: PROCEED — verification passed, {relevance.lower()} "
            f"match ({case_label})."
        )

    # REVIEW
    flags = verification.review_flags[:1]
    flag_str = flags[0] if flags else "review required"
    return (
        f"{company}: REVIEW — {flag_str}"
    )
