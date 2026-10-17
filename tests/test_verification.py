"""
tests/test_verification.py
---------------------------
Unit tests for src/verification/ (Stage 4).

All tests use synthetic fixtures.
No network access, no real recruiter emails, no live websites.

Test coverage
-------------
A. verifier.py
   1. Company with consistent supporting evidence -> VERIFIED + PROCEED
   2. Company with insufficient evidence (both dimensions unknown) -> BLOCKED
   3. Identity supported but domain unknown -> REVIEW
   4. Domain supported but identity unknown -> REVIEW
   5. Contact with matching domain -> VERIFIED contact
   6. Contact with free provider -> REVIEW (cannot confirm affiliation)
   7. Contact with mismatched domain -> REVIEW contact
   8. Overall BLOCKED when company is BLOCKED
   9. Overall REVIEW when domain is REVIEW (both company and contact are REVIEW)

B. matcher.py
   10. AI/ML-relevant technical areas + filled profile -> RELEVANT
   11. No technical areas from evidence -> REVIEW
   12. Technical areas found but profile has no filled skills -> REVIEW
   13. Non-AI/ML company evidence -> NOT_A_MATCH for AI/ML profile
   14. Specific internship (role FOUND) -> case_type=SPECIFIC_INTERNSHIP
   15. No internship listing -> case_type=GENERAL_INQUIRY
   16. Matched areas appear in matched_areas field

C. pipeline.py (run_verification)
   17. Full happy path: PROCEED
   18. BLOCKED verification, good match -> still BLOCKED
   19. PROCEED verification, NOT_A_MATCH relevance -> demoted to REVIEW
   20. REVIEW verification, RELEVANT match -> stays REVIEW
   21. Summary includes company name

D. Integration
   22. Conflicting company name -> company BLOCKED (both dims REVIEW)
   23. General HR inquiry: no role, company passes -> PROCEED / REVIEW
        based on evidence; must not claim role is FOUND
   24. Technically relevant company with insufficient verification
   25. Missing source information (no website) -> REVIEW or BLOCKED
   26. All existing research + contact-import tests still pass (regression)
"""

from __future__ import annotations

import pytest

from src.research.models import (
    EvidenceItem,
    ResearchInput,
    ResearchResult,
    SUPPORTED,
    REVIEW as RES_REVIEW,
    UNKNOWN,
    NOT_APPLICABLE,
    FOUND,
    NOT_FOUND,
    RETRIEVAL_OK,
    RETRIEVAL_ERROR,
    APPLICATION_OPEN,
    APPLICATION_UNKNOWN,
)
from src.verification.models import (
    VerificationResult,
    MatchResult,
    PipelineResult,
    VERIFIED,
    BLOCKED,
    RELEVANT,
    NOT_A_MATCH,
    SPECIFIC_INTERNSHIP,
    GENERAL_INQUIRY,
    PROCEED,
)
from src.verification.verifier import verify
from src.verification.matcher import match_profile
from src.verification.pipeline import run_verification


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_research_input(**kwargs) -> ResearchInput:
    defaults = dict(
        company_name="Fable Technologies",
        contact_name="Alice",
        normalized_email="alice@fabletech.example",
        email_domain="fabletech.example",
        registered_domain="fabletech.example",
        is_free_provider=False,
        website="https://fabletech.example",
        role_or_job_url="",
        source_url="",
        notes="",
    )
    defaults.update(kwargs)
    return ResearchInput(**defaults)


def _make_research_result(**kwargs) -> ResearchResult:
    """Build a ResearchResult with sensible defaults, overridden by kwargs."""
    defaults = dict(
        company_identity_status=SUPPORTED,
        domain_relationship_status=SUPPORTED,
        role_status=NOT_FOUND,
        application_status=APPLICATION_UNKNOWN,
        technical_areas=[],
        evidence=[],
        limitations=[],
        recommended_next_action="",
    )
    defaults.update(kwargs)
    return ResearchResult(**defaults)


def _minimal_profile(
    ml_skills: list[str] | None = None,
    frameworks: list[str] | None = None,
    target_roles: list[str] | None = None,
    project_name: str | None = None,
    project_techs: list[str] | None = None,
) -> dict:
    """
    Return a minimal but filled-in profile for matching tests.
    Only sets fields that are passed; everything else is a placeholder.
    """
    ml = ml_skills or []
    fw = frameworks or []
    roles = target_roles or []
    proj_name = project_name or "[PLACEHOLDER: Project Name]"
    proj_techs = project_techs or ["[PLACEHOLDER: Technology 1]"]

    return {
        "personal_info": {"full_name": "Alex Example"},
        "education": [],
        "experience": [],
        "projects": [
            {
                "id": "proj-001",
                "name": proj_name,
                "technologies": proj_techs,
                "highlight_for_roles": [],
            }
        ],
        "technical_skills": {
            "languages": ["Python"],
            "frameworks_and_libraries": fw,
            "databases": [],
            "tools_and_platforms": [],
            "ml_and_data": ml,
            "soft_skills": [],
        },
        "preferences": {
            "target_roles": roles,
        },
        "achievements_and_awards": [],
        "verified_metrics": {},
    }


def _ai_profile() -> dict:
    """A filled-in AI/ML profile for use in matching tests."""
    return _minimal_profile(
        ml_skills=["PyTorch", "scikit-learn", "Transformers", "LangChain"],
        frameworks=["FastAPI", "HuggingFace"],
        target_roles=["ML Intern", "AI Research Intern"],
        project_name="RAG Question Answering System",
        project_techs=["Python", "LangChain", "OpenAI API", "FAISS"],
    )


def _blank_profile() -> dict:
    """Profile with all placeholder values (not yet filled in)."""
    return {
        "personal_info": {"full_name": "[PLACEHOLDER: Your Full Name]"},
        "technical_skills": {
            "languages": ["[PLACEHOLDER: Python]"],
            "frameworks_and_libraries": ["[PLACEHOLDER: e.g. FastAPI]"],
            "ml_and_data": ["[PLACEHOLDER: e.g. NumPy, Pandas]"],
        },
        "preferences": {
            "target_roles": ["[PLACEHOLDER: e.g. ML Intern]"],
        },
        "projects": [],
    }


# ===========================================================================
# A. verifier.py
# ===========================================================================

class TestVerifierCompanyStatus:

    def test_verified_when_identity_and_domain_both_supported(self):
        """Case 1: consistent evidence -> VERIFIED company."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
        )
        inp = _make_research_input()
        vr = verify(result, inp)
        assert vr.company_status == VERIFIED

    def test_blocked_when_both_identity_and_domain_unknown(self):
        """Case 2: insufficient evidence on both dimensions -> BLOCKED."""
        result = _make_research_result(
            company_identity_status=UNKNOWN,
            domain_relationship_status=UNKNOWN,
        )
        inp = _make_research_input()
        vr = verify(result, inp)
        assert vr.company_status == BLOCKED
        assert len(vr.blocking_reasons) > 0

    def test_blocked_when_identity_review_and_domain_review(self):
        """Both dimensions REVIEW simultaneously -> BLOCKED."""
        result = _make_research_result(
            company_identity_status=RES_REVIEW,
            domain_relationship_status=RES_REVIEW,
        )
        inp = _make_research_input()
        vr = verify(result, inp)
        assert vr.company_status == BLOCKED
        assert any("unresolved" in r.lower() for r in vr.blocking_reasons)

    def test_blocked_when_identity_review_and_domain_unknown(self):
        """Identity REVIEW + domain UNKNOWN -> both weak -> BLOCKED."""
        result = _make_research_result(
            company_identity_status=RES_REVIEW,
            domain_relationship_status=UNKNOWN,
        )
        inp = _make_research_input()
        vr = verify(result, inp)
        assert vr.company_status == BLOCKED

    def test_review_when_identity_supported_domain_unknown(self):
        """Case 4: identity supported but no website to compare domain."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=UNKNOWN,
        )
        inp = _make_research_input(website="")
        vr = verify(result, inp)
        assert vr.company_status == "REVIEW"
        assert any("domain" in f.lower() for f in vr.review_flags)

    def test_review_when_identity_unknown_domain_supported(self):
        """Case 3: domain matches but no pages retrieved for identity."""
        result = _make_research_result(
            company_identity_status=UNKNOWN,
            domain_relationship_status=SUPPORTED,
        )
        inp = _make_research_input()
        vr = verify(result, inp)
        assert vr.company_status == "REVIEW"
        assert any("identity" in f.lower() for f in vr.review_flags)

    def test_verified_company_when_domain_not_applicable(self):
        """Free email + supported identity -> company VERIFIED, but contact is REVIEW."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=NOT_APPLICABLE,
        )
        inp = _make_research_input(
            email_domain="gmail.com",
            registered_domain="gmail.com",
            is_free_provider=True,
        )
        vr = verify(result, inp)
        assert vr.company_status == VERIFIED
        assert vr.contact_status == "REVIEW"
        assert vr.overall_status == "REVIEW"

    def test_company_status_is_review_when_domain_relationship_is_review(self):
        """domain_relationship_status == REVIEW must yield company_status == REVIEW, never VERIFIED."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=RES_REVIEW,
        )
        inp = _make_research_input()
        vr = verify(result, inp)
        assert vr.company_status == "REVIEW"
        assert vr.company_status != VERIFIED
        assert any("domain" in f.lower() for f in vr.review_flags)

    def test_blocking_reasons_non_empty_only_when_blocked(self):
        """VERIFIED result must have empty blocking_reasons."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
        )
        vr = verify(result, _make_research_input())
        assert vr.blocking_reasons == []


class TestVerifierContactStatus:

    def test_contact_verified_when_domain_matches(self):
        """Case 5: matching domain -> VERIFIED contact."""
        result = _make_research_result(domain_relationship_status=SUPPORTED)
        vr = verify(result, _make_research_input())
        assert vr.contact_status == VERIFIED

    def test_contact_review_when_free_provider(self):
        """Case 6: free provider -> REVIEW (free provider cannot prove employer affiliation)."""
        result = _make_research_result(domain_relationship_status=NOT_APPLICABLE)
        inp = _make_research_input(is_free_provider=True, registered_domain="gmail.com")
        vr = verify(result, inp)
        assert vr.contact_status == "REVIEW"
        assert vr.contact_status != VERIFIED
        # But a review flag must mention the free provider
        assert any("free provider" in f.lower() for f in vr.review_flags)

    def test_contact_review_when_domain_mismatches(self):
        """Case 7: domain mismatch -> REVIEW contact, not BLOCKED."""
        result = _make_research_result(domain_relationship_status=RES_REVIEW)
        vr = verify(result, _make_research_input())
        assert vr.contact_status == "REVIEW"
        # Contact is never BLOCKED by this verifier
        assert vr.contact_status != BLOCKED

    def test_contact_review_when_domain_unknown(self):
        """Unknown domain -> REVIEW contact."""
        result = _make_research_result(domain_relationship_status=UNKNOWN)
        vr = verify(result, _make_research_input(website=""))
        assert vr.contact_status == "REVIEW"


class TestVerifierOverallStatus:

    def test_proceed_when_both_verified(self):
        """Case 1 end-to-end: both company and contact VERIFIED -> PROCEED."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
        )
        vr = verify(result, _make_research_input())
        assert vr.overall_status == "PROCEED"

    def test_blocked_when_company_blocked(self):
        """Case 8: company BLOCKED -> overall BLOCKED regardless of contact."""
        result = _make_research_result(
            company_identity_status=UNKNOWN,
            domain_relationship_status=UNKNOWN,
        )
        vr = verify(result, _make_research_input())
        assert vr.overall_status == BLOCKED

    def test_review_when_domain_is_review(self):
        """Case 9: domain is REVIEW -> company REVIEW, contact REVIEW -> overall REVIEW."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=RES_REVIEW,
        )
        vr = verify(result, _make_research_input())
        assert vr.company_status == "REVIEW"
        assert vr.contact_status == "REVIEW"
        assert vr.overall_status == "REVIEW"

    def test_free_provider_contact_never_yields_proceed(self):
        """A free-provider contact must never yield overall_status == PROCEED automatically."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=NOT_APPLICABLE,
        )
        inp = _make_research_input(is_free_provider=True, registered_domain="gmail.com")
        vr = verify(result, inp)
        assert vr.company_status == VERIFIED
        assert vr.contact_status == "REVIEW"
        assert vr.overall_status == "REVIEW"
        assert vr.overall_status != "PROCEED"

    def test_domain_match_does_not_claim_recruiter_authorized(self):
        """Verification must NOT positively claim the person is an authorised recruiter."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
        )
        vr = verify(result, _make_research_input())
        all_text = " ".join(vr.verification_notes + vr.review_flags).lower()
        # Positive claim patterns that must NOT appear (affirmative + recruiter)
        forbidden_phrases = [
            "confirms the sender is an authorised recruiter",
            "verified as an authorised recruiter",
            "sender is a confirmed recruiter",
            "contact is authorised to recruit",
        ]
        for phrase in forbidden_phrases:
            assert phrase not in all_text, (
                f"Verification text must not positively claim recruiter status. "
                f"Found forbidden phrase: '{phrase}'"
            )
        # Caveat form IS acceptable: "does not confirm ... is an authorised recruiter"
        # That is what the note should say.



# ===========================================================================
# B. matcher.py
# ===========================================================================

class TestMatcherRelevance:

    def test_relevant_when_aiml_areas_and_filled_profile(self):
        """Case 10: AI/ML company + AI/ML profile -> RELEVANT."""
        result = _make_research_result(
            technical_areas=["AI/ML", "Generative AI"],
            role_status=NOT_FOUND,
        )
        mr = match_profile(result, _ai_profile())
        assert mr.relevance_status == RELEVANT
        assert "AI/ML" in mr.matched_areas or "Generative AI" in mr.matched_areas

    def test_review_when_no_technical_areas(self):
        """Case 11: no technical evidence -> REVIEW."""
        result = _make_research_result(technical_areas=[])
        mr = match_profile(result, _ai_profile())
        assert mr.relevance_status == "REVIEW"
        assert any("technical areas" in n.lower() for n in mr.match_notes)

    def test_review_when_profile_not_filled_in(self):
        """Case 12: technical areas found but profile blank -> REVIEW."""
        result = _make_research_result(technical_areas=["AI/ML"])
        mr = match_profile(result, _blank_profile())
        assert mr.relevance_status == "REVIEW"
        assert any("filled" in n.lower() or "placeholder" in n.lower()
                   for n in mr.match_notes)

    def test_not_a_match_when_no_aiml_overlap(self):
        """Case 13: company is non-AI (e.g. Software Engineering only) + AI/ML profile."""
        result = _make_research_result(technical_areas=["Software Engineering"])
        profile = _minimal_profile(
            # Profile has ONLY AI/ML skills — no generic SW skills that would match
            ml_skills=["machine learning", "deep learning"],
            target_roles=["ML Intern"],
        )
        # Software Engineering area checks for Python, frameworks etc.
        # Profile has Python so this will actually match — use a profile with no overlap.
        pure_ai_profile = {
            "technical_skills": {
                "languages": [],
                "frameworks_and_libraries": [],
                "ml_and_data": ["deep learning", "pytorch"],
                "tools_and_platforms": [],
                "soft_skills": [],
            },
            "preferences": {"target_roles": ["ML Intern"]},
            "projects": [],
        }
        # Now: Software Engineering area looks for python/js/frameworks etc.
        # Profile has none of those -> NOT_A_MATCH
        mr = match_profile(result, pure_ai_profile)
        assert mr.relevance_status == NOT_A_MATCH

    def test_specific_internship_case_type(self):
        """Case 14: role FOUND -> SPECIFIC_INTERNSHIP case."""
        result = _make_research_result(
            technical_areas=["AI/ML"],
            role_status=FOUND,
        )
        mr = match_profile(result, _ai_profile())
        assert mr.case_type == SPECIFIC_INTERNSHIP

    def test_general_inquiry_case_type(self):
        """Case 15: role NOT_FOUND -> GENERAL_INQUIRY case."""
        result = _make_research_result(
            technical_areas=["AI/ML"],
            role_status=NOT_FOUND,
        )
        mr = match_profile(result, _ai_profile())
        assert mr.case_type == GENERAL_INQUIRY

    def test_unknown_role_gives_general_inquiry(self):
        """role UNKNOWN -> GENERAL_INQUIRY (no known specific listing)."""
        result = _make_research_result(
            technical_areas=["AI/ML"],
            role_status=UNKNOWN,
        )
        mr = match_profile(result, _ai_profile())
        assert mr.case_type == GENERAL_INQUIRY

    def test_matched_areas_populated(self):
        """Case 16: matched_areas contains the matching area labels."""
        result = _make_research_result(
            technical_areas=["Generative AI", "AI/ML"],
        )
        mr = match_profile(result, _ai_profile())
        assert mr.relevance_status == RELEVANT
        assert len(mr.matched_areas) >= 1

    def test_profile_evidence_not_invented(self):
        """profile_evidence must only contain items from the actual profile."""
        result = _make_research_result(technical_areas=["AI/ML"])
        profile = _ai_profile()
        mr = match_profile(result, profile)
        # Check nothing in profile_evidence mentions a project name not in profile
        for item in mr.profile_evidence:
            if item.startswith("Project:"):
                project_name = item.replace("Project: ", "")
                # Must be a real project name in the profile
                real_names = [
                    p.get("name", "") for p in profile.get("projects", [])
                ]
                assert project_name in real_names

    def test_explanation_references_profile_facts(self):
        """Explanation must reference actual profile content, not generic text."""
        result = _make_research_result(technical_areas=["AI/ML"])
        mr = match_profile(result, _ai_profile())
        assert mr.relevance_status == RELEVANT
        assert mr.explanation  # non-empty
        # Explanation should mention the area or a profile item
        assert "ai/ml" in mr.explanation.lower() or "ml intern" in mr.explanation.lower() \
               or "pytorch" in mr.explanation.lower() or "langchain" in mr.explanation.lower()

    def test_no_role_invented_in_general_inquiry(self):
        """
        When case_type is GENERAL_INQUIRY, the explanation must not claim
        a specific internship was found.
        """
        result = _make_research_result(
            technical_areas=["AI/ML"],
            role_status=NOT_FOUND,
        )
        mr = match_profile(result, _ai_profile())
        assert mr.case_type == GENERAL_INQUIRY
        # Explanation must not say "specific internship listing was found"
        assert "specific internship listing was found" not in mr.explanation

    def test_generic_software_engineering_keywords_alone_do_not_establish_aiml_relevance(self):
        """A company with only generic Software Engineering (Python, APIs) must not be RELEVANT for AI/ML profile."""
        result = _make_research_result(technical_areas=["Software Engineering"])
        # Candidate targets ML Intern and has Python in languages
        profile = _minimal_profile(
            ml_skills=["PyTorch", "scikit-learn"],
            target_roles=["ML Intern"],
        )
        mr = match_profile(result, profile)
        assert mr.relevance_status == NOT_A_MATCH
        assert mr.relevance_status != RELEVANT

    def test_generic_software_engineering_with_swe_target_gives_review_not_relevant(self):
        """If candidate also targets SWE, generic SWE gives REVIEW (uncertain AI relevance), not RELEVANT."""
        result = _make_research_result(technical_areas=["Software Engineering"])
        profile = _minimal_profile(
            ml_skills=["PyTorch"],
            target_roles=["ML Intern", "Software Engineering Intern"],
        )
        mr = match_profile(result, profile)
        assert mr.relevance_status == "REVIEW"
        assert mr.relevance_status != RELEVANT
        all_text = " ".join(mr.match_notes + [mr.explanation]).lower()
        assert "software-engineering" in all_text or "uncertain" in all_text

    def test_adjacent_ml_tools_recognized(self):
        """Adjacent ML tools (JAX, Keras, XGBoost, CUDA) in profile match AI/ML areas."""
        result = _make_research_result(technical_areas=["AI/ML"])
        profile = _minimal_profile(
            ml_skills=["JAX", "Keras", "XGBoost", "CUDA"],
            target_roles=["AI Research Intern"],
        )
        mr = match_profile(result, profile)
        assert mr.relevance_status == RELEVANT
        assert "AI/ML" in mr.matched_areas


# ===========================================================================
# C. pipeline.py (run_verification)
# ===========================================================================

class TestRunVerification:

    def test_full_happy_path_proceeds(self):
        """Case 17: all evidence good, AI/ML profile -> PROCEED."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            technical_areas=["AI/ML"],
            role_status=NOT_FOUND,
        )
        inp = _make_research_input()
        pipeline = run_verification(result, inp, _ai_profile())
        assert pipeline.verification.overall_status == "PROCEED"
        assert pipeline.match.relevance_status == RELEVANT

    def test_blocked_verification_stays_blocked_despite_good_match(self):
        """Case 18: BLOCKED company + good AI/ML match -> still BLOCKED."""
        result = _make_research_result(
            company_identity_status=UNKNOWN,
            domain_relationship_status=UNKNOWN,
            technical_areas=["AI/ML"],  # good match
        )
        inp = _make_research_input()
        pipeline = run_verification(result, inp, _ai_profile())
        assert pipeline.verification.overall_status == BLOCKED

    def test_proceed_demoted_to_review_when_not_a_match(self):
        """Case 19: PROCEED verification + NOT_A_MATCH relevance -> REVIEW."""
        # Create a profile with only AI/ML skills (no generic SW)
        ai_only_profile = {
            "technical_skills": {
                "languages": [],
                "frameworks_and_libraries": [],
                "ml_and_data": ["machine learning", "deep learning"],
                "tools_and_platforms": [],
                "soft_skills": [],
            },
            "preferences": {"target_roles": ["ML Intern"]},
            "projects": [],
        }
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            # Software Engineering area won't match an AI-only profile
            technical_areas=["Software Engineering"],
        )
        inp = _make_research_input()
        pipeline = run_verification(result, inp, ai_only_profile)
        assert pipeline.verification.overall_status == "REVIEW"

    def test_review_verification_with_relevant_match_stays_review(self):
        """Case 20: REVIEW verification + RELEVANT match -> stays REVIEW."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=RES_REVIEW,
            technical_areas=["AI/ML"],
        )
        inp = _make_research_input()
        pipeline = run_verification(result, inp, _ai_profile())
        assert pipeline.verification.overall_status == "REVIEW"
        assert pipeline.match.relevance_status == RELEVANT

    def test_summary_includes_company_name(self):
        """Case 21: summary must reference the company name."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            technical_areas=["AI/ML"],
        )
        inp = _make_research_input(company_name="Fable Technologies")
        pipeline = run_verification(result, inp, _ai_profile())
        assert "Fable Technologies" in pipeline.summary

    def test_summary_non_empty(self):
        """Summary must always be a non-empty string."""
        result = _make_research_result()
        pipeline = run_verification(result, _make_research_input(), _ai_profile())
        assert isinstance(pipeline.summary, str)
        assert len(pipeline.summary) > 0

    def test_pipeline_result_has_all_fields(self):
        """PipelineResult must have verification, match, and summary."""
        result = _make_research_result()
        pipeline = run_verification(result, _make_research_input(), _ai_profile())
        assert isinstance(pipeline.verification, VerificationResult)
        assert isinstance(pipeline.match, MatchResult)
        assert isinstance(pipeline.summary, str)


# ===========================================================================
# D. Integration scenarios
# ===========================================================================

class TestIntegrationScenarios:

    def test_conflicting_company_name_blocks(self):
        """
        Case 22: company name mismatch + domain mismatch -> BLOCKED.
        Both identity and domain are REVIEW -> BLOCKED.
        """
        result = _make_research_result(
            company_identity_status=RES_REVIEW,   # name not found on page
            domain_relationship_status=RES_REVIEW,  # email domain doesn't match
        )
        inp = _make_research_input()
        vr = verify(result, inp)
        assert vr.company_status == BLOCKED
        assert vr.overall_status == BLOCKED

    def test_general_hr_inquiry_no_internship_listed(self):
        """
        Case 23: company passes verification, no internship listing found.
        Must produce GENERAL_INQUIRY, must not claim role is FOUND.
        """
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            role_status=NOT_FOUND,
            application_status=APPLICATION_UNKNOWN,
            technical_areas=["AI/ML"],
        )
        inp = _make_research_input()
        pipeline = run_verification(result, inp, _ai_profile())
        assert pipeline.match.case_type == GENERAL_INQUIRY
        assert pipeline.match.relevance_status == RELEVANT
        # Must not claim a specific internship was found
        assert pipeline.verification.overall_status in ("PROCEED", "REVIEW")

    def test_technically_relevant_company_with_insufficient_verification(self):
        """
        Case 24: AI/ML company but both verification dimensions unknown.
        Good match must NOT promote to PROCEED — stays BLOCKED.
        """
        result = _make_research_result(
            company_identity_status=UNKNOWN,
            domain_relationship_status=UNKNOWN,
            technical_areas=["Generative AI", "AI/ML"],
        )
        inp = _make_research_input()
        pipeline = run_verification(result, inp, _ai_profile())
        assert pipeline.match.relevance_status == RELEVANT   # match is good
        assert pipeline.verification.overall_status == BLOCKED  # verification fails

    def test_missing_website_produces_review_or_blocked(self):
        """
        Case 25: no website supplied -> domain UNKNOWN, identity may be unknown.
        Must not silently PROCEED.
        """
        result = _make_research_result(
            company_identity_status=UNKNOWN,
            domain_relationship_status=UNKNOWN,
        )
        inp = _make_research_input(website="")
        vr = verify(result, inp)
        assert vr.overall_status in (BLOCKED, "REVIEW")
        assert vr.overall_status != "PROCEED"

    def test_conflicting_source_information_review(self):
        """
        Case 26: domain matches but identity is REVIEW (name not on page).
        Should produce REVIEW (company status), not PROCEED.
        """
        result = _make_research_result(
            company_identity_status=RES_REVIEW,
            domain_relationship_status=SUPPORTED,
        )
        inp = _make_research_input()
        vr = verify(result, inp)
        # Company: identity REVIEW + domain SUPPORTED -> not both weak -> REVIEW company
        assert vr.company_status == "REVIEW"
        assert vr.overall_status != "PROCEED"

    def test_advertised_internship_with_open_application(self):
        """
        Case 5 (spec): internship with active application -> SPECIFIC_INTERNSHIP.
        """
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            role_status=FOUND,
            application_status=APPLICATION_OPEN,
            technical_areas=["AI/ML"],
        )
        inp = _make_research_input(role_or_job_url="https://fabletech.example/jobs/intern")
        pipeline = run_verification(result, inp, _ai_profile())
        assert pipeline.match.case_type == SPECIFIC_INTERNSHIP
        assert pipeline.match.relevance_status == RELEVANT

    def test_internship_with_uncertain_status_is_not_claimed_open(self):
        """
        Case 6 (spec): role UNKNOWN + application UNKNOWN -> GENERAL_INQUIRY.
        The pipeline must not claim the internship is open.
        """
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            role_status=UNKNOWN,
            application_status=APPLICATION_UNKNOWN,
            technical_areas=["AI/ML"],
        )
        inp = _make_research_input()
        pipeline = run_verification(result, inp, _ai_profile())
        assert pipeline.match.case_type == GENERAL_INQUIRY

    def test_closed_internship_not_treated_as_open(self):
        """
        Case 7 (spec): role NOT_FOUND (closed) -> GENERAL_INQUIRY, not SPECIFIC.
        """
        from src.research.models import APPLICATION_CLOSED
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            role_status=NOT_FOUND,
            application_status=APPLICATION_CLOSED,
            technical_areas=["AI/ML"],
        )
        inp = _make_research_input()
        pipeline = run_verification(result, inp, _ai_profile())
        assert pipeline.match.case_type == GENERAL_INQUIRY


# ===========================================================================
# E. Evidence source preservation & Human override limitation
# ===========================================================================

class TestEvidenceSourcePreservation:

    def test_verification_preserves_valid_source_urls(self):
        """Source URLs from OK evidence are preserved in VerificationResult; empty/error excluded."""
        result = _make_research_result(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            evidence=[
                EvidenceItem(
                    source_url="https://fabletech.example",
                    source_type="official_website",
                    extracted_text="Fable Technologies official site",
                    retrieval_status=RETRIEVAL_OK,
                    supports=["company_identity", "domain_relationship"],
                ),
                EvidenceItem(
                    source_url="https://fabletech.example",  # duplicate URL
                    source_type="official_website",
                    extracted_text="Duplicate page",
                    retrieval_status=RETRIEVAL_OK,
                    supports=["company_identity"],
                ),
                EvidenceItem(
                    source_url="",  # empty URL
                    source_type="user_supplied",
                    extracted_text="User note",
                    retrieval_status=RETRIEVAL_OK,
                ),
                EvidenceItem(
                    source_url="https://broken.example",  # retrieval error
                    source_type="official_website",
                    extracted_text="",
                    retrieval_status=RETRIEVAL_ERROR,
                ),
            ],
        )
        inp = _make_research_input()
        vr = verify(result, inp)
        assert vr.evidence_sources == ["https://fabletech.example"]
        assert vr.source_urls == ["https://fabletech.example"]

    def test_matcher_preserves_valid_evidence_sources_separately_from_profile(self):
        """MatchResult preserves research source URLs and keeps candidate profile evidence separate."""
        result = _make_research_result(
            technical_areas=["AI/ML"],
            role_status=FOUND,
            evidence=[
                EvidenceItem(
                    source_url="https://fabletech.example",
                    source_type="official_website",
                    extracted_text="We specialize in machine learning and artificial intelligence.",
                    retrieval_status=RETRIEVAL_OK,
                    supports=["technical_relevance"],
                ),
                EvidenceItem(
                    source_url="https://fabletech.example/jobs/intern",
                    source_type="job_listing",
                    extracted_text="Apply now for our Machine Learning Internship opening.",
                    retrieval_status=RETRIEVAL_OK,
                    supports=["role_evidence", "technical_relevance"],
                ),
                EvidenceItem(
                    source_url="",
                    source_type="notes",
                    extracted_text="User notes",
                    retrieval_status=RETRIEVAL_OK,
                ),
            ],
        )
        profile = _ai_profile()
        mr = match_profile(result, profile)
        assert mr.relevance_status == RELEVANT
        assert mr.evidence_sources == [
            "https://fabletech.example",
            "https://fabletech.example/jobs/intern",
        ]
        assert mr.source_urls == mr.evidence_sources
        # profile_evidence contains candidate profile citations, NOT URLs
        assert any(item.startswith(("ML/Data", "Project", "Target")) for item in mr.profile_evidence)
        for p_item in mr.profile_evidence:
            assert not p_item.startswith("http")

    def test_matcher_no_sources_when_not_a_match(self):
        """When relevance is NOT_A_MATCH, evidence_sources is empty."""
        result = _make_research_result(
            technical_areas=["Software Engineering"],
            evidence=[
                EvidenceItem(
                    source_url="https://generic.example",
                    source_type="official_website",
                    extracted_text="General software engineering services.",
                    retrieval_status=RETRIEVAL_OK,
                )
            ],
        )
        pure_ai_profile = {
            "technical_skills": {
                "languages": ["Python"],
                "frameworks_and_libraries": [],
                "ml_and_data": ["deep learning", "pytorch"],
                "tools_and_platforms": [],
                "soft_skills": [],
            },
            "preferences": {"target_roles": ["ML Intern"]},
            "projects": [],
        }
        mr = match_profile(result, pure_ai_profile)
        assert mr.relevance_status == NOT_A_MATCH
        assert mr.evidence_sources == []


class TestHumanOverrideLimitation:

    def test_human_override_limitation_documented_blocked_not_bypassed(self):
        """Blocked verification cannot be bypassed; override mechanism is not implemented."""
        result = _make_research_result(
            company_identity_status=UNKNOWN,
            domain_relationship_status=UNKNOWN,
            technical_areas=["AI/ML"],
        )
        pipeline = run_verification(result, _make_research_input(), _ai_profile())
        assert pipeline.verification.overall_status == BLOCKED
        # Ensure BLOCKED is not bypassed and no override bypass exists
        assert pipeline.verification.overall_status != "PROCEED"


# ===========================================================================
# F. Regression: existing tests are not broken
# ===========================================================================

class TestRegression:
    """
    Smoke-level regression for the Stage 1-3 modules.
    The full regression runs in the other test files (test_csv_importer.py,
    test_email_parser.py, test_profile_valid.py, test_research.py).
    These tests confirm Stage 4 does not accidentally break imports.
    """

    def test_research_models_importable(self):
        from src.research.models import (
            EvidenceItem, ResearchInput, ResearchResult,
            SUPPORTED, REVIEW, UNKNOWN, FOUND, NOT_FOUND,
            APPLICATION_OPEN, APPLICATION_CLOSED, APPLICATION_UNKNOWN,
        )

    def test_research_checks_importable(self):
        from src.research.checks import (
            check_company_identity,
            check_domain_relationship,
            check_role_evidence,
            check_technical_relevance,
        )

    def test_csv_importer_importable(self):
        from src.contacts.csv_importer import import_contacts_from_csv, ContactRecord

    def test_email_parser_importable(self):
        from src.contacts.email_parser import parse_email

    def test_researcher_importable(self):
        from src.research.researcher import research_contact, build_research_input

    def test_csv_import_still_works(self):
        """Confirm Stage 1 CSV import functionality is unaffected."""
        import textwrap, tempfile
        from src.contacts.csv_importer import import_contacts_from_csv

        csv_content = textwrap.dedent("""
            company_name,email,website
            Fable Technologies,alice@fabletech.example,https://fabletech.example
        """).strip()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_content)
            tmp_path = f.name

        result = import_contacts_from_csv(tmp_path)
        assert len(result.contacts) == 1
        assert result.contacts[0].company_name == "Fable Technologies"

    def test_research_role_detection_still_requires_active_signal(self):
        """
        Confirm Stage 3 correction is still in effect:
        intern keyword alone must not return FOUND.
        """
        from src.research.checks import check_role_evidence
        from src.research.models import ResearchInput, EvidenceItem, RETRIEVAL_OK

        inp = ResearchInput(role_or_job_url="https://example.com/intern")
        evidence = [EvidenceItem(
            source_url="https://example.com/intern",
            source_type="job_listing",
            extracted_text="We had a great internship programme last year.",
            retrieval_status=RETRIEVAL_OK,
        )]
        status, _, app_status = check_role_evidence(inp, evidence)
        assert status == UNKNOWN  # not FOUND — no active signal
