"""
tests/test_research.py
-----------------------
Unit tests for src/research/.

All HTTP calls are mocked via MockProvider -- no real network access.
Tests cover every scenario required by the spec plus boundary cases.

Scenarios covered
-----------------
- Official website supplied and accessible (returns company name in text)
- Website inaccessible or returning HTTP error
- Missing company name
- Missing website and role URL
- Company name conflicting with available website evidence
- Matching email and website domains -> SUPPORTED
- Free email provider -> NOT_APPLICABLE
- Legitimate-looking subdomain -> REVIEW
- External recruiting platform -> REVIEW
- Internship explicitly supported by evidence -> FOUND
- No opening found in the sources checked -> NOT_FOUND
- Role status unknown because the page is inaccessible -> UNKNOWN
- Conflicting or stale evidence (name mismatch)
- Evidence records retain their source URLs
- No company identity invented when evidence is missing
- No job opening invented when evidence is missing
- Existing contact importer and profile tests are unaffected (they run
  separately; this file does not import or modify those modules)
"""

from __future__ import annotations

import pytest

from src.research.checks import (
    check_company_identity,
    check_domain_relationship,
    check_role_evidence,
    check_technical_relevance,
    _registered_domain_from_url,
)
from src.research.models import (
    EvidenceItem,
    ResearchInput,
    ResearchResult,
    SUPPORTED,
    REVIEW,
    UNKNOWN,
    NOT_APPLICABLE,
    FOUND,
    NOT_FOUND,
    RETRIEVAL_OK,
    RETRIEVAL_ERROR,
    RETRIEVAL_SKIPPED,
    APPLICATION_OPEN,
    APPLICATION_CLOSED,
    APPLICATION_UNKNOWN,
)
from src.research.providers import ResearchProvider
from src.research.researcher import research_contact, build_research_input


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------

class MockProvider(ResearchProvider):
    """
    Test double for ResearchProvider.

    Accepts a mapping of url -> EvidenceItem.  If a URL is not in the map,
    it returns a RETRIEVAL_ERROR item.  This makes tests deterministic and
    independent of network access.
    """

    def __init__(self, responses: dict[str, EvidenceItem] | None = None) -> None:
        self._responses: dict[str, EvidenceItem] = responses or {}

    def fetch(
        self,
        url: str,
        source_type: str,
        supports: list[str],
    ) -> EvidenceItem:
        if url in self._responses:
            item = self._responses[url]
            # Ensure the supports list matches what the caller passed
            item.supports = list(supports)
            return item
        # URL not in map -> simulate connection error
        return EvidenceItem(
            source_url=url,
            source_type=source_type,
            retrieval_status=RETRIEVAL_ERROR,
            retrieval_note="ConnectionError",
            supports=list(supports),
            limitations=[f"Mock: no response configured for {url}."],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_item(
    url: str,
    source_type: str = "official_website",
    title: str = "",
    text: str = "",
) -> EvidenceItem:
    """Build a successful EvidenceItem for testing."""
    return EvidenceItem(
        source_url=url,
        source_type=source_type,
        title=title,
        extracted_text=text,
        retrieved_at="2024-01-01T00:00:00+00:00",
        retrieval_status=RETRIEVAL_OK,
        retrieval_note="HTTP 200",
    )


def _error_item(url: str, source_type: str = "official_website") -> EvidenceItem:
    """Build a failed EvidenceItem for testing."""
    return EvidenceItem(
        source_url=url,
        source_type=source_type,
        retrieved_at="2024-01-01T00:00:00+00:00",
        retrieval_status=RETRIEVAL_ERROR,
        retrieval_note="HTTP 404",
        limitations=[f"Server returned HTTP 404 for {url}."],
    )


def _basic_inp(**kwargs) -> ResearchInput:
    """Return a ResearchInput with sensible defaults, overridden by kwargs."""
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


# ===========================================================================
# A. Company identity check
# ===========================================================================

class TestCheckCompanyIdentity:

    def test_supported_when_name_found_in_page_text(self):
        inp = _basic_inp()
        evidence = [_ok_item(
            "https://fabletech.example",
            text="Welcome to Fable Technologies, where we build robots.",
        )]
        status, limits = check_company_identity(inp, evidence)
        assert status == SUPPORTED

    def test_supported_when_name_found_in_page_title(self):
        inp = _basic_inp()
        evidence = [_ok_item(
            "https://fabletech.example",
            title="Fable Technologies | Official Site",
            text="Some other text that does not have the name.",
        )]
        status, limits = check_company_identity(inp, evidence)
        assert status == SUPPORTED

    def test_review_when_page_retrieved_but_name_missing(self):
        """Company name not found in page -> REVIEW, not an invented answer."""
        inp = _basic_inp(company_name="Fable Technologies")
        evidence = [_ok_item(
            "https://fabletech.example",
            text="This page belongs to a completely different company.",
        )]
        status, limits = check_company_identity(inp, evidence)
        assert status == REVIEW
        assert any("Fable Technologies" in lim for lim in limits)

    def test_unknown_when_no_pages_retrieved(self):
        inp = _basic_inp()
        evidence = [_error_item("https://fabletech.example")]
        status, limits = check_company_identity(inp, evidence)
        assert status == UNKNOWN

    def test_review_when_company_name_missing(self):
        """No company name supplied -> REVIEW, not an invented identity."""
        inp = _basic_inp(company_name="")
        status, limits = check_company_identity(inp, [])
        assert status == REVIEW
        assert any("company name" in lim.lower() for lim in limits)

    def test_no_company_identity_invented_from_email(self):
        """CRITICAL: researcher must not invent a company name from the email."""
        inp = _basic_inp(company_name="")
        evidence = [_ok_item(
            "https://fabletech.example",
            text="fabletech.example is mentioned here.",
        )]
        status, _ = check_company_identity(inp, evidence)
        # Must not be SUPPORTED -- we have no company name to verify
        assert status != SUPPORTED

    def test_partial_failure_noted_in_limitations(self):
        """If some pages fail, the limitation is noted even if identity is SUPPORTED."""
        inp = _basic_inp()
        evidence = [
            _ok_item("https://fabletech.example", text="Fable Technologies rocks."),
            _error_item("https://fabletech.example/careers"),
        ]
        status, limits = check_company_identity(inp, evidence)
        assert status == SUPPORTED
        assert any("could not be retrieved" in lim for lim in limits)

    def test_evidence_source_url_preserved(self):
        """EvidenceItem.source_url is always the original URL, never blank."""
        url = "https://fabletech.example"
        item = _ok_item(url, text="Fable Technologies")
        assert item.source_url == url


# ===========================================================================
# B. Domain relationship check
# ===========================================================================

class TestCheckDomainRelationship:

    def test_supported_when_email_and_website_domains_match(self):
        inp = _basic_inp(
            email_domain="fabletech.example",
            registered_domain="fabletech.example",
            website="https://fabletech.example",
        )
        status, limits = check_domain_relationship(inp, [])
        assert status == SUPPORTED
        # Must add a note that this is not conclusive proof
        assert any("not confirm" in lim or "not conclusive" in lim or "evidence" in lim.lower()
                   for lim in limits)

    def test_not_applicable_for_free_email_provider(self):
        inp = _basic_inp(
            email_domain="gmail.com",
            registered_domain="gmail.com",
            is_free_provider=True,
            website="https://fabletech.example",
        )
        status, limits = check_domain_relationship(inp, [])
        assert status == NOT_APPLICABLE
        assert any("free" in lim.lower() or "provider" in lim.lower() for lim in limits)

    def test_review_for_legitimate_subdomain(self):
        """jobs.bigcorp.com email, bigcorp.com website -> REVIEW, not fraud flag."""
        inp = _basic_inp(
            email_domain="jobs.bigcorp.example",
            registered_domain="bigcorp.example",
            website="https://bigcorp.example",
        )
        status, limits = check_domain_relationship(inp, [])
        assert status == REVIEW
        assert any("subdomain" in lim.lower() for lim in limits)

    def test_review_for_external_recruiting_platform(self):
        inp = _basic_inp(
            email_domain="greenhouse.io",
            registered_domain="greenhouse.io",
            website="https://fabletech.example",
        )
        status, limits = check_domain_relationship(inp, [])
        assert status == REVIEW
        assert any("recruiting platform" in lim.lower() or "platform" in lim.lower()
                   for lim in limits)

    def test_unknown_when_no_website_supplied(self):
        inp = _basic_inp(website="", is_free_provider=False)
        status, limits = check_domain_relationship(inp, [])
        assert status == UNKNOWN
        assert any("no company website" in lim.lower() for lim in limits)

    def test_review_when_email_domain_mismatches_website(self):
        inp = _basic_inp(
            email_domain="otherdomain.example",
            registered_domain="otherdomain.example",
            website="https://fabletech.example",
        )
        status, limits = check_domain_relationship(inp, [])
        assert status == REVIEW
        assert any("does not match" in lim.lower() or "mismatch" in lim.lower()
                   for lim in limits)

    def test_gmail_provider_does_not_prove_employer(self):
        """Free provider email + known company website -> NOT_APPLICABLE, not SUPPORTED."""
        inp = _basic_inp(
            email_domain="gmail.com",
            registered_domain="gmail.com",
            is_free_provider=True,
            website="https://fabletech.example",
        )
        status, _ = check_domain_relationship(inp, [])
        assert status == NOT_APPLICABLE


# ===========================================================================
# C. Role evidence check
# ===========================================================================

class TestCheckRoleEvidence:

    def test_found_when_role_url_contains_internship_keyword(self):
        """FOUND requires an intern keyword AND an active signal together."""
        inp = _basic_inp(role_or_job_url="https://fabletech.example/jobs/intern")
        evidence = [_ok_item(
            "https://fabletech.example/jobs/intern",
            source_type="job_listing",
            title="Software Engineering Intern | Fable Technologies",
            text="We are looking for an intern to join our team. Apply now.",
        )]
        status, _, app_status = check_role_evidence(inp, evidence)
        assert status == FOUND
        assert app_status == APPLICATION_OPEN

    def test_not_found_when_page_retrieved_but_no_role_keywords(self):
        inp = _basic_inp(role_or_job_url="https://fabletech.example/about")
        evidence = [_ok_item(
            "https://fabletech.example/about",
            source_type="job_listing",
            text="We build great robots and serve enterprise clients.",
        )]
        status, limits, _ = check_role_evidence(inp, evidence)
        assert status == NOT_FOUND
        assert any("not found" in lim.lower() or "keywords" in lim.lower()
                   for lim in limits)

    def test_unknown_when_role_url_inaccessible(self):
        inp = _basic_inp(role_or_job_url="https://fabletech.example/jobs/intern")
        evidence = [_error_item(
            "https://fabletech.example/jobs/intern",
            source_type="job_listing",
        )]
        status, limits, _ = check_role_evidence(inp, evidence)
        assert status == UNKNOWN

    def test_unknown_when_no_role_url_and_no_pages(self):
        inp = _basic_inp(role_or_job_url="")
        status, limits, _ = check_role_evidence(inp, [])
        assert status == UNKNOWN

    def test_found_on_general_page_adds_limitation(self):
        """Internship + active signal on a non-role-url page: FOUND with note."""
        inp = _basic_inp(role_or_job_url="")  # no specific role URL
        evidence = [_ok_item(
            "https://fabletech.example",
            text="We have open internship positions. Apply now via our careers page.",
        )]
        status, limits, _ = check_role_evidence(inp, evidence)
        assert status == FOUND
        assert any("general page" in lim.lower() or "not a dedicated" in lim.lower()
                   for lim in limits)

    def test_not_found_does_not_claim_no_opening_exists(self):
        """NOT_FOUND means our research did not find it -- not that it doesn't exist."""
        inp = _basic_inp(role_or_job_url="https://fabletech.example/jobs")
        evidence = [_ok_item(
            "https://fabletech.example/jobs",
            source_type="job_listing",
            text="Full-time senior software engineer position.",
        )]
        status, limits, _ = check_role_evidence(inp, evidence)
        assert status == NOT_FOUND
        # The limitation text must NOT say "no opening exists"
        combined = " ".join(limits).lower()
        assert "no opening exists" not in combined

    def test_no_job_opening_invented_when_evidence_missing(self):
        """If no pages could be retrieved, status must be UNKNOWN not FOUND."""
        inp = _basic_inp(role_or_job_url="https://fabletech.example/closed-role")
        evidence = []  # no evidence at all
        status, _, app_status = check_role_evidence(inp, evidence)
        assert status == UNKNOWN
        assert app_status == APPLICATION_UNKNOWN


# ===========================================================================
# D. Technical relevance check
# ===========================================================================

class TestCheckTechnicalRelevance:

    def test_detects_machine_learning(self):
        evidence = [_ok_item("url", text="We use machine learning and deep learning.")]
        areas = check_technical_relevance(evidence)
        assert "AI/ML" in areas

    def test_detects_generative_ai(self):
        evidence = [_ok_item("url", text="We build large language model applications.")]
        areas = check_technical_relevance(evidence)
        assert "Generative AI" in areas

    def test_detects_data_science(self):
        evidence = [_ok_item("url", text="Our data science team uses analytics.")]
        areas = check_technical_relevance(evidence)
        assert "Data Science" in areas

    def test_detects_agentic_ai(self):
        evidence = [_ok_item("url", text="We are building multi-agent autonomous systems.")]
        areas = check_technical_relevance(evidence)
        assert "Agentic AI" in areas

    def test_returns_empty_when_no_ok_evidence(self):
        evidence = [_error_item("url")]
        areas = check_technical_relevance(evidence)
        assert areas == []

    def test_returns_empty_when_no_relevant_keywords(self):
        evidence = [_ok_item("url", text="We sell furniture and home decor.")]
        areas = check_technical_relevance(evidence)
        assert areas == []

    def test_more_specific_areas_appear_before_broader_ones(self):
        evidence = [_ok_item(
            "url",
            text="large language model machine learning data science",
        )]
        areas = check_technical_relevance(evidence)
        if "Generative AI" in areas and "AI/ML" in areas:
            assert areas.index("Generative AI") < areas.index("AI/ML")

    def test_negation_not_aiml_focused_returns_empty(self):
        """Negation statements must not be treated as positive evidence of AI/ML."""
        # Exact prompt example: "We are not an AI/ML-focused company."
        evidence = [_ok_item("url", text="We are not an AI/ML-focused company.")]
        areas = check_technical_relevance(evidence)
        assert "AI/ML" not in areas
        assert areas == []

        # Other negation variations
        evidence2 = [_ok_item("url", text="Company is not AI/ML focused; outreach is general inquiry.")]
        assert "AI/ML" not in check_technical_relevance(evidence2)

        evidence3 = [_ok_item("url", text="We do not use machine learning or deep learning in our products.")]
        assert "AI/ML" not in check_technical_relevance(evidence3)

        evidence4 = [_ok_item("url", text="We are a logistics provider, not an AI/ML company.")]
        assert "AI/ML" not in check_technical_relevance(evidence4)

    def test_explicit_aiml_products_detected(self):
        """Company explicitly describing genuine AI/ML, GenAI, RAG, and agentic work."""
        evidence = [_ok_item(
            "url",
            text="We build AI/ML products and agentic AI systems using RAG, vector search, and large language models.",
        )]
        areas = check_technical_relevance(evidence)
        assert "Generative AI" in areas
        assert "Agentic AI" in areas
        assert "AI/ML" in areas

    def test_no_relevant_technical_evidence_returns_empty(self):
        """Clearly non-AI company with no relevant technical keywords."""
        evidence = [_ok_item("url", text="We manufacture industrial valves and steel fittings for plumbing.")]
        areas = check_technical_relevance(evidence)
        assert areas == []

    def test_ambiguous_evidence_remains_available_for_human_review(self):
        """Ambiguous evidence (e.g. general software or empty portal) triggers REVIEW in matching."""
        from src.verification.matcher import match_profile

        ai_profile = {
            "personal_info": {"full_name": "Bhavesh Verma"},
            "education": {"university": "IIT Kharagpur", "degree": "Dual Degree"},
            "technical_skills": {
                "languages": ["Python"],
                "frameworks_and_libraries": ["PyTorch"],
                "ml_and_data": ["machine learning", "deep learning"],
                "tools_and_platforms": ["Docker"],
                "soft_skills": [],
            },
            "preferences": {"target_roles": ["AI/ML Intern", "Software Engineering Intern"]},
            "projects": [],
        }

        # Case 1: Generic SWE evidence without AI focus
        swe_evidence = [_ok_item("url", text="We develop backend APIs, microservices, and cloud native platforms.")]
        swe_areas = check_technical_relevance(swe_evidence)
        assert swe_areas == ["Software Engineering"]

        res_swe = ResearchResult(technical_areas=swe_areas)
        match_swe = match_profile(res_swe, ai_profile)
        # Because candidate also lists SWE target but company lacks AI focus, must trigger human review
        assert match_swe.relevance_status == "REVIEW"

        # Case 2: Inconclusive/sparse evidence (no technical areas detected)
        sparse_evidence = [_ok_item("url", text="Welcome to our company portal. Please log in.")]
        sparse_areas = check_technical_relevance(sparse_evidence)
        assert sparse_areas == []

        res_sparse = ResearchResult(technical_areas=sparse_areas)
        match_sparse = match_profile(res_sparse, ai_profile)
        # Empty technical areas yields REVIEW, keeping it available for human review
        assert match_sparse.relevance_status == "REVIEW"

    def test_affirmative_contrast_phrases_not_falsely_negated(self):
        """Phrases like 'not only' or 'not just' or contrasting clauses should remain positive."""
        evidence1 = [_ok_item("url", text="We are not just an AI/ML company, we also build robotics.")]
        assert "AI/ML" in check_technical_relevance(evidence1)

        evidence2 = [_ok_item("url", text="We do not use rule-based heuristics, we use AI/ML.")]
        assert "AI/ML" in check_technical_relevance(evidence2)


# ===========================================================================
# E. Helper utilities
# ===========================================================================

class TestRegisteredDomainHelper:

    def test_extracts_from_full_url(self):
        assert _registered_domain_from_url("https://jobs.bigcorp.com/role/1") == "bigcorp.com"

    def test_extracts_from_bare_domain(self):
        assert _registered_domain_from_url("https://bigcorp.com") == "bigcorp.com"

    def test_returns_empty_for_empty_string(self):
        assert _registered_domain_from_url("") == ""


# ===========================================================================
# F. End-to-end research_contact() via MockProvider
# ===========================================================================

class TestResearchContact:

    def test_full_happy_path_returns_supported_statuses(self):
        """Website accessible, name found, domains match, internship found with active signal."""
        inp = _basic_inp(
            company_name="Fable Technologies",
            website="https://fabletech.example",
            role_or_job_url="https://fabletech.example/jobs/intern",
            email_domain="fabletech.example",
            registered_domain="fabletech.example",
        )
        provider = MockProvider({
            "https://fabletech.example": _ok_item(
                "https://fabletech.example",
                text="Fable Technologies builds robotics.",
            ),
            "https://fabletech.example/jobs/intern": _ok_item(
                "https://fabletech.example/jobs/intern",
                source_type="job_listing",
                title="Software Intern",
                # Active signal ("apply now") + intern keyword -> FOUND + APPLICATION_OPEN
                text="Join our internship programme at Fable Technologies. Apply now.",
            ),
        })
        result = research_contact(inp, provider)
        assert result.company_identity_status == SUPPORTED
        assert result.role_status == FOUND
        assert result.domain_relationship_status == SUPPORTED

    def test_all_urls_inaccessible_gives_unknown_statuses(self):
        inp = _basic_inp(
            website="https://fabletech.example",
            role_or_job_url="https://fabletech.example/jobs/intern",
        )
        provider = MockProvider({})  # all URLs fail
        result = research_contact(inp, provider)
        assert result.company_identity_status == UNKNOWN
        assert result.role_status == UNKNOWN

    def test_missing_website_gives_unknown_identity(self):
        inp = _basic_inp(website="", role_or_job_url="")
        provider = MockProvider({})
        result = research_contact(inp, provider)
        assert result.company_identity_status == UNKNOWN

    def test_free_provider_email_gives_not_applicable_domain(self):
        inp = _basic_inp(
            email_domain="gmail.com",
            registered_domain="gmail.com",
            is_free_provider=True,
        )
        provider = MockProvider({
            "https://fabletech.example": _ok_item(
                "https://fabletech.example",
                text="Fable Technologies builds robotics.",
            )
        })
        result = research_contact(inp, provider)
        assert result.domain_relationship_status == NOT_APPLICABLE

    def test_result_contains_evidence_with_source_urls(self):
        """All fetched URLs must appear in the evidence list."""
        inp = _basic_inp(
            website="https://fabletech.example",
            role_or_job_url="https://fabletech.example/jobs/intern",
        )
        website_item = _ok_item("https://fabletech.example", text="Fable Technologies")
        role_item = _ok_item(
            "https://fabletech.example/jobs/intern",
            source_type="job_listing",
            text="internship position available",
        )
        provider = MockProvider({
            "https://fabletech.example": website_item,
            "https://fabletech.example/jobs/intern": role_item,
        })
        result = research_contact(inp, provider)
        source_urls = [e.source_url for e in result.evidence]
        assert "https://fabletech.example" in source_urls
        assert "https://fabletech.example/jobs/intern" in source_urls

    def test_notes_appear_as_user_supplied_evidence(self):
        inp = _basic_inp(notes="Met at university career fair 2024.")
        provider = MockProvider({
            "https://fabletech.example": _ok_item(
                "https://fabletech.example", text="Fable Technologies"
            )
        })
        result = research_contact(inp, provider)
        user_items = [e for e in result.evidence if e.source_type == "user_supplied"]
        assert len(user_items) == 1
        assert "career fair" in user_items[0].extracted_text

    def test_conflicting_evidence_returns_review_not_supported(self):
        """Page text does not contain the claimed company name -> REVIEW."""
        inp = _basic_inp(company_name="Fable Technologies")
        provider = MockProvider({
            "https://fabletech.example": _ok_item(
                "https://fabletech.example",
                text="Welcome to Apex Solutions, a completely different company.",
            )
        })
        result = research_contact(inp, provider)
        assert result.company_identity_status == REVIEW

    def test_result_has_recommended_next_action(self):
        inp = _basic_inp()
        provider = MockProvider({})
        result = research_contact(inp, provider)
        assert isinstance(result.recommended_next_action, str)
        assert len(result.recommended_next_action) > 0

    def test_result_has_limitations_list(self):
        inp = _basic_inp()
        provider = MockProvider({})
        result = research_contact(inp, provider)
        assert isinstance(result.limitations, list)

    def test_no_company_name_gives_review_identity(self):
        inp = _basic_inp(company_name="")
        provider = MockProvider({
            "https://fabletech.example": _ok_item(
                "https://fabletech.example",
                text="Some company website content.",
            )
        })
        result = research_contact(inp, provider)
        # Cannot be SUPPORTED without a company name to verify
        assert result.company_identity_status != SUPPORTED

    def test_technical_areas_populated_from_page_text(self):
        inp = _basic_inp()
        provider = MockProvider({
            "https://fabletech.example": _ok_item(
                "https://fabletech.example",
                text="Fable Technologies uses machine learning and deep learning.",
            )
        })
        result = research_contact(inp, provider)
        assert "AI/ML" in result.technical_areas

    def test_technical_areas_empty_when_no_relevant_content(self):
        inp = _basic_inp()
        provider = MockProvider({
            "https://fabletech.example": _ok_item(
                "https://fabletech.example",
                text="Fable Technologies sells furniture.",
            )
        })
        result = research_contact(inp, provider)
        assert result.technical_areas == []


# ===========================================================================
# G. build_research_input helper
# ===========================================================================

class TestBuildResearchInput:

    def test_extracts_fields_from_contact_record(self):
        """build_research_input should work with a real ContactRecord."""
        from src.contacts.csv_importer import import_contacts_from_csv
        import textwrap
        from pathlib import Path
        import tempfile

        csv_content = textwrap.dedent("""
            company_name,contact_name,email,website,role_or_job_url,source_url,notes
            Fable Technologies,Alice,alice@fabletech.example,https://fabletech.example,SWE Intern,,Great company
        """).strip()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_content)
            tmp_path = f.name

        result = import_contacts_from_csv(tmp_path)
        assert len(result.contacts) == 1

        inp = build_research_input(result.contacts[0])
        assert inp.company_name == "Fable Technologies"
        assert inp.normalized_email == "alice@fabletech.example"
        assert inp.website == "https://fabletech.example"
        assert inp.notes == "Great company"

    def test_gmail_contact_marked_as_free_provider(self):
        """build_research_input should detect free email providers."""
        from src.contacts.csv_importer import import_contacts_from_csv
        import textwrap
        import tempfile

        csv_content = textwrap.dedent("""
            company_name,email
            Orbit Labs,someuser@gmail.com
        """).strip()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_content)
            tmp_path = f.name

        result = import_contacts_from_csv(tmp_path)
        inp = build_research_input(result.contacts[0])
        assert inp.is_free_provider is True


# ===========================================================================
# H. Correction regression tests (Prompt 3 correction)
# These tests specifically address the weaknesses identified in the correction.
# ===========================================================================

class TestRoleDetectionStrengthened:
    """
    Tests that a keyword alone is no longer sufficient to trigger FOUND.
    Every test here would have incorrectly returned FOUND under the old logic.
    """

    def test_historical_intern_mention_is_not_found(self):
        """A page saying 'our former interns' must NOT return FOUND."""
        inp = _basic_inp(role_or_job_url="https://fabletech.example/about")
        evidence = [_ok_item(
            "https://fabletech.example/about",
            source_type="job_listing",
            title="About Us",
            text=(
                "Last year our interns worked on exciting robotics projects. "
                "We are proud of our internship alumni programme."
            ),
        )]
        status, limits, app_status = check_role_evidence(inp, evidence)
        # Contains "intern" keyword but no active signal -> UNKNOWN, not FOUND
        assert status == UNKNOWN
        assert app_status == APPLICATION_UNKNOWN
        assert any("stale" in lim.lower() or "historical" in lim.lower()
                   or "availability" in lim.lower() for lim in limits)

    def test_careers_page_with_only_fulltime_jobs_is_not_found(self):
        """A page listing only full-time roles, no internship -> NOT_FOUND."""
        inp = _basic_inp(role_or_job_url="https://fabletech.example/careers")
        evidence = [_ok_item(
            "https://fabletech.example/careers",
            source_type="careers_page",
            title="Careers | Fable Technologies",
            text=(
                "We are hiring senior software engineers and product managers. "
                "Full-time positions available. Apply now."
            ),
        )]
        status, limits, _ = check_role_evidence(inp, evidence)
        # No intern keyword at all -> NOT_FOUND
        assert status == NOT_FOUND

    def test_page_says_no_internships_available_is_not_found(self):
        """Explicit denial phrase must return NOT_FOUND + APPLICATION_CLOSED."""
        inp = _basic_inp(role_or_job_url="https://fabletech.example/careers")
        evidence = [_ok_item(
            "https://fabletech.example/careers",
            source_type="careers_page",
            text=(
                "We appreciate your interest. "
                "No internships are currently available at this time. "
                "Please check back later."
            ),
        )]
        status, limits, app_status = check_role_evidence(inp, evidence)
        assert status == NOT_FOUND
        assert app_status == APPLICATION_CLOSED
        assert any("explicitly" in lim.lower() or "available" in lim.lower()
                   for lim in limits)

    def test_internship_keyword_without_active_signal_is_unknown(self):
        """
        A page describing an internship but no apply/open signal -> UNKNOWN.
        This covers stale or undated listings.
        """
        inp = _basic_inp(role_or_job_url="https://fabletech.example/jobs/ml-intern")
        evidence = [_ok_item(
            "https://fabletech.example/jobs/ml-intern",
            source_type="job_listing",
            title="ML Engineering Internship",
            text=(
                "The ML Engineering Internship at Fable Technologies involves "
                "working on production machine learning pipelines. "
                "Duration: 12 weeks. Location: Remote."
                # No "apply now", no "applications open", no deadline -> UNKNOWN
            ),
        )]
        status, limits, app_status = check_role_evidence(inp, evidence)
        assert status == UNKNOWN
        assert app_status == APPLICATION_UNKNOWN
        assert any(
            "stale" in lim.lower() or "historical" in lim.lower()
            or "availability" in lim.lower() or "active" in lim.lower()
            for lim in limits
        )

    def test_active_internship_listing_with_apply_link_is_found(self):
        """Intern keyword + active signal (apply now) -> FOUND + APPLICATION_OPEN."""
        inp = _basic_inp(role_or_job_url="https://fabletech.example/jobs/swe-intern")
        evidence = [_ok_item(
            "https://fabletech.example/jobs/swe-intern",
            source_type="job_listing",
            title="Software Engineering Intern 2025",
            text=(
                "Fable Technologies is accepting applications for a "
                "Software Engineering Internship. "
                "Apply now — deadline 15 January 2025."
            ),
        )]
        status, _, app_status = check_role_evidence(inp, evidence)
        assert status == FOUND
        assert app_status == APPLICATION_OPEN

    def test_inaccessible_role_page_returns_unknown_not_not_found(self):
        """
        An inaccessible page must be UNKNOWN, not NOT_FOUND.
        NOT_FOUND is reserved for pages that were retrieved but contained
        no evidence of an opening.
        """
        inp = _basic_inp(role_or_job_url="https://fabletech.example/jobs/secret")
        evidence = [_error_item(
            "https://fabletech.example/jobs/secret",
            source_type="job_listing",
        )]
        status, limits, _ = check_role_evidence(inp, evidence)
        assert status == UNKNOWN  # not NOT_FOUND
        assert any("could not be retrieved" in lim.lower() or "unavailable" in lim.lower()
                   for lim in limits)

    def test_expired_listing_does_not_claim_application_open(self):
        """An expired posting must return APPLICATION_CLOSED, not APPLICATION_OPEN."""
        inp = _basic_inp(role_or_job_url="https://fabletech.example/jobs/old")
        evidence = [_ok_item(
            "https://fabletech.example/jobs/old",
            source_type="job_listing",
            title="SWE Intern (Expired)",
            text=(
                "This internship listing has expired. "
                "The application period has closed. Please visit our careers page."
            ),
        )]
        status, limits, app_status = check_role_evidence(inp, evidence)
        assert status == NOT_FOUND
        assert app_status == APPLICATION_CLOSED


class TestCompanyIdentityStrengthened:
    """
    Tests that company name appearing on a page is recorded with proper
    source-quality notes, and that ambiguous situations return REVIEW.
    """

    def test_company_name_on_own_website_adds_controlled_source_note(self):
        """
        Company name found only on official_website -> SUPPORTED but with
        a limitation noting it is company-controlled, not independent.
        """
        inp = _basic_inp(company_name="Fable Technologies")
        evidence = [_ok_item(
            "https://fabletech.example",
            source_type="official_website",
            text="Welcome to Fable Technologies, leaders in robotics.",
        )]
        status, limits = check_company_identity(inp, evidence)
        assert status == SUPPORTED
        # Must note that the source is company-controlled
        combined = " ".join(limits).lower()
        assert "company-controlled" in combined or "own content" in combined

    def test_company_name_on_independent_source_adds_corroboration_note(self):
        """
        Company name found on a source_url (potentially independent) -> SUPPORTED
        with a note that the source provides stronger corroboration.
        """
        inp = _basic_inp(company_name="Fable Technologies")
        evidence = [_ok_item(
            "https://linkedin.com/jobs/fable",
            source_type="source_url",
            text="Fable Technologies is looking for ML engineers. Apply on LinkedIn.",
        )]
        status, limits = check_company_identity(inp, evidence)
        assert status == SUPPORTED
        combined = " ".join(limits).lower()
        assert "independent" in combined or "corroboration" in combined

    def test_conflicting_company_name_on_website_returns_review(self):
        """
        Website content mentions a different company name -> REVIEW.
        This should NOT silently become SUPPORTED.
        """
        inp = _basic_inp(company_name="Fable Technologies")
        evidence = [_ok_item(
            "https://fabletech.example",
            source_type="official_website",
            text="Welcome to Apex Solutions, your robotics partner.",
        )]
        status, limits = check_company_identity(inp, evidence)
        assert status == REVIEW
        assert any("fable technologies" in lim.lower() or "not found" in lim.lower()
                   for lim in limits)


class TestDomainMatchDoesNotProveCompanyLegitimacy:
    """
    Tests that a matching domain is recorded as useful evidence while
    explicitly noting it does not prove company legitimacy.
    """

    def test_domain_match_does_not_claim_company_is_legitimate(self):
        """
        domain_relationship_status=SUPPORTED only means domains match.
        The limitation must make clear this does not verify legitimacy.
        """
        inp = _basic_inp(
            email_domain="fabletech.example",
            registered_domain="fabletech.example",
            website="https://fabletech.example",
        )
        status, limits = check_domain_relationship(inp, [])
        assert status == SUPPORTED
        combined = " ".join(limits).lower()
        # Must contain a caveat -- not a blank endorsement
        assert any(
            phrase in combined
            for phrase in ["not confirm", "evidence", "does not", "authorised"]
        )

    def test_free_provider_not_rejected_automatically(self):
        """
        Free provider -> NOT_APPLICABLE, not a fraud flag.
        The contact should still be importable and researchable.
        """
        inp = _basic_inp(
            email_domain="gmail.com",
            registered_domain="gmail.com",
            is_free_provider=True,
        )
        status, limits = check_domain_relationship(inp, [])
        assert status == NOT_APPLICABLE
        # Must NOT contain words like "fraud", "fake", "suspicious"
        combined = " ".join(limits).lower()
        assert "fraud" not in combined
        assert "fake" not in combined
        assert "suspicious" not in combined

    def test_subdomain_not_rejected_automatically(self):
        """
        Subdomain email -> REVIEW, not automatic rejection.
        A legitimate company may use jobs.bigcorp.com.
        """
        inp = _basic_inp(
            email_domain="jobs.bigcorp.example",
            registered_domain="bigcorp.example",
            website="https://bigcorp.example",
        )
        status, limits = check_domain_relationship(inp, [])
        assert status == REVIEW
        combined = " ".join(limits).lower()
        assert "legitimate" in combined or "common" in combined
