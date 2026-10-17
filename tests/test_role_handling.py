"""
tests/test_role_handling.py
----------------------------
Regression tests verifying clean separation of plain-text role titles
from genuine HTTP/HTTPS job URLs in research and workflow.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.contacts.csv_importer import ContactRecord
from src.contacts.email_parser import parse_email
from src.outreach.models import WorkflowMode
from src.research.checks import _is_http_url, check_role_evidence
from src.research.models import (
    EvidenceItem,
    ResearchInput,
    ResearchResult,
    RETRIEVAL_OK,
    FOUND,
    NOT_FOUND,
    UNKNOWN,
    APPLICATION_OPEN,
    APPLICATION_UNKNOWN,
)
from src.research.providers import ResearchProvider
from src.research.researcher import build_research_input, research_contact
from src.workflow.engine import OutreachWorkflow


class RecordingProvider(ResearchProvider):
    """Test provider that records every fetched URL."""

    def __init__(self, responses: dict[str, EvidenceItem] | None = None) -> None:
        self.fetched_urls: list[tuple[str, str, list[str]]] = []
        self.responses = responses or {}

    def fetch(self, url: str, source_type: str, supports: list[str]) -> EvidenceItem:
        self.fetched_urls.append((url, source_type, list(supports)))
        if url in self.responses:
            return self.responses[url]
        return EvidenceItem(
            source_url=url,
            source_type=source_type,
            extracted_text="Default page text",
            retrieval_status=RETRIEVAL_OK,
            supports=list(supports),
        )


def _make_contact(role_or_job_url: str = "") -> ContactRecord:
    return ContactRecord(
        row_number=2,
        company_name="Apex AI Labs",
        contact_name="Dr. Jane Smith",
        email_result=parse_email("jane.smith@example.com"),
        website="https://example.com",
        role_or_job_url=role_or_job_url,
        source_url="",
        notes="",
    )


class TestIsHttpUrlHelper:

    def test_http_and_https_recognized(self):
        assert _is_http_url("http://example.com/job") is True
        assert _is_http_url("https://example.com/careers/ai-intern") is True
        assert _is_http_url("  HTTPS://company.org/apply  ") is True

    def test_plain_text_roles_not_recognized_as_urls(self):
        assert _is_http_url("AI/ML Intern") is False
        assert _is_http_url("Data Science Intern") is False
        assert _is_http_url("GenAI Research Fellow") is False
        assert _is_http_url("") is False
        assert _is_http_url(None) is False


class TestRoleTitleHandlingInResearch:

    def test_plain_text_role_not_fetched_as_url(self):
        """Passing a plain-text role title must not trigger provider.fetch for that string."""
        provider = RecordingProvider()
        inp = ResearchInput(
            company_name="Apex AI Labs",
            website="https://example.com",
            role_or_job_url="AI/ML Intern",
        )
        res = research_contact(inp, provider=provider)

        # Provider should ONLY have fetched website, not "AI/ML Intern"
        fetched_urls = [u for u, _, _ in provider.fetched_urls]
        assert "https://example.com" in fetched_urls
        assert "AI/ML Intern" not in fetched_urls

        # Limitations must NOT claim a role URL could not be retrieved
        assert not any("AI/ML Intern" in lim and "could not be retrieved" in lim for lim in res.limitations)

    def test_build_research_input_strips_non_url_role(self):
        """build_research_input must leave role_or_job_url empty when given a plain role title."""
        contact = _make_contact(role_or_job_url="AI/ML Intern")
        inp = build_research_input(contact)
        assert inp.role_or_job_url == ""

    def test_valid_http_job_url_is_fetched(self):
        """A valid HTTP/HTTPS URL in role_or_job_url must be fetched as a job_listing."""
        job_url = "https://example.com/careers/intern"
        provider = RecordingProvider()
        inp = ResearchInput(
            company_name="Apex AI Labs",
            website="https://example.com",
            role_or_job_url=job_url,
        )
        res = research_contact(inp, provider=provider)

        fetched_urls = [u for u, _, _ in provider.fetched_urls]
        assert job_url in fetched_urls
        role_items = [e for e in res.evidence if e.source_url == job_url]
        assert len(role_items) == 1
        assert role_items[0].source_type == "job_listing"

    def test_check_role_evidence_ignores_non_url_strings(self):
        """check_role_evidence should not treat plain-text titles as failed role URLs."""
        inp = ResearchInput(role_or_job_url="AI/ML Intern")
        evidence = [
            EvidenceItem(
                source_url="https://example.com",
                source_type="official_website",
                extracted_text="We have summer internships available. Apply now!",
                retrieval_status=RETRIEVAL_OK,
            )
        ]
        role_status, limitations, app_status = check_role_evidence(inp, evidence)
        assert role_status == FOUND
        assert app_status == APPLICATION_OPEN
        assert not any("AI/ML Intern" in lim for lim in limitations)


class TestWorkflowRoleHandling:

    def test_workflow_target_role_without_url(self, tmp_path):
        """CLI passing --role 'AI/ML Intern' does not perform invalid URL fetches and preserves role in draft."""
        dummy_cv = tmp_path / "resume.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy cv content")

        provider = RecordingProvider({
            "https://example.com": EvidenceItem(
                source_url="https://example.com",
                source_type="official_website",
                extracted_text="Apex AI Labs builds AI/ML and deep learning solutions.",
                retrieval_status=RETRIEVAL_OK,
            )
        })

        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )

        contact = _make_contact(role_or_job_url="")
        result = workflow.process_contact(
            contact=contact,
            target_role="AI/ML Intern",
            research_provider=provider,
        )

        # 1. No invalid URL fetch attempted
        fetched_urls = [u for u, _, _ in provider.fetched_urls]
        assert "AI/ML Intern" not in fetched_urls
        assert "https://example.com" in fetched_urls

        # 2. Draft uses target_role in subject
        assert result.draft is not None
        assert "AI/ML Intern" in result.draft.subject

        # 3. Status is PROCEED (or valid review)
        assert result.status == "PROCEED"

    def test_workflow_valid_http_job_url(self, tmp_path):
        """Passing a job URL via contact or argument fetches it and uses it in workflow."""
        dummy_cv = tmp_path / "resume.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy cv content")

        job_url = "https://example.com/careers/ai-intern"
        provider = RecordingProvider({
            "https://example.com": EvidenceItem(
                source_url="https://example.com",
                source_type="official_website",
                extracted_text="Apex AI Labs. AI/ML.",
                retrieval_status=RETRIEVAL_OK,
            ),
            job_url: EvidenceItem(
                source_url=job_url,
                source_type="job_listing",
                extracted_text="Machine learning internship. Applications open. Apply now!",
                retrieval_status=RETRIEVAL_OK,
            )
        })

        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )

        contact = _make_contact(role_or_job_url=job_url)
        result = workflow.process_contact(
            contact=contact,
            target_role="AI/ML Intern",
            research_provider=provider,
        )

        # Job URL was fetched
        fetched_urls = [u for u, _, _ in provider.fetched_urls]
        assert job_url in fetched_urls

        # Verified role status is FOUND
        assert result.verification.match.case_type == "SPECIFIC_INTERNSHIP"

    def test_workflow_missing_role_and_job_url(self, tmp_path):
        """When neither role title nor job URL is supplied, workflow defaults safely without invalid fetches."""
        dummy_cv = tmp_path / "resume.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy cv content")

        provider = RecordingProvider({
            "https://example.com": EvidenceItem(
                source_url="https://example.com",
                source_type="official_website",
                extracted_text="Apex AI Labs builds machine learning applications.",
                retrieval_status=RETRIEVAL_OK,
            )
        })

        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )

        contact = _make_contact(role_or_job_url="")
        result = workflow.process_contact(
            contact=contact,
            target_role="",
            research_provider=provider,
        )

        fetched_urls = [u for u, _, _ in provider.fetched_urls]
        assert fetched_urls == ["https://example.com"]
        assert result.draft is not None
        # Defaults to generic inquiry
        assert "Inquiry" in result.draft.subject
