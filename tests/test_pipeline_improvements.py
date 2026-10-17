"""
tests/test_pipeline_improvements.py
-----------------------------------
Tests validating SmartMail's core pipeline improvements:
- Custom domain with discoverable website and fallback.
- Custom domain whose website cannot be found.
- Gmail/Hotmail contact with supplied company name.
- Free-provider contact with unknown company identity preserved for human review.
- Founder, recruiter, and generic inbox handling without HR bias.
- Malformed email detection and error handling.
- AI company, clearly non-AI company, and inconclusive evidence classification.
- User notes mentioning GenAI for non-AI company NEVER establishing technical relevance.
- Website request failure and empty page content handling.
- Batch processing: separate counts, review queue, resilience to individual failures.
- No automatic email sending safeguard.

NOTE: All tests use deterministic MockResearchProvider fixtures clearly labeled
as mocked test doubles without live network access.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from src.contacts.csv_importer import (
    ContactRecord,
    import_contacts_from_csv,
)
from src.contacts.email_parser import parse_email
from src.outreach.models import DraftStatus, WorkflowMode
from src.outreach.send_log import SendLog
from src.research.models import (
    EvidenceItem,
    ResearchInput,
    ResearchResult,
    RETRIEVAL_ERROR,
    RETRIEVAL_OK,
    SUPPORTED,
    REVIEW,
    UNKNOWN,
    NOT_APPLICABLE,
)
from src.research.providers import ResearchProvider
from src.research.researcher import build_research_input, research_contact
from src.research.checks import (
    check_company_identity,
    check_domain_relationship,
    check_technical_relevance,
)
from src.verification.matcher import match_profile
from src.verification.models import BLOCKED, PROCEED, RELEVANT, NOT_A_MATCH
from src.verification.pipeline import run_verification
from src.verification.verifier import verify
from src.workflow.engine import OutreachWorkflow, BatchResult, WorkflowResult


# ---------------------------------------------------------------------------
# Clearly labeled test double (mock provider)
# ---------------------------------------------------------------------------

class MockResearchProvider(ResearchProvider):
    """
    Mocked test double for ResearchProvider.
    Deterministic in-memory URL responses without network calls.
    """
    def __init__(self, responses: dict[str, EvidenceItem] | None = None) -> None:
        self.responses = responses or {}
        self.requested_urls: list[str] = []

    def fetch(self, url: str, source_type: str, supports: list[str]) -> EvidenceItem:
        self.requested_urls.append(url)
        if url in self.responses:
            item = self.responses[url]
            item.supports = list(supports)
            item.source_type = source_type
            return item
        return EvidenceItem(
            source_url=url,
            source_type=source_type,
            retrieval_status=RETRIEVAL_ERROR,
            retrieval_note="MockConnectionError",
            supports=list(supports),
            limitations=[f"Mocked test double: URL '{url}' not found in mock responses."],
        )


class MockEmailSenderProvider:
    """Mocked email delivery provider to verify sending safeguards."""
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []

    def send_email(self, recipient: str, subject: str, body: str, attachment_path: str) -> None:
        self.sent_messages.append({
            "recipient": recipient,
            "subject": subject,
            "body": body,
            "attachment_path": attachment_path,
        })


def _make_ok_item(url: str, text: str, title: str = "", source_type: str = "official_website") -> EvidenceItem:
    return EvidenceItem(
        source_url=url,
        source_type=source_type,
        title=title or "Test Page Title",
        extracted_text=text,
        retrieved_at="2026-10-14T12:00:00+00:00",
        retrieval_status=RETRIEVAL_OK,
        retrieval_note="HTTP 200 (mocked)",
    )


# ---------------------------------------------------------------------------
# 1. Custom Domain Discovery Tests
# ---------------------------------------------------------------------------

class TestCustomDomainDiscovery:
    def test_custom_domain_with_discoverable_website(self):
        """When no website is given in CSV, candidate website is safely discovered from domain."""
        inp = ResearchInput(
            company_name="Nova AI",
            normalized_email="team@nova-ai.example",
            email_domain="nova-ai.example",
            registered_domain="nova-ai.example",
            is_free_provider=False,
            website="",  # Missing website
        )
        mock_provider = MockResearchProvider({
            "https://nova-ai.example": _make_ok_item(
                "https://nova-ai.example",
                text="Nova AI builds next-generation generative AI and agentic AI architectures.",
                title="Nova AI — Official Home",
            )
        })
        res = research_contact(inp, provider=mock_provider)

        assert res.company_identity_status == SUPPORTED
        assert res.domain_relationship_status == SUPPORTED
        assert "Generative AI" in res.technical_areas
        assert "Agentic AI" in res.technical_areas
        assert any(e.source_type == "discovered_website" for e in res.evidence)

    def test_custom_domain_whose_website_cannot_be_found(self):
        """When candidate website probes fail, discovery records error and halts safely as BLOCKED."""
        inp = ResearchInput(
            company_name="Ghost Corp",
            normalized_email="contact@ghostcorp.example",
            email_domain="ghostcorp.example",
            registered_domain="ghostcorp.example",
            is_free_provider=False,
            website="",
        )
        mock_provider = MockResearchProvider({})  # All URLs fail
        res = research_contact(inp, provider=mock_provider)

        assert res.company_identity_status == UNKNOWN
        assert res.domain_relationship_status == UNKNOWN
        
        vr = verify(res, inp)
        assert vr.overall_status == BLOCKED
        assert "Company identity and domain relationship are both unresolved" in vr.blocking_reasons[0]


# ---------------------------------------------------------------------------
# 2. Free Email Provider Handling Tests
# ---------------------------------------------------------------------------

class TestFreeProviderHandling:
    def test_gmail_contact_with_supplied_company_name(self, tmp_path: Path):
        """Gmail contact with known company proceeds to human review without being blocked."""
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 test")

        contact = ContactRecord(
            row_number=2,
            company_name="Veritas Robotics",
            contact_name="Sarah",
            email_result=parse_email("sarah.recruiter@gmail.com"),
            website="https://veritas.example",
            role_or_job_url="AI/ML Intern",
            source_url="",
            notes="",
        )
        mock_provider = MockResearchProvider({
            "https://veritas.example": _make_ok_item(
                "https://veritas.example",
                text="Veritas Robotics develops machine learning, computer vision, and neural network robotics.",
                title="Veritas Robotics",
            )
        })

        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            dry_run=True,
        )
        wf_res = workflow.process_contact(contact, research_provider=mock_provider)

        # Contact is on a free provider -> requires human review, not blocked
        assert wf_res.status == "REVIEW"
        assert wf_res.draft is not None
        assert wf_res.draft.review_required is True
        assert wf_res.draft.recipient == "sarah.recruiter@gmail.com"

    def test_free_provider_contact_with_unknown_company_identity(self, tmp_path: Path):
        """Free-provider contact without company details is preserved for human review, not discarded."""
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 test")

        contact = ContactRecord(
            row_number=3,
            company_name="",
            contact_name="",
            email_result=parse_email("Hr.qrm@hotmail.com"),
            website="",
            role_or_job_url="",
            source_url="",
            notes="",
        )
        workflow = OutreachWorkflow(uploaded_cv_path=dummy_cv, dry_run=True)
        wf_res = workflow.process_contact(contact, research_provider=MockResearchProvider({}))

        # Preserved for review, not silently dropped or falsely verified
        assert wf_res.status == "REVIEW"
        assert wf_res.draft is not None
        assert wf_res.draft.review_required is True


# ---------------------------------------------------------------------------
# 3. Founder, Recruiter, and Generic Inbox Tests
# ---------------------------------------------------------------------------

class TestContactRoleTypes:
    @pytest.mark.parametrize("email", [
        "founder@ai-startup.example",
        "rec@alojaimi.example",
        "careers@cloudtech.example",
        "info@deepmind-labs.example",
        "engineering-interns@vision.example",
    ])
    def test_arbitrary_inbox_roles_handled_equally_without_hr_bias(self, email: str, tmp_path: Path):
        """Pipeline does not discriminate against founders, generic inboxes, or non-HR addresses."""
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 test")

        parsed = parse_email(email)
        assert parsed.is_syntax_valid is True

        contact = ContactRecord(
            row_number=2,
            company_name="AI Startup",
            contact_name="Alex",
            email_result=parsed,
            website=f"https://{parsed.registered_domain}",
            role_or_job_url="",
            source_url="",
            notes="",
        )
        mock_provider = MockResearchProvider({
            f"https://{parsed.registered_domain}": _make_ok_item(
                f"https://{parsed.registered_domain}",
                text="AI Startup builds deep learning and machine learning systems.",
                title="AI Startup",
            )
        })

        workflow = OutreachWorkflow(uploaded_cv_path=dummy_cv, dry_run=True)
        wf_res = workflow.process_contact(contact, research_provider=mock_provider)

        assert wf_res.status == PROCEED
        assert wf_res.draft is not None
        assert wf_res.draft.recipient == email


# ---------------------------------------------------------------------------
# 4. Malformed Email Handling Tests
# ---------------------------------------------------------------------------

class TestMalformedEmailHandling:
    def test_malformed_email_captured_cleanly(self, tmp_path: Path):
        csv_file = tmp_path / "contacts.csv"
        csv_file.write_text("email,company_name\nnot-an-email,Bad Corp\nvalid@goodcorp.example,Good Corp\n", encoding="utf-8")

        res = import_contacts_from_csv(csv_file)
        assert len(res.errors) == 1
        assert res.errors[0].row_number == 2
        assert "failed syntax validation" in res.errors[0].reason
        assert len(res.contacts) == 1
        assert res.contacts[0].email_result.normalized_email == "valid@goodcorp.example"


# ---------------------------------------------------------------------------
# 5. Technical Relevance & User Notes Isolation Tests
# ---------------------------------------------------------------------------

class TestTechnicalRelevanceAndNotesIsolation:
    def test_ai_company_classified_relevant(self):
        ev = [_make_ok_item("https://test.example", text="We specialize in generative ai, rag, and agentic workflows.")]
        tech_areas = check_technical_relevance(ev)
        assert "Generative AI" in tech_areas
        assert "Agentic AI" in tech_areas

    def test_clearly_non_ai_company(self):
        ev = [_make_ok_item("https://test.example", text="We are a construction and civil engineering contractor. We are not an AI/ML company.")]
        tech_areas = check_technical_relevance(ev)
        assert tech_areas == []

    def test_inconclusive_company_evidence(self):
        ev = [_make_ok_item("https://test.example", text="Welcome to our official website. Contact us for inquiries.")]
        tech_areas = check_technical_relevance(ev)
        assert tech_areas == []

    def test_user_notes_mentioning_genai_for_non_ai_company_never_establishes_relevance(self):
        """User notes MUST NEVER be treated as external evidence establishing company AI relevance."""
        ev = [
            _make_ok_item("https://accounting.example", text="Certified public accounting and audit services."),
            EvidenceItem(
                source_url="",
                source_type="user_supplied",
                title="User notes",
                extracted_text="Candidate notes: Target GenAI and LLM internship at this firm.",
                retrieval_status=RETRIEVAL_OK,
                retrieval_note="user supplied",
            ),
        ]
        # Technical relevance must ignore user_supplied evidence
        tech_areas = check_technical_relevance(ev)
        assert tech_areas == []
        assert "Generative AI" not in tech_areas
        assert "AI/ML" not in tech_areas


# ---------------------------------------------------------------------------
# 6. Page Retrieval Failure & Empty Content Tests
# ---------------------------------------------------------------------------

class TestRetrievalFailuresAndEmptyContent:
    def test_website_request_failure_handled_as_error_limitation(self):
        item = EvidenceItem(
            source_url="https://broken.example",
            source_type="official_website",
            retrieval_status=RETRIEVAL_ERROR,
            retrieval_note="HTTP 500",
            limitations=["Server returned HTTP 500."],
        )
        status, lims = check_company_identity(
            ResearchInput(company_name="Broken Corp", website="https://broken.example"),
            [item],
        )
        assert status == UNKNOWN
        assert "No external pages could be retrieved" in lims[0]

    def test_empty_page_content_handled_as_inconclusive(self):
        item = EvidenceItem(
            source_url="https://empty.example",
            source_type="official_website",
            title="",
            extracted_text="",
            retrieval_status=RETRIEVAL_OK,
        )
        status, lims = check_company_identity(
            ResearchInput(company_name="Empty Corp", website="https://empty.example"),
            [item],
        )
        assert status == UNKNOWN
        assert "contained no visible text" in lims[0]


# ---------------------------------------------------------------------------
# 7. Batch Workflow Tests
# ---------------------------------------------------------------------------

class TestBatchWorkflow:
    def test_batch_processing_comprehensive_counts_and_review_queue(self, tmp_path: Path):
        """Batch processes 4 contacts: 1 proceed, 1 review, 1 blocked, 1 malformed."""
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 test")

        csv_content = (
            "email,company_name,website,role_or_job_url,notes\n"
            "ai@hyperai.example,HyperAI,https://hyperai.example,,AI research firm\n"
            "recruiter@gmail.com,Alpha Dynamics,https://alpha.example,,Recruiter contact\n"
            "nobody@unreachable.example,Ghost Co,,,No website available\n"
            "malformed-email-row,Bad Data,,,\n"
        )
        batch_csv = tmp_path / "batch_contacts.csv"
        batch_csv.write_text(csv_content, encoding="utf-8")

        mock_provider = MockResearchProvider({
            "https://hyperai.example": _make_ok_item(
                "https://hyperai.example",
                text="HyperAI conducts machine learning and artificial intelligence research.",
                title="HyperAI",
            ),
            "https://alpha.example": _make_ok_item(
                "https://alpha.example",
                text="Alpha Dynamics builds deep learning robotics.",
                title="Alpha Dynamics",
            ),
        })

        mock_sender = MockEmailSenderProvider()
        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            email_sender_provider=mock_sender,
            dry_run=True,
        )

        batch_result = workflow.process_csv(batch_csv, research_provider=mock_provider)

        assert isinstance(batch_result, BatchResult)
        assert batch_result.total_rows == 4
        assert batch_result.processed_count == 3  # 3 valid email rows processed
        assert batch_result.drafted_count == 2    # HyperAI (proceed) and Alpha Dynamics (review)
        assert batch_result.review_required_count == 1  # Gmail recruiter requires review
        assert batch_result.blocked_count == 1    # Ghost Co is blocked
        assert batch_result.failed_count == 1     # Malformed row
        assert len(batch_result.review_queue) == 1
        assert batch_result.review_queue[0].contact_email == "recruiter@gmail.com"

        # Verification of summary string
        summary = batch_result.summary()
        assert "Total input rows:      4" in summary
        assert "Drafts created:        2" in summary
        assert "Review required:       1" in summary
        assert "Blocked contacts:      1" in summary
        assert "Failed / Invalid rows: 1" in summary

        # CRITICAL SAFEGUARD: Zero emails sent automatically during batch
        assert len(mock_sender.sent_messages) == 0

    def test_single_failed_contact_does_not_abort_rest_of_batch(self, tmp_path: Path):
        """If one contact raises an unexpected exception, subsequent contacts must continue processing."""
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 test")

        csv_content = (
            "email,company_name,website\n"
            "c1@first.example,First Corp,https://first.example\n"
            "c2@crash.example,Crash Corp,https://crash.example\n"
            "c3@third.example,Third Corp,https://third.example\n"
        )
        batch_csv = tmp_path / "batch_crash.csv"
        batch_csv.write_text(csv_content, encoding="utf-8")

        class FaultyProvider(ResearchProvider):
            def fetch(self, url: str, source_type: str, supports: list[str]) -> EvidenceItem:
                if "crash" in url:
                    raise RuntimeError("Simulated unhandled network/hardware crash!")
                return _make_ok_item(url, text="Machine learning and AI systems.")

        workflow = OutreachWorkflow(uploaded_cv_path=dummy_cv, dry_run=True)
        batch_result = workflow.process_csv(batch_csv, research_provider=FaultyProvider())

        assert batch_result.total_rows == 3
        assert batch_result.processed_count == 3
        assert batch_result.failed_count == 1  # c2 failed
        assert batch_result.drafted_count == 2  # c1 and c3 drafted successfully
        assert any(e["email"] == "c2@crash.example" for e in batch_result.errors)
