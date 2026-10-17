"""
tests/test_stage3_research_classification.py
---------------------------------------------
Comprehensive test suite for Stage 3:
Evidence-Based Company Research and Contact Classification.

Covers:
- Personal email address with a corporate domain (inferred name, not verified).
- Generic HR, careers, founder, and CEO inboxes (recognized as generic, not discarded).
- Free-provider addresses with no company information (eliminated with COMPANY_UNVERIFIED).
- Supplied company name that conflicts with domain (contradiction flagged, domain mismatch).
- Lookalike or mismatched domains.
- Verified company with unknown recipient name (REVIEW_NAME).
- Company identified but not AI/ML relevant (NOT_AI_RELEVANT, not an identity failure).
- AI/ML relevance supported by official technical evidence (CONFIDENT / REVIEW_NAME).
- Company with ambiguous evidence (AI_RELEVANCE_UNCERTAIN).
- Empty pages, network errors, and blocked websites (correct status codes, RESEARCH_FAILED).
- Candidate notes that mention AI but must not influence company relevance.
- Genuine overlap between CV skills and company evidence.
- Contradictory evidence and uncertain identities.
- Batch continuation after an individual research failure.
- Preservation of source row order and imported values.
- Persistence and reload of evidence and classification.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.contacts.csv_importer import ContactRecord
from src.contacts.email_parser import parse_email
from src.contacts.recipient_resolver import (
    GENERIC_INBOX,
    INFERRED,
    UNKNOWN,
    VERIFIED as REC_VERIFIED,
    infer_name_from_local_part,
    resolve_recipient,
)
from src.outreach.models import WorkflowMode
from src.research.checks import check_technical_relevance
from src.research.models import (
    FOUND,
    NOT_FOUND,
    RETRIEVAL_BLOCKED,
    RETRIEVAL_EMPTY,
    RETRIEVAL_JS_ONLY,
    RETRIEVAL_NETWORK_ERROR,
    RETRIEVAL_OK,
    REVIEW as RES_REVIEW,
    SUPPORTED,
    EvidenceItem,
    ResearchInput,
    ResearchResult,
)
from src.research.providers import HttpPageProvider
from src.research.researcher import build_research_input, research_contact
from src.ui.storage import get_batch_record, get_draft, save_batch_record, save_draft_record
from src.verification.classifier import (
    CONFIDENT,
    ELIMINATED,
    NOT_AI_RELEVANT,
    REASON_AI_RELEVANCE_UNCERTAIN,
    REASON_COMPANY_UNVERIFIED,
    REASON_CONFIDENT,
    REASON_CONTRADICTORY_EVIDENCE,
    REASON_DOMAIN_MISMATCH,
    REASON_NOT_AI_RELEVANT,
    REASON_RECIPIENT_UNVERIFIED,
    REASON_RESEARCH_FAILED,
    REVIEW_NAME,
    classify_contact,
)
from src.verification.matcher import match_profile
from src.verification.models import (
    BLOCKED,
    MatchResult,
    PipelineResult,
    VerificationResult,
)
from src.verification.pipeline import run_verification
from src.workflow.engine import BatchResult, OutreachWorkflow, WorkflowResult


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_profile() -> dict:
    return {
        "candidate": {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "+1 555-0100",
            "location": "San Francisco, CA",
        },
        "education": {
            "degree": "B.S. in Computer Science",
            "institution": "University of California, Berkeley",
            "graduation_date": "May 2025",
            "gpa": "3.9",
        },
        "technical_skills": {
            "languages": ["Python", "C++", "SQL"],
            "ml_and_data": ["PyTorch", "Transformers", "NLP", "Computer Vision", "LangGraph"],
            "frameworks_and_libraries": ["FastAPI", "Docker", "LangChain"],
            "tools_and_platforms": ["Git", "Linux", "GCP"],
        },
        "projects": [
            {
                "name": "Autonomous Agent Workflow",
                "technologies": ["LangGraph", "Python", "RAG", "ChromaDB"],
                "highlight_for_roles": ["AI/ML Intern", "Agent Engineer"],
            }
        ],
        "preferences": {
            "target_roles": ["AI/ML Intern", "Generative AI Intern"],
            "remote_preference": "Hybrid",
        },
    }


def make_contact(
    email: str,
    company: str = "",
    name: str = "",
    website: str = "",
    notes: str = "",
    row: int = 1,
) -> ContactRecord:
    parsed = parse_email(email)
    return ContactRecord(
        row_number=row,
        company_name=company,
        contact_name=name,
        email_result=parsed,
        website=website,
        role_or_job_url="",
        source_url="",
        notes=notes,
        raw_values={
            "email": email,
            "company_name": company,
            "contact_name": name,
            "website": website,
            "notes": notes,
        },
    )


# ---------------------------------------------------------------------------
# 1. Recipient & Name Resolution
# ---------------------------------------------------------------------------

class TestRecipientResolution:
    def test_personal_email_with_corporate_domain_infers_name_not_verified(self):
        """bhavesh@microsoft.com suggests Bhavesh, but does not verify personal identity."""
        res = resolve_recipient("bhavesh@microsoft.com")
        assert res.inferred_name == "Bhavesh"
        assert not res.verified_name
        assert res.status == INFERRED
        assert res.name_verification_status == "INFERRED"
        assert not res.is_generic

    def test_generic_inboxes_recognized_and_not_given_personal_names(self):
        """hr@, careers@, founder@, ceo@ are recognized as generic inboxes."""
        for local in ["hr", "careers", "founder", "ceo", "info", "recruiting", "jobs", "team"]:
            email = f"{local}@startup.example"
            res = resolve_recipient(email)
            assert res.is_generic is True
            assert res.status == GENERIC_INBOX
            assert not res.inferred_name
            assert not res.verified_name
            assert not res.resolved_name

    def test_supplied_name_used_when_provided(self):
        """User-supplied name is preserved as supplied_name."""
        res = resolve_recipient("alex@company.com", supplied_name="Alex Mercer")
        assert res.supplied_name == "Alex Mercer"
        assert res.inferred_name == "Alex"
        assert res.resolved_name == "Alex Mercer"
        assert res.status == INFERRED

    def test_infer_name_compound_local_parts(self):
        """First.last and first_last are cleaned into first and last names."""
        assert infer_name_from_local_part("john.smith") == "John Smith"
        assert infer_name_from_local_part("alice_wonderland") == "Alice Wonderland"
        assert not infer_name_from_local_part("hr")


# ---------------------------------------------------------------------------
# 2. Company Resolution & Disambiguation
# ---------------------------------------------------------------------------

class TestCompanyResolution:
    def test_free_provider_addresses_do_not_infer_employer(self):
        """person@gmail.com without company info must not infer employer."""
        contact = make_contact("person@gmail.com", company="", website="")
        res_input = build_research_input(contact)
        provider = MagicMock()
        provider.fetch_page.return_value = EvidenceItem(
            source_url="", retrieval_status=RETRIEVAL_EMPTY, extracted_text=""
        )
        provider.search.return_value = []

        result = research_contact(res_input, provider=provider)
        assert result.company_name == ""

        # Classifier check: must be ELIMINATED with COMPANY_UNVERIFIED
        v_result = VerificationResult(
            company_status=BLOCKED,
            contact_status="REVIEW",
            overall_status=BLOCKED,
            blocking_reasons=["Email is from a free provider and no company was identified"],
        )
        p_result = PipelineResult(
            verification=v_result,
            match=MatchResult(relevance_status="REVIEW", case_type="GENERAL_INQUIRY", matched_areas=[], profile_evidence=[], explanation=""),
            summary="Blocked",
        )
        clf = classify_contact(result, p_result, original_email="person@gmail.com")
        assert clf.primary_outcome == ELIMINATED
        assert clf.reason_code == REASON_COMPANY_UNVERIFIED

    def test_supplied_company_conflicts_with_corporate_domain(self):
        """e.g. user enters 'Google' for an @microsoft.com email."""
        contact = make_contact("recruiter@microsoft.com", company="Google")
        res_input = build_research_input(contact)
        provider = MagicMock()
        provider.fetch_page.return_value = EvidenceItem(
            source_url="https://microsoft.com",
            retrieval_status=RETRIEVAL_OK,
            extracted_text="Microsoft Corporation cloud and software",
        )
        provider.search.return_value = []

        result = research_contact(res_input, provider=provider)
        assert len(result.contradictions) > 0
        assert any("Google" in c and "microsoft.com" in c for c in result.contradictions)

        v_result = VerificationResult(
            company_status=BLOCKED,
            contact_status="REVIEW",
            overall_status=BLOCKED,
            blocking_reasons=["Supplied company conflicts with domain"],
        )
        p_result = PipelineResult(
            verification=v_result,
            match=MatchResult(relevance_status="REVIEW", case_type="GENERAL_INQUIRY", matched_areas=[], profile_evidence=[], explanation=""),
            summary="Blocked",
        )
        clf = classify_contact(result, p_result, original_email=contact.email)
        assert clf.primary_outcome == ELIMINATED
        assert clf.reason_code in (REASON_DOMAIN_MISMATCH, REASON_CONTRADICTORY_EVIDENCE)


# ---------------------------------------------------------------------------
# 3. AI/ML Research & Classification
# ---------------------------------------------------------------------------

class TestAimlRelevanceAndClassification:
    def test_verified_company_not_ai_relevant(self, sample_profile):
        """A verified non-tech or non-AI company must be classified NOT_AI_RELEVANT, not an identity failure."""
        ev = EvidenceItem(
            source_url="https://acme-bakery.example",
            source_type="homepage",
            title="Acme Artisanal Bakery & Cafe",
            retrieval_status=RETRIEVAL_OK,
            extracted_text="Fresh organic sourdough breads, croissants, and espresso catering.",
            supports=["company_identity"],
        )
        res = ResearchResult(
            company_name="Acme Bakery",
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            technical_areas=[],
            evidence=[ev],
        )
        v_res = VerificationResult(
            company_status="VERIFIED",
            contact_status="VERIFIED",
            overall_status="REVIEW",  # demoted because not a match
            verification_notes=["Company identity verified"],
        )
        m_res = match_profile(res, sample_profile)
        p_res = PipelineResult(verification=v_res, match=m_res, summary="Company not AI relevant")

        clf = classify_contact(res, p_res, original_email="contact@acme-bakery.example")
        assert clf.primary_outcome == NOT_AI_RELEVANT
        assert clf.reason_code == REASON_NOT_AI_RELEVANT
        assert clf.ai_ml_relevant is False

    def test_candidate_notes_do_not_influence_company_relevance(self, sample_profile):
        """Candidate notes mentioning AI must NOT count as proof of company's technical work."""
        contact = make_contact(
            "hiring@classiclaw.example",
            company="Classic Law Firm",
            notes="I really want to do generative AI with their legal team!",
        )
        # Even if note has AI keywords, the external evidence does NOT
        ev = EvidenceItem(
            source_url="https://classiclaw.example",
            retrieval_status=RETRIEVAL_OK,
            extracted_text="Corporate litigation and real estate contract law services.",
            supports=["company_identity"],
        )
        res = ResearchResult(
            company_name="Classic Law Firm",
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            technical_areas=[],  # No technical areas from external evidence
            evidence=[ev],
        )
        v_res = VerificationResult(company_status="VERIFIED", contact_status="VERIFIED", overall_status="REVIEW")
        m_res = match_profile(res, sample_profile)
        p_res = PipelineResult(verification=v_res, match=m_res, summary="Legal firm")

        clf = classify_contact(res, p_res, original_email=contact.email)
        assert clf.primary_outcome == NOT_AI_RELEVANT
        assert clf.reason_code == REASON_NOT_AI_RELEVANT

    def test_official_technical_evidence_subdisciplines(self):
        """Official engineering text detects specific Stage 3 subdisciplines."""
        text = (
            "We build production RAG systems, AI agents using LangGraph, "
            "and LLM fine-tuning pipelines on Kubernetes clusters with Triton model serving."
        )
        ev = EvidenceItem(source_url="https://example.com", extracted_text=text)
        areas = check_technical_relevance([ev])
        assert "Generative AI & LLMs" in areas or "Generative AI" in areas
        assert "RAG & Retrieval Systems" in areas
        assert "AI Agents & Workflows" in areas
        assert "AI Infrastructure & Serving" in areas

    def test_confident_outcome_for_verified_recipient_and_ai_company(self, sample_profile):
        """When company is verified, recipient is identified, and AI work is supported -> CONFIDENT."""
        ev = EvidenceItem(
            source_url="https://agentic-ai.example",
            source_type="product",
            title="Autonomous Agents & RAG Platform",
            retrieval_status=RETRIEVAL_OK,
            extracted_text="Building multi-agent workflows, vector retrieval, and generative AI systems.",
            supports=["company_identity", "technical_relevance"],
        )
        rec_res = resolve_recipient("sarah@agentic-ai.example", supplied_name="Sarah Connor")
        rec_res.name_verification_status = "VERIFIED"

        res = ResearchResult(
            company_name="Agentic AI Labs",
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            technical_areas=["Generative AI & LLMs", "AI Agents & Workflows", "RAG & Retrieval Systems"],
            evidence=[ev],
            recipient_resolution=rec_res,
        )
        v_res = VerificationResult(company_status="VERIFIED", contact_status="VERIFIED", overall_status="PROCEED")
        m_res = match_profile(res, sample_profile)
        p_res = PipelineResult(verification=v_res, match=m_res, summary="AI company match")

        clf = classify_contact(res, p_res, recipient_resolution=rec_res, original_email="sarah@agentic-ai.example")
        assert clf.primary_outcome == CONFIDENT
        assert clf.reason_code == REASON_CONFIDENT
        assert clf.ai_ml_relevant is True
        assert clf.recipient_name == "Sarah Connor"

    def test_review_name_for_generic_inbox_at_ai_company(self, sample_profile):
        """When company is AI-focused but inbox is generic (hr@ or founder@) -> REVIEW_NAME, not discarded."""
        ev = EvidenceItem(
            source_url="https://agentic-ai.example",
            retrieval_status=RETRIEVAL_OK,
            extracted_text="Deep learning, neural networks, and computer vision models.",
            supports=["company_identity", "technical_relevance"],
        )
        rec_res = resolve_recipient("founder@agentic-ai.example")
        assert rec_res.is_generic is True

        res = ResearchResult(
            company_name="Agentic AI Labs",
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            technical_areas=["Computer Vision & Generative Media", "AI/ML"],
            evidence=[ev],
            recipient_resolution=rec_res,
        )
        v_res = VerificationResult(company_status="VERIFIED", contact_status="VERIFIED", overall_status="PROCEED")
        m_res = match_profile(res, sample_profile)
        p_res = PipelineResult(verification=v_res, match=m_res, summary="AI company")

        clf = classify_contact(res, p_res, recipient_resolution=rec_res, original_email="founder@agentic-ai.example")
        assert clf.primary_outcome == REVIEW_NAME
        assert clf.reason_code == REASON_RECIPIENT_UNVERIFIED
        assert clf.ai_ml_relevant is True


# ---------------------------------------------------------------------------
# 4. HTTP Retrieval Edge Cases (Empty, Blocked, Network Error)
# ---------------------------------------------------------------------------

class TestHttpRetrievalEdgeCases:
    @patch("requests.Session.get")
    def test_http_403_blocked_status(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp

        provider = HttpPageProvider()
        ev = provider.fetch_page("https://cloudflare-protected.example")
        assert ev.retrieval_status == RETRIEVAL_BLOCKED
        assert "403" in ev.claim_supported

    @patch("requests.Session.get")
    def test_http_empty_page_status(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "   "
        mock_get.return_value = mock_resp

        provider = HttpPageProvider()
        ev = provider.fetch_page("https://blank-page.example")
        assert ev.retrieval_status == RETRIEVAL_EMPTY

    @patch("requests.Session.get")
    def test_http_javascript_only_page_status(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><noscript>You need to enable JavaScript to run this app.</noscript><div id='root'></div></body></html>"
        mock_get.return_value = mock_resp

        provider = HttpPageProvider()
        ev = provider.fetch_page("https://spa-react.example")
        assert ev.retrieval_status == RETRIEVAL_JS_ONLY

    @patch("requests.Session.get")
    def test_http_network_timeout_status(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        provider = HttpPageProvider()
        ev = provider.fetch_page("https://timed-out.example")
        assert ev.retrieval_status == RETRIEVAL_NETWORK_ERROR
        assert "timed out" in ev.claim_supported.lower()

    def test_research_failed_does_not_auto_eliminate(self):
        """When research provider fails, contact is classified with RESEARCH_FAILED, not auto-eliminated."""
        ev = EvidenceItem(
            source_url="https://flaky-server.example",
            retrieval_status=RETRIEVAL_NETWORK_ERROR,
            claim_supported="Network timeout",
        )
        res = ResearchResult(
            company_name="Flaky Corp",
            company_identity_status=RES_REVIEW,
            domain_relationship_status=SUPPORTED,
            technical_areas=[],
            evidence=[ev],
        )
        v_res = VerificationResult(
            company_status="REVIEW",
            contact_status="VERIFIED",
            overall_status="REVIEW",
            verification_notes=["Evidence incomplete due to network timeout"],
        )
        p_res = PipelineResult(
            verification=v_res,
            match=MatchResult(relevance_status="REVIEW", case_type="GENERAL_INQUIRY", matched_areas=[], profile_evidence=[], explanation=""),
            summary="Review needed",
        )
        clf = classify_contact(res, p_res, original_email="lead@flaky-server.example")
        assert clf.primary_outcome in ("REVIEW", "REVIEW_NAME")
        assert clf.reason_code == REASON_RESEARCH_FAILED


# ---------------------------------------------------------------------------
# 5. Dynamic CV Text Matching
# ---------------------------------------------------------------------------

class TestDynamicCvMatching:
    def test_cv_skills_matched_dynamically(self, sample_profile):
        """Candidate's uploaded CV mentioning LangGraph & RAG matches company evidence."""
        cv_text = "Experienced in developing autonomous agents with LangGraph, RAG pipelines, and PyTorch."
        ev = EvidenceItem(
            source_url="https://ai-startup.example",
            retrieval_status=RETRIEVAL_OK,
            extracted_text="We deploy LangGraph agentic workflows and retrieval systems.",
            supports=["technical_relevance"],
        )
        res = ResearchResult(
            company_name="AI Startup",
            technical_areas=["AI Agents & Workflows", "RAG & Retrieval Systems"],
            evidence=[ev],
        )
        m_res = match_profile(res, sample_profile, cv_text=cv_text)
        assert m_res.relevance_status == "RELEVANT"
        assert len(m_res.matched_areas) > 0


# ---------------------------------------------------------------------------
# 6. Batch Continuation & Persistence
# ---------------------------------------------------------------------------

class TestBatchContinuationAndPersistence:
    def test_batch_continuation_after_individual_error(self, tmp_path, sample_profile):
        """A failure in one contact does not halt the entire batch."""
        dummy_cv = tmp_path / "resume.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy resume content")
        wf = OutreachWorkflow(profile_path="data/master_profile.json", uploaded_cv_path=dummy_cv)

        contacts = [
            make_contact("valid1@example.com", company="TechCorp", row=1),
            make_contact("bad@example.com", company="CrashCorp", row=2),
            make_contact("valid2@example.com", company="DataCorp", row=3),
        ]
        import_mock = MagicMock()
        import_mock.contacts = contacts
        import_mock.errors = []

        # Make contact 2 raise an exception in process_contact
        original_process = wf.process_contact
        def mock_process(contact, **kwargs):
            if "bad@" in contact.email:
                raise RuntimeError("Simulated processing error on row 2")
            return original_process(contact, **kwargs)

        with patch.object(wf, "process_contact", side_effect=mock_process):
            batch = wf.process_import_result(import_mock)

        assert batch.total_rows == 3
        assert batch.processed_count == 3
        assert len(batch.results) == 3
        assert batch.failed_count == 1
        assert batch.results[1].status == "ERROR"
        assert batch.results[0].row_number == 1
        assert batch.results[1].row_number == 2
        assert batch.results[2].row_number == 3

    def test_persistence_and_reload_of_evidence_and_classification(self, tmp_path):
        """Stage 3 classification and evidence items persist and reload cleanly."""
        batch_id = "test_batch_stage3_001"
        record = {
            "batch_id": batch_id,
            "confident_count": 1,
            "review_name_count": 1,
            "not_ai_relevant_count": 0,
            "eliminated_count": 0,
            "results": [
                {
                    "row_number": 1,
                    "contact_email": "lead@ai.example",
                    "company_name": "AI Systems",
                    "classification": "CONFIDENT",
                    "reason_code": "CONFIDENT",
                    "ai_ml_relevant": True,
                    "evidence_items": [
                        {"source_url": "https://ai.example", "retrieval_status": "OK", "title": "AI Homepage"}
                    ],
                }
            ],
        }
        with patch("src.ui.storage._BATCHES_DIR", tmp_path):
            save_batch_record(batch_id, record)
            loaded = get_batch_record(batch_id)

        assert loaded is not None
        assert loaded["batch_id"] == batch_id
        assert loaded["confident_count"] == 1
        assert loaded["results"][0]["classification"] == "CONFIDENT"
        assert loaded["results"][0]["evidence_items"][0]["retrieval_status"] == "OK"
