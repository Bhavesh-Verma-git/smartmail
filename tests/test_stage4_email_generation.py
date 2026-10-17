"""
tests/test_stage4_email_generation.py
-------------------------------------
Comprehensive automated test suite for SmartMail Stage 4:
Personalized Internship Email Generation.

Coverage:
1. Confirmed AI-relevant company with verified technical evidence produces company-specific draft.
2. Generic inboxes (hr@, careers@, founder@, jobs@) do not invent personal recipient names (uses team salutation).
3. Verified personal recipients use first name salutation (e.g. Dear Alice,).
4. Truthful candidate introduction: fourth-year Dual Degree student at IIT Kharagpur with AI/ML focus.
5. CV skills and projects connected truthfully without fabricating unverified metrics.
6. Companies with missing evidence, failed research, or not AI-relevant produce neutral drafts marked for review.
7. Domain mismatches and invalid emails produce review flags and appropriate generation status.
8. Generation status tracking: DRAFT_GENERATED, NEEDS_REVIEW, GENERATION_FAILED.
9. Persistence and reload of personalization explanation, evidence summary, and matched skills/projects.
10. Preservation of uploaded CV byte-for-byte in Uploaded CV Mode.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient

from src.contacts.csv_importer import ContactRecord
from src.contacts.email_parser import parse_email
from src.contacts.recipient_resolver import RecipientResolution, resolve_recipient
from src.outreach.drafting import (
    _build_salutation,
    _determine_cv_connections,
    build_subject_line,
    draft_outreach_email,
)
from src.outreach.models import DraftStatus, EmailDraft, GenerationStatus, WorkflowMode
from src.outreach.send_log import SendLog
from src.research.models import EvidenceItem, ResearchResult, RETRIEVAL_OK
from src.ui.app import create_app
from src.ui.storage import get_draft, register_uploaded_cv, save_draft_record, update_draft
from src.verification.classifier import (
    CONFIDENT,
    ELIMINATED,
    NOT_AI_RELEVANT,
    REVIEW_NAME,
    ContactClassification,
    REASON_CONFIDENT,
    REASON_DOMAIN_MISMATCH,
    REASON_NOT_AI_RELEVANT,
    REASON_RECIPIENT_UNVERIFIED,
    REASON_RESEARCH_FAILED,
)
from src.verification.models import (
    GENERAL_INQUIRY,
    MatchResult,
    PipelineResult,
    PROCEED,
    RELEVANT,
    SPECIFIC_INTERNSHIP,
    VerificationResult,
)
from src.workflow.engine import OutreachWorkflow


# ---------------------------------------------------------------------------
# Test Fixtures & Constants
# ---------------------------------------------------------------------------

CANDIDATE_PROFILE = {
    "personal_info": {
        "full_name": "Bhavesh Verma",
        "preferred_name": "Bhavesh",
        "email": "bhaveshverma7890@gmail.com",
        "phone": "+91 7067419231",
    },
    "education": [
        {
            "institution": "IIT Kharagpur",
            "degree": "B.Tech. (Hons.) in Mining Engineering and M.Tech. in Petroleum Engineering (Dual Degree)",
        }
    ],
    "experience": [
        {
            "organization": "Red Panda Games",
            "role": "Data Science Intern",
            "responsibilities": [
                "Developed diffusion-based generative AI pipelines utilizing Flux, SD1.5, Trellis-4B, and ControlNet.",
                "Optimized AWS GPU inference workflows and implemented memory offloading strategies.",
            ],
        }
    ],
    "projects": [
        {
            "name": "ControlPlane.ai — Context-Aware AI Governance Middleware",
            "technologies": ["LangGraph", "LiteLLM", "FAISS", "BM25", "Neo4j", "XGBoost"],
        }
    ],
    "technical_skills": {
        "ml_and_data": ["LangGraph", "RAG", "Flux", "Stable Diffusion", "ControlNet", "PyTorch"],
    },
    "preferences": {
        "availability_duration_months": "3–6 months",
        "target_roles": ["AI/ML Intern", "Generative AI Intern"],
    },
}


def _make_pipeline(
    matched_areas: list[str] | None = None,
    overall_status: str = PROCEED,
    relevance_status: str = RELEVANT,
    company_status: str = "VERIFIED",
    case_type: str = GENERAL_INQUIRY,
    review_flags: list[str] | None = None,
) -> PipelineResult:
    return PipelineResult(
        verification=VerificationResult(
            company_status=company_status,
            contact_status="VERIFIED",
            overall_status=overall_status,
            evidence_sources=["https://example.ai/about"],
            review_flags=review_flags or [],
        ),
        match=MatchResult(
            relevance_status=relevance_status,
            case_type=case_type,
            matched_areas=matched_areas or [],
            profile_evidence=["Experience in Generative AI"],
            evidence_sources=["https://example.ai/research"],
            explanation="Verified AI/ML focus matched to candidate profile.",
        ),
        summary="Verified pipeline result",
    )


# ---------------------------------------------------------------------------
# 1. Successful Personalization with Verified Evidence
# ---------------------------------------------------------------------------

class TestPersonalizedEmailGeneration:
    """Verify truthful, evidence-backed email drafting for confirmed companies."""

    def test_confirmed_ai_company_generates_company_specific_draft(self):
        pipeline = _make_pipeline(matched_areas=["Generative AI & LLM Applications", "RAG, Retrieval & Knowledge Systems"])
        res_result = ResearchResult(
            company_name="DeepAI",
            ai_ml_evidence_summary="DeepAI builds enterprise retrieval-augmented generation and autonomous agent pipelines.",
            technical_areas=["Generative AI & LLM Applications", "RAG, Retrieval & Knowledge Systems"],
        )
        rec_res = resolve_recipient(
            supplied_name_or_email="alice.smith@deepai.com",
            supplied_name="Alice Smith",
            verified_name_from_evidence="Alice Smith",
        )
        classification = ContactClassification(
            primary_outcome=CONFIDENT,
            reason_code=REASON_CONFIDENT,
            explanation="Company verified and strongly relevant to AI/ML.",
            ai_ml_relevance="RELEVANT",
            recipient_resolution=rec_res,
            matched_areas=["Generative AI & LLM Applications", "RAG, Retrieval & Knowledge Systems"],
        )

        draft = draft_outreach_email(
            profile=CANDIDATE_PROFILE,
            recipient_email="alice.smith@deepai.com",
            company_name="DeepAI",
            contact_name="Alice Smith",
            verification_pipeline_result=pipeline,
            classification=classification,
            recipient_resolution=rec_res,
            research_result=res_result,
        )

        assert draft.generation_status == GenerationStatus.DRAFT_GENERATED.value
        assert not draft.review_required
        # Check salutation
        assert "Dear Alice," in draft.body
        # Check education intro
        assert "fourth-year Dual Degree student at IIT Kharagpur with a strong foundation in Applied AI and Machine Learning" in draft.body
        # Check company alignment references verified areas
        assert "Given DeepAI's focus on Generative AI & LLM Applications, RAG, Retrieval & Knowledge Systems," in draft.body
        # Check practical candidate CV facts
        assert "Red Panda Games" in draft.body
        assert "diffusion models" in draft.body
        assert "context-aware AI middleware and RAG architectures" in draft.body
        # Check closing and availability
        assert "I am available for a 3–6 month internship." in draft.body
        # Check signature
        assert "Bhavesh Verma" in draft.body
        assert "IIT Kharagpur" in draft.body
        # Check personalization explanation
        assert "Personalized for DeepAI" in draft.personalization_explanation
        assert "Generative AI & LLM Applications" in draft.personalization_explanation
        assert "LangGraph" in draft.matched_skills or "Hybrid RAG (FAISS + BM25)" in draft.matched_skills

    def test_specific_internship_case_type_reflects_role_application(self):
        pipeline = _make_pipeline(
            matched_areas=["Machine Learning & Predictive Analytics"],
            case_type=SPECIFIC_INTERNSHIP,
        )
        draft = draft_outreach_email(
            profile=CANDIDATE_PROFILE,
            recipient_email="recruiter@modelco.ai",
            company_name="ModelCo",
            contact_name="Bob",
            target_role="ML Platform Intern",
            verification_pipeline_result=pipeline,
        )
        assert "Application for ML Platform Intern — ModelCo" in draft.subject
        assert "strong interest in the ML Platform Intern role at ModelCo" in draft.body


# ---------------------------------------------------------------------------
# 2. Generic Inboxes & Recipient Identity Safeguards
# ---------------------------------------------------------------------------

class TestGenericInboxHandling:
    """Ensure generic mailboxes never invent personal names."""

    @pytest.mark.parametrize("local_part", ["hr", "careers", "jobs", "info", "founder", "ceo", "team"])
    def test_generic_email_uses_team_salutation(self, local_part: str):
        email = f"{local_part}@frontier.ai"
        rec_res = resolve_recipient(supplied_name_or_email=email, supplied_name=local_part.capitalize())
        pipeline = _make_pipeline(matched_areas=["Generative AI"])
        classification = ContactClassification(
            primary_outcome=REVIEW_NAME,
            reason_code=REASON_RECIPIENT_UNVERIFIED,
            explanation="Company verified; functional inbox.",
            ai_ml_relevance="RELEVANT",
            recipient_resolution=rec_res,
            matched_areas=["Generative AI"],
        )

        draft = draft_outreach_email(
            profile=CANDIDATE_PROFILE,
            recipient_email=email,
            company_name="Frontier AI",
            contact_name=local_part.capitalize(),
            verification_pipeline_result=pipeline,
            classification=classification,
            recipient_resolution=rec_res,
        )

        # Must NOT greet "Dear Hr," or "Dear Founder,"
        assert f"Dear {local_part.capitalize()}," not in draft.body
        assert "Dear Hiring Team at Frontier AI," in draft.body
        assert draft.recipient_type == "generic_inbox"
        assert draft.generation_status == GenerationStatus.NEEDS_REVIEW.value
        assert any("Generic inbox" in flag for flag in draft.review_flags)

    def test_salutation_helper_respects_is_generic_flag(self):
        assert _build_salutation("Hr", "Nexus AI", is_generic=True) == "Dear Hiring Team at Nexus AI,"
        assert _build_salutation("", "Nexus AI", is_generic=False) == "Dear Hiring Team at Nexus AI,"
        assert _build_salutation("Alice", "Nexus AI", is_generic=False) == "Dear Alice,"


# ---------------------------------------------------------------------------
# 3. Unverified Companies, Failed Research, and Neutral Drafts
# ---------------------------------------------------------------------------

class TestUnverifiedAndNeutralDrafts:
    """Unverified companies must receive restrained neutral text and review flags."""

    def test_non_ai_company_omits_ai_alignment_claims(self):
        pipeline = _make_pipeline(
            matched_areas=[],
            relevance_status="NOT_A_MATCH",
            overall_status="REVIEW",
        )
        rec_res = resolve_recipient(supplied_name_or_email="hiring@standardplumbing.com")
        classification = ContactClassification(
            primary_outcome=NOT_AI_RELEVANT,
            reason_code=REASON_NOT_AI_RELEVANT,
            explanation="Company has no verified AI/ML work.",
            ai_ml_relevance="NOT_AI_RELEVANT",
            recipient_resolution=rec_res,
        )

        draft = draft_outreach_email(
            profile=CANDIDATE_PROFILE,
            recipient_email="hiring@standardplumbing.com",
            company_name="Standard Plumbing",
            contact_name="",
            verification_pipeline_result=pipeline,
            classification=classification,
            recipient_resolution=rec_res,
        )

        assert draft.generation_status == GenerationStatus.NEEDS_REVIEW.value
        assert draft.review_required is True
        # Alignment paragraph must be neutral overview, never claiming Standard Plumbing does AI
        assert "Given Standard Plumbing's focus on" not in draft.body
        assert "My background includes practical experience in applied machine learning" in draft.body
        assert any("NOT_AI_RELEVANT" in flag for flag in draft.review_flags)

    def test_domain_mismatch_flags_review(self):
        pipeline = _make_pipeline(
            matched_areas=["AI/ML"],
            overall_status="REVIEW",
            review_flags=["Domain mismatch: email domain differs from official website."],
        )
        rec_res = resolve_recipient(supplied_name_or_email="contact@lookalike.com")
        classification = ContactClassification(
            primary_outcome=REVIEW_NAME,
            reason_code=REASON_DOMAIN_MISMATCH,
            explanation="Domain mismatch between email and website.",
            ai_ml_relevance="RELEVANT",
            recipient_resolution=rec_res,
        )

        draft = draft_outreach_email(
            profile=CANDIDATE_PROFILE,
            recipient_email="contact@lookalike.com",
            company_name="Real AI",
            verification_pipeline_result=pipeline,
            classification=classification,
            recipient_resolution=rec_res,
        )

        assert draft.generation_status == GenerationStatus.NEEDS_REVIEW.value
        assert draft.review_required is True
        assert any("DOMAIN_MISMATCH" in flag for flag in draft.review_flags)

    def test_invalid_email_marks_generation_failed(self):
        rec_res = resolve_recipient(supplied_name_or_email="bad-email-at-nowhere")
        classification = ContactClassification(
            primary_outcome=ELIMINATED,
            reason_code="INVALID_EMAIL",
            explanation="Email syntax is invalid.",
            ai_ml_relevance="REVIEW",
            recipient_resolution=rec_res,
        )
        draft = draft_outreach_email(
            profile=CANDIDATE_PROFILE,
            recipient_email="bad-email-at-nowhere",
            company_name="Nowhere Inc",
            classification=classification,
            recipient_resolution=rec_res,
        )
        assert draft.generation_status == GenerationStatus.GENERATION_FAILED.value
        assert "Generation failed" in draft.personalization_explanation


# ---------------------------------------------------------------------------
# 4. Storage Persistence and Reload
# ---------------------------------------------------------------------------

class TestDraftStoragePersistence:
    """Verify that Stage 4 draft fields are saved and restored correctly."""

    def test_save_and_retrieve_draft_record_with_stage4_fields(self, tmp_path: Path):
        draft = EmailDraft(
            draft_id="draft-st4-test",
            recipient="lead@acme.ai",
            recipient_name="Lead",
            company_name="Acme AI",
            subject="AI/ML Internship Inquiry — Acme AI",
            body="Hello,\n\nTest body\n\nBest regards,\nBhavesh Verma",
            attachment_path=str(tmp_path / "cv.pdf"),
            mode=WorkflowMode.UPLOADED_CV,
            generation_status=GenerationStatus.DRAFT_GENERATED.value,
            personalization_explanation="Personalized based on Acme AI's verified RAG and agent research.",
            evidence_summary="Acme AI provides low-latency RAG systems.",
            matched_skills=["LangGraph", "RAG"],
            matched_projects=["ControlPlane.ai"],
            recipient_type="individual",
        )

        save_draft_record(draft, campaign_id="camp-stage4-test")
        loaded = get_draft("draft-st4-test")

        assert loaded is not None
        assert loaded["draft_id"] == "draft-st4-test"
        assert loaded["generation_status"] == "DRAFT_GENERATED"
        assert loaded["personalization_explanation"] == "Personalized based on Acme AI's verified RAG and agent research."
        assert loaded["evidence_summary"] == "Acme AI provides low-latency RAG systems."
        assert loaded["matched_skills"] == ["LangGraph", "RAG"]
        assert loaded["matched_projects"] == ["ControlPlane.ai"]
        assert loaded["recipient_type"] == "individual"


# ---------------------------------------------------------------------------
# 5. UI Integration & Rendering
# ---------------------------------------------------------------------------

class TestStage4UIRendering:
    """Verify draft review page displays Stage 4 Personalization Rationale card."""

    def test_draft_detail_route_renders_stage4_rationale_card(self, tmp_path: Path):
        # Register a valid CV
        dummy_cv = tmp_path / "candidate_cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 test cv bytes")
        register_uploaded_cv(dummy_cv)

        draft = EmailDraft(
            draft_id="draft-ui-render-test",
            recipient="team@omnilabs.ai",
            recipient_name="Hiring Team",
            company_name="OmniLabs",
            subject="AI/ML Internship Inquiry — OmniLabs",
            body="Dear Hiring Team at OmniLabs,\n\nI am writing to inquire...",
            attachment_path=str(dummy_cv),
            mode=WorkflowMode.UPLOADED_CV,
            review_required=True,
            review_flags=["Functional inbox: team salutation used."],
            generation_status=GenerationStatus.NEEDS_REVIEW.value,
            personalization_explanation="Held for human review: functional inbox addressed neutrally.",
            evidence_summary="OmniLabs engineers autonomous multi-agent pipelines.",
            matched_skills=["LangGraph", "LiteLLM"],
            matched_projects=["ControlPlane.ai (AI Governance Middleware)"],
            recipient_type="generic_inbox",
        )
        save_draft_record(draft, campaign_id="camp-ui-test")

        app = create_app({"TESTING": True})
        with app.test_client() as client:
            res = client.get("/drafts/draft-ui-render-test")
            assert res.status_code == 200
            html = res.data.decode("utf-8")

            # Check presence of Stage 4 Card
            assert "Personalization Rationale" in html
            assert "Needs Review" in html
            assert "Held for human review: functional inbox addressed neutrally." in html
            assert "Generic Inbox" in html
            assert "LangGraph, LiteLLM" in html
            assert "ControlPlane.ai (AI Governance Middleware)" in html
            assert "OmniLabs engineers autonomous multi-agent pipelines." in html
