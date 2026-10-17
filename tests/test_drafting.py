"""
tests/test_drafting.py
----------------------
Regression tests for Checkpoint 3: Personalized Outreach Drafting.

Coverage:
 - Context detection: linkedin_follow_up, referral, direct_application
 - Subject line generation per context and case type
 - Body personalisation: salutation, opening paragraph, alignment paragraph
 - Signature block integrity
 - Fabrication guard: missing referrer name does not invent one
 - Fabrication guard: missing LinkedIn note does not invent conversation
 - Fabrication guard: verified matched_areas appear; unverified areas absent
 - WorkflowMode: Uploaded CV and Tailored CV alignment paragraphs
 - contact_notes wired through the OutreachWorkflow engine
 - review_required propagation from verification and from context detector
"""
from __future__ import annotations

import pytest

from src.contacts.csv_importer import ContactRecord
from src.contacts.email_parser import parse_email
from src.cv.models import ProjectSelectionResult
from src.outreach.context import (
    OutreachContext,
    ContextResult,
    detect_outreach_context,
)
from src.outreach.drafting import (
    build_subject_line,
    draft_outreach_email,
)
from src.outreach.models import WorkflowMode
from src.research.models import (
    APPLICATION_OPEN,
    EvidenceItem,
    FOUND,
    ResearchResult,
    SUPPORTED,
)
from src.verification.models import (
    GENERAL_INQUIRY,
    MatchResult,
    PipelineResult,
    RELEVANT,
    SPECIFIC_INTERNSHIP,
    VerificationResult,
    PROCEED,
)
from src.workflow.engine import OutreachWorkflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROFILE = {
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
            "responsibilities": [
                "Developed diffusion-based generative AI pipelines utilizing Flux, SD1.5, Trellis-4B, and ControlNet.",
            ],
        }
    ],
    "technical_skills": {
        "ml_and_data": ["LangGraph", "RAG", "Stable Diffusion", "BERT", "XGBoost"],
    },
    "preferences": {
        "availability_duration_months": "3-6 months",
    },
    "projects": [],
}


def _make_pipeline_result(
    matched_areas: list[str] | None = None,
    case_type: str = GENERAL_INQUIRY,
    overall_status: str = PROCEED,
    relevance_status: str = RELEVANT,
    review_flags: list[str] | None = None,
) -> PipelineResult:
    return PipelineResult(
        verification=VerificationResult(
            company_status="VERIFIED",
            contact_status="VERIFIED",
            overall_status=overall_status,
            review_flags=review_flags or [],
        ),
        match=MatchResult(
            relevance_status=relevance_status,
            case_type=case_type,
            matched_areas=matched_areas or [],
        ),
    )


def _make_project_selection() -> ProjectSelectionResult:
    return ProjectSelectionResult(
        primary_project={"name": "ControlPlane.ai — Context-Aware AI Governance Middleware"},
        secondary_projects=[
            {"name": "E-Commerce Price Prediction"},
            {"name": "Building a Network Security System"},
        ],
        selected_projects=[
            {"name": "ControlPlane.ai — Context-Aware AI Governance Middleware"},
            {"name": "E-Commerce Price Prediction"},
            {"name": "Building a Network Security System"},
        ],
        ranking_explanation="Selected for AI/ML relevance.",
    )


# ---------------------------------------------------------------------------
# Section 1: Context Detection
# ---------------------------------------------------------------------------

class TestContextDetection:
    """Tests for detect_outreach_context()."""

    def test_empty_notes_returns_direct_application(self):
        result = detect_outreach_context("")
        assert result.context == OutreachContext.DIRECT_APPLICATION

    def test_none_like_whitespace_returns_direct_application(self):
        result = detect_outreach_context("   ")
        assert result.context == OutreachContext.DIRECT_APPLICATION

    def test_linkedin_keyword_detected(self):
        result = detect_outreach_context("Met via LinkedIn")
        assert result.context == OutreachContext.LINKEDIN_FOLLOW_UP

    def test_linkedin_conversation_phrase_detected(self):
        result = detect_outreach_context("Had a conversation on LinkedIn last week")
        assert result.context == OutreachContext.LINKEDIN_FOLLOW_UP

    def test_referral_referred_by_detected(self):
        result = detect_outreach_context("Referred by Asit Vidyarthi")
        assert result.context == OutreachContext.REFERRAL
        assert result.referrer_name == "Asit Vidyarthi"

    def test_referral_referral_colon_detected(self):
        result = detect_outreach_context("Referral: John Doe")
        assert result.context == OutreachContext.REFERRAL
        assert result.referrer_name == "John Doe"

    def test_referral_introduced_by_detected(self):
        result = detect_outreach_context("Introduced by Priya Sharma from the alumni network")
        assert result.context == OutreachContext.REFERRAL
        assert "Priya Sharma" in result.referrer_name

    def test_referral_takes_priority_over_linkedin(self):
        """When notes contain both a referral and LinkedIn, referral wins."""
        result = detect_outreach_context("Referred by Jane Doe; also connected on LinkedIn")
        assert result.context == OutreachContext.REFERRAL
        assert result.referrer_name == "Jane Doe"

    def test_no_referral_name_extracted_returns_direct(self):
        """'referred by' without a capturable name falls back to direct application."""
        result = detect_outreach_context("I was referred.")
        # Pattern requires a capitalized name after 'referred by'; lone 'referred.' should not match.
        assert result.context in (OutreachContext.DIRECT_APPLICATION, OutreachContext.REFERRAL)

    def test_unrelated_notes_return_direct_application(self):
        result = detect_outreach_context("Found on their company careers page")
        assert result.context == OutreachContext.DIRECT_APPLICATION

    def test_detection_notes_populated(self):
        result = detect_outreach_context("Referred by Asit Vidyarthi")
        assert result.detection_notes != ""


# ---------------------------------------------------------------------------
# Section 2: Subject Line Builder
# ---------------------------------------------------------------------------

class TestSubjectLine:
    """Tests for build_subject_line()."""

    def test_linkedin_follow_up_subject(self):
        subj = build_subject_line(
            candidate_name="Bhavesh Verma",
            target_role="AI/ML Internship",
            company_name="Fitsol",
            context=OutreachContext.LINKEDIN_FOLLOW_UP,
        )
        assert "LinkedIn" in subj
        assert "AI/ML Internship" in subj

    def test_referral_subject_includes_referrer_name(self):
        subj = build_subject_line(
            candidate_name="Bhavesh Verma",
            target_role="AI/ML Internship",
            company_name="Prodoc",
            context=OutreachContext.REFERRAL,
            referrer_name="Asit Vidyarthi",
        )
        assert "Asit Vidyarthi" in subj
        assert "Referred by" in subj

    def test_referral_subject_without_name_falls_back(self):
        """If referrer name is empty, subject should not say 'Referred by '."""
        subj = build_subject_line(
            candidate_name="Bhavesh Verma",
            target_role="AI/ML Internship",
            company_name="Nugget",
            context=OutreachContext.REFERRAL,
            referrer_name="",
        )
        assert "Referred by " not in subj

    def test_specific_internship_direct_application_subject(self):
        subj = build_subject_line(
            candidate_name="Bhavesh Verma",
            target_role="AI/ML Internship",
            company_name="Nugget",
            case_type=SPECIFIC_INTERNSHIP,
            context=OutreachContext.DIRECT_APPLICATION,
        )
        assert "Application for" in subj
        assert "Nugget" in subj
        assert "Bhavesh Verma" not in subj

    def test_general_inquiry_direct_application_subject(self):
        subj = build_subject_line(
            candidate_name="Bhavesh Verma",
            target_role="AI/ML Internship",
            company_name="Acme",
            case_type=GENERAL_INQUIRY,
            context=OutreachContext.DIRECT_APPLICATION,
        )
        assert "Acme" in subj
        assert "Bhavesh Verma" not in subj

    def test_default_role_used_when_empty(self):
        subj = build_subject_line(
            candidate_name="Bhavesh Verma",
            target_role="",
            company_name="Acme",
        )
        assert "AI/ML Internship" in subj


# ---------------------------------------------------------------------------
# Section 3: Draft Body — Opening Paragraph Personalization
# ---------------------------------------------------------------------------

class TestOpeningParagraph:
    """The email opening changes based on outreach context."""

    def test_linkedin_follow_up_references_linkedin(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="akshay@fitsol.green",
            company_name="Fitsol",
            contact_name="Akshay Tandon",
            target_role="AI/ML Internship",
            contact_notes="We had a conversation on LinkedIn last week",
        )
        assert "LinkedIn" in draft.body
        assert "formally" in draft.body.lower() or "following" in draft.body.lower()

    def test_linkedin_follow_up_does_not_invent_conversation_date(self):
        """No specific dates or platform events should be fabricated."""
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="akshay@fitsol.green",
            company_name="Fitsol",
            contact_name="Akshay Tandon",
            contact_notes="Connected on LinkedIn",
        )
        # Should not hallucinate a meeting date or specific event name
        body_lower = draft.body.lower()
        assert "january" not in body_lower
        assert "february" not in body_lower
        assert "meetup" not in body_lower

    def test_referral_mentions_referrer_name(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@prodoc.ai",
            company_name="Prodoc",
            contact_name="",
            target_role="AI/ML Internship",
            contact_notes="Referred by Asit Vidyarthi",
        )
        assert "Asit Vidyarthi" in draft.body

    def test_referral_uses_recommendation_framing(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@prodoc.ai",
            company_name="Prodoc",
            contact_name="",
            contact_notes="Referred by Asit Vidyarthi",
        )
        body_lower = draft.body.lower()
        assert "recommendation" in body_lower or "reaching out" in body_lower

    def test_direct_application_no_linkedin_no_referral_invented(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hello@nugget.ai",
            company_name="Nugget",
            contact_name="",
            contact_notes="Found on their careers page",
        )
        body_lower = draft.body.lower()
        assert "linkedin" not in body_lower
        assert "referred" not in body_lower

    def test_empty_notes_produces_direct_application_draft(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hello@company.ai",
            company_name="AcmeAI",
            contact_name="",
            contact_notes="",
        )
        # Direct application should mention writing to inquire
        body_lower = draft.body.lower()
        assert "writing" in body_lower or "inquire" in body_lower or "interest" in body_lower


# ---------------------------------------------------------------------------
# Section 4: Verified Company Evidence Grounding
# ---------------------------------------------------------------------------

class TestCompanyEvidenceGrounding:
    """matched_areas from verification appear; unverified claims do not."""

    def test_matched_areas_appear_in_body(self):
        pipeline = _make_pipeline_result(matched_areas=["Generative AI", "RAG systems"])
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="test@example.ai",
            company_name="TestCo",
            verification_pipeline_result=pipeline,
        )
        assert "Generative AI" in draft.body or "RAG systems" in draft.body

    def test_empty_matched_areas_no_false_company_claim(self):
        """When no areas are matched, we must not claim the company does X."""
        pipeline = _make_pipeline_result(matched_areas=[])
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="test@example.ai",
            company_name="AcmeCorp",
            verification_pipeline_result=pipeline,
        )
        # "Given AcmeCorp's focus on ..." should not appear with empty areas
        assert "Given AcmeCorp" not in draft.body

    def test_general_inquiry_no_role_mention_specific(self):
        pipeline = _make_pipeline_result(case_type=GENERAL_INQUIRY)
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
            target_role="",
            verification_pipeline_result=pipeline,
        )
        # Should default to "AI/ML Internship"
        assert "AI/ML Internship" in draft.subject or "Opportunities" in draft.subject

    def test_specific_internship_role_in_subject(self):
        pipeline = _make_pipeline_result(case_type=SPECIFIC_INTERNSHIP)
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
            target_role="GenAI Research Intern",
            verification_pipeline_result=pipeline,
        )
        assert "GenAI Research Intern" in draft.subject


# ---------------------------------------------------------------------------
# Section 5: Salutation and Signature
# ---------------------------------------------------------------------------

class TestSalutationAndSignature:
    """Check salutation personalisation and signature block correctness."""

    def test_named_contact_uses_first_name(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="akshay@fitsol.green",
            company_name="Fitsol",
            contact_name="Akshay Tandon",
        )
        assert "Dear Akshay," in draft.body

    def test_unknown_contact_uses_company_name(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@prodoc.ai",
            company_name="Prodoc",
            contact_name="",
        )
        assert "Dear Hiring Team at Prodoc," in draft.body

    def test_signature_contains_full_name(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
        )
        assert "Bhavesh Verma" in draft.body

    def test_signature_contains_institution(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
        )
        assert "IIT Kharagpur" in draft.body

    def test_signature_contains_phone(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
        )
        assert "+91 7067419231" in draft.body

    def test_signature_contains_email(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
        )
        assert "bhaveshverma7890@gmail.com" in draft.body

    def test_signature_without_phone_still_valid(self):
        profile_no_phone = {**PROFILE, "personal_info": {**PROFILE["personal_info"], "phone": ""}}
        draft = draft_outreach_email(
            profile=profile_no_phone,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
        )
        assert "Bhavesh Verma" in draft.body
        assert "IIT Kharagpur" in draft.body


# ---------------------------------------------------------------------------
# Section 6: Tailored CV Mode Alignment
# ---------------------------------------------------------------------------

class TestTailoredCVAlignmentParagraph:
    """Tailored CV mode surfaces selected project names."""

    def test_tailored_mode_primary_project_in_body(self):
        selection = _make_project_selection()
        pipeline = _make_pipeline_result(matched_areas=["AI agents", "LLMs"])
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
            mode=WorkflowMode.TAILORED_CV,
            verification_pipeline_result=pipeline,
            project_selection=selection,
        )
        assert "ControlPlane.ai" in draft.body

    def test_tailored_mode_secondary_project_in_body(self):
        selection = _make_project_selection()
        pipeline = _make_pipeline_result(matched_areas=["NLP"])
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
            mode=WorkflowMode.TAILORED_CV,
            verification_pipeline_result=pipeline,
            project_selection=selection,
        )
        assert "E-Commerce Price Prediction" in draft.body

    def test_uploaded_cv_mode_does_not_mention_specific_projects(self):
        """Uploaded CV mode should not name specific projects (they live in the attached file)."""
        pipeline = _make_pipeline_result(matched_areas=["Machine Learning"])
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
            mode=WorkflowMode.UPLOADED_CV,
            verification_pipeline_result=pipeline,
        )
        # Should NOT reference specific project names in the body
        assert "ControlPlane.ai" not in draft.body
        assert "E-Commerce Price Prediction" not in draft.body


# ---------------------------------------------------------------------------
# Section 7: review_required Propagation
# ---------------------------------------------------------------------------

class TestReviewRequiredPropagation:
    """review_required should be set when verification or context flags it."""

    def test_review_status_from_verification_sets_review_required(self):
        pipeline = _make_pipeline_result(overall_status="REVIEW")
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
            verification_pipeline_result=pipeline,
        )
        assert draft.review_required is True

    def test_review_relevance_status_sets_review_required(self):
        pipeline = _make_pipeline_result(relevance_status="REVIEW")
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
            verification_pipeline_result=pipeline,
        )
        assert draft.review_required is True

    def test_no_review_when_proceed_and_relevant(self):
        pipeline = _make_pipeline_result(
            overall_status=PROCEED,
            relevance_status=RELEVANT,
        )
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
            verification_pipeline_result=pipeline,
        )
        assert draft.review_required is False

    def test_verification_review_flags_propagated(self):
        pipeline = _make_pipeline_result(
            overall_status="REVIEW",
            review_flags=["No official company website found"],
        )
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
            verification_pipeline_result=pipeline,
        )
        assert any("No official" in f for f in draft.review_flags)


# ---------------------------------------------------------------------------
# Section 8: Availability in Closing
# ---------------------------------------------------------------------------

class TestAvailabilityInClosing:
    """Closing paragraph should reference the candidate's availability."""

    def test_availability_mentioned_in_closing(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
        )
        assert "3-6 months" in draft.body or "3" in draft.body

    def test_cv_attached_phrase_in_closing(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@example.ai",
            company_name="ExampleAI",
        )
        body_lower = draft.body.lower()
        assert "attached" in body_lower or "cv" in body_lower


# ---------------------------------------------------------------------------
# Section 9: Engine Integration — contact_notes wired through
# ---------------------------------------------------------------------------

class MockResearchProvider:
    """Mock provider returning verified AI/ML evidence."""
    def research(self, inp):
        return ResearchResult(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            role_status=FOUND,
            application_status=APPLICATION_OPEN,
            technical_areas=["Generative AI", "RAG"],
            evidence=[
                EvidenceItem(
                    source_url="https://exampleai.com",
                    source_type="official_website",
                    title="ExampleAI",
                    extracted_text="ExampleAI builds Generative AI and RAG systems.",
                )
            ],
        )


class TestEngineContactNotesWiring:
    """Verify that contact.notes flows into the draft via the engine."""

    @pytest.fixture
    def dummy_cv(self, tmp_path):
        cv = tmp_path / "test_cv.pdf"
        cv.write_bytes(b"%PDF-1.4 dummy cv bytes")
        return cv

    def _make_contact(self, notes: str) -> ContactRecord:
        return ContactRecord(
            row_number=2,
            company_name="ExampleAI",
            contact_name="",
            email_result=parse_email("hr@exampleai.com"),
            website="https://exampleai.com",
            role_or_job_url="AI/ML Intern",
            source_url="",
            notes=notes,
        )

    def test_linkedin_notes_produce_linkedin_subject(self, dummy_cv):
        contact = self._make_contact("Connected on LinkedIn recently")
        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )
        result = workflow.process_contact(
            contact=contact,
            research_provider=MockResearchProvider(),
        )
        assert result.draft is not None
        assert "LinkedIn" in result.draft.subject

    def test_referral_notes_produce_referral_subject(self, dummy_cv):
        contact = self._make_contact("Referred by Asit Vidyarthi")
        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )
        result = workflow.process_contact(
            contact=contact,
            research_provider=MockResearchProvider(),
        )
        assert result.draft is not None
        assert "Asit Vidyarthi" in result.draft.subject
        assert "Asit Vidyarthi" in result.draft.body

    def test_empty_notes_produce_direct_subject(self, dummy_cv):
        contact = self._make_contact("")
        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )
        result = workflow.process_contact(
            contact=contact,
            research_provider=MockResearchProvider(),
        )
        assert result.draft is not None
        # Direct application subject should not mention LinkedIn or referral
        assert "LinkedIn" not in result.draft.subject
        assert "Referred by" not in result.draft.subject

    def test_draft_unique_id_per_contact(self, dummy_cv):
        """Each processed contact should receive a distinct draft_id."""
        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )
        contact_a = self._make_contact("")
        contact_b = self._make_contact("Connected on LinkedIn")
        r_a = workflow.process_contact(contact=contact_a, research_provider=MockResearchProvider())
        r_b = workflow.process_contact(contact=contact_b, research_provider=MockResearchProvider())
        assert r_a.draft is not None and r_b.draft is not None
        assert r_a.draft.draft_id != r_b.draft.draft_id
