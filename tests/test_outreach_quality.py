"""
tests/test_outreach_quality.py
------------------------------
Regression tests for Prompt 3: Outreach context detection, email draft quality,
evidence grounding, grammatical accuracy, and transparent review flags.

Covers:
- All context test cases (LinkedIn conversation vs discovery, referral parsing, negative referrals, ambiguous notes).
- Grounding: company statements strictly match verified evidence.
- Style: subjects include company and role without appending candidate full name.
- Grammar: 'a 3–6 month internship' or 'a six-month internship' (never 'months').
- Signature: 'Best regards,' with IIT Kharagpur affiliation.
- Transparent review flags: missing verification, missing company name, unverified AI, ambiguous context.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.contacts.csv_importer import ContactRecord
from src.contacts.email_parser import parse_email
from src.outreach.context import (
    OutreachContext,
    ContextResult,
    detect_outreach_context,
)
from src.outreach.drafting import (
    build_subject_line,
    draft_outreach_email,
    _format_internship_duration,
)
from src.outreach.models import DraftStatus, EmailDraft, WorkflowMode
from src.research.models import (
    EvidenceItem,
    ResearchInput,
    ResearchResult,
    RETRIEVAL_OK,
    SUPPORTED,
)
from src.verification.models import (
    GENERAL_INQUIRY,
    MatchResult,
    PipelineResult,
    RELEVANT,
    NOT_A_MATCH,
    SPECIFIC_INTERNSHIP,
    VerificationResult,
    PROCEED,
)
from src.workflow.engine import OutreachWorkflow


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
            "degree": "Dual Degree in Engineering",
        }
    ],
    "experience": [
        {
            "organization": "Red Panda Games",
            "responsibilities": [
                "Developed diffusion-based generative AI pipelines utilizing Stable Diffusion and ControlNet.",
            ],
        }
    ],
    "technical_skills": {
        "ml_and_data": ["LangGraph", "RAG", "Stable Diffusion", "BERT", "XGBoost"],
    },
    "preferences": {
        "availability_duration_months": "3–6 months",
        "target_roles": ["AI/ML Intern", "Generative AI Intern"],
    },
    "projects": [
        {"name": "ControlPlane.ai — Context-Aware AI Governance Middleware"},
    ],
}


# ---------------------------------------------------------------------------
# 1. Context Detection Granular Tests
# ---------------------------------------------------------------------------

class TestContextDetectionGranular:
    def test_we_discussed_on_linkedin(self):
        res = detect_outreach_context("We discussed the internship on LinkedIn last week")
        assert res.context == OutreachContext.LINKEDIN_FOLLOW_UP
        assert res.review_required is False

    def test_found_company_on_linkedin_is_not_conversation(self):
        res = detect_outreach_context("Found this company on LinkedIn")
        assert res.context == OutreachContext.DIRECT_APPLICATION
        assert res.review_required is False

    def test_linkedin_job_post_is_not_conversation(self):
        res = detect_outreach_context("Applied through LinkedIn job post")
        assert res.context == OutreachContext.DIRECT_APPLICATION

    def test_found_founder_profile_is_not_conversation(self):
        res = detect_outreach_context("I found the founder's profile on LinkedIn")
        assert res.context == OutreachContext.DIRECT_APPLICATION

    def test_referral_with_clause_extracts_only_person_name(self):
        res = detect_outreach_context("Referred by Asit Vidyarthi who is the Co-Founder & CEO")
        assert res.context == OutreachContext.REFERRAL
        assert res.referrer_name == "Asit Vidyarthi"
        assert "Co-Founder" not in res.referrer_name
        assert "CEO" not in res.referrer_name

    def test_not_referred_by_anyone(self):
        res = detect_outreach_context("Not referred by anyone")
        assert res.context == OutreachContext.DIRECT_APPLICATION
        assert res.referrer_name == ""

    def test_no_referral(self):
        res = detect_outreach_context("No referral; direct cold outreach")
        assert res.context == OutreachContext.DIRECT_APPLICATION
        assert res.referrer_name == ""

    def test_ambiguous_notes_require_human_review(self):
        res = detect_outreach_context("Notes: LinkedIn connection or referral?")
        assert res.context == OutreachContext.DIRECT_APPLICATION
        assert res.review_required is True
        assert "ambiguous" in res.detection_notes.lower() or "linkedin" in res.detection_notes.lower()


# ---------------------------------------------------------------------------
# 2. Subject Line & Style Tests
# ---------------------------------------------------------------------------

class TestSubjectLineQuality:
    def test_direct_application_includes_company_not_candidate_name(self):
        subj = build_subject_line(
            candidate_name="Bhavesh Verma",
            target_role="AI/ML Internship",
            company_name="Fitsol",
            case_type=GENERAL_INQUIRY,
            context=OutreachContext.DIRECT_APPLICATION,
        )
        assert subj == "AI/ML Internship Application — Fitsol"
        assert "Bhavesh Verma" not in subj

    def test_direct_application_missing_company_does_not_invent(self):
        subj = build_subject_line(
            candidate_name="Bhavesh Verma",
            target_role="AI/ML Internship",
            company_name="",
            case_type=GENERAL_INQUIRY,
            context=OutreachContext.DIRECT_APPLICATION,
        )
        assert subj == "Inquiry regarding AI/ML Internship Opportunities"
        assert "Unknown Company" not in subj
        assert "Bhavesh Verma" not in subj

    def test_referral_subject_format(self):
        subj = build_subject_line(
            candidate_name="Bhavesh Verma",
            target_role="AI/ML Internship",
            company_name="Prodoc",
            context=OutreachContext.REFERRAL,
            referrer_name="Asit Vidyarthi",
        )
        assert subj == "AI/ML Internship Application — Referred by Asit Vidyarthi"

    def test_linkedin_subject_format(self):
        subj = build_subject_line(
            candidate_name="Bhavesh Verma",
            target_role="AI/ML Internship",
            company_name="Fitsol",
            context=OutreachContext.LINKEDIN_FOLLOW_UP,
        )
        assert subj == "AI/ML Internship Opportunity — Following Our LinkedIn Conversation"


# ---------------------------------------------------------------------------
# 3. Grammar, Signature, and Body Grounding Tests
# ---------------------------------------------------------------------------

class TestDraftBodyQualityAndGrounding:
    def test_availability_grammar_no_months_plural(self):
        assert _format_internship_duration("3–6 months") == "a 3–6 month internship"
        assert _format_internship_duration("3-6 months") == "a 3–6 month internship"
        assert _format_internship_duration("6 months") == "a six-month internship"

    def test_signature_uses_best_regards_and_iit_kharagpur(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="test@ai.example",
            company_name="AI Labs",
        )
        assert "Best regards," in draft.body
        assert "Bhavesh Verma" in draft.body
        assert "IIT Kharagpur" in draft.body
        assert "Sincerely," not in draft.body

    def test_verified_ai_company_alignment_grounded(self):
        pipeline = PipelineResult(
            verification=VerificationResult(
                company_status="VERIFIED",
                contact_status="VERIFIED",
                overall_status=PROCEED,
                evidence_sources=["https://fitsol.green"],
            ),
            match=MatchResult(
                relevance_status=RELEVANT,
                case_type=SPECIFIC_INTERNSHIP,
                matched_areas=["Generative AI", "AI/ML"],
            ),
        )
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="akshay@fitsol.green",
            company_name="Fitsol",
            target_role="AI/ML Internship",
            verification_pipeline_result=pipeline,
            contact_notes="We discussed the internship on LinkedIn last week",
        )
        # Verified areas appear in body
        assert "Given Fitsol's focus on Generative AI, AI/ML" in draft.body
        # LinkedIn conversation opening appears
        assert "Following our recent conversation on LinkedIn" in draft.body
        # Profile facts appear
        assert "Red Panda Games" in draft.body
        # Grammatically correct duration
        assert "a 3–6 month internship" in draft.body

    def test_non_ai_company_notes_mentioning_genai_never_claims_company_ai(self):
        """When company has no verified AI work, draft never claims company specializes in AI."""
        pipeline = PipelineResult(
            verification=VerificationResult(
                company_status="VERIFIED",
                contact_status="VERIFIED",
                overall_status="REVIEW",
                evidence_sources=["https://plumbing.example"],
            ),
            match=MatchResult(
                relevance_status=NOT_A_MATCH,
                case_type=GENERAL_INQUIRY,
                matched_areas=[],
            ),
        )
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="contact@plumbing.example",
            company_name="Plumbing Supplies Ltd",
            verification_pipeline_result=pipeline,
            contact_notes="Candidate desires GenAI or LLM role",
        )
        assert "focus on Generative AI" not in draft.body
        assert "focus on AI/ML" not in draft.body
        assert "Given Plumbing Supplies Ltd's focus" not in draft.body
        assert draft.review_required is True

    def test_missing_verification_result_produces_review_flag(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="test@unknown.example",
            company_name="Unknown Corp",
            verification_pipeline_result=None,
        )
        assert draft.review_required is True
        assert any("Missing verification result" in f for f in draft.review_flags)

    def test_missing_company_name_produces_clean_neutral_draft(self):
        draft = draft_outreach_email(
            profile=PROFILE,
            recipient_email="recruiter@example.com",
            company_name="",
            verification_pipeline_result=None,
        )
        assert "Dear Hiring Team," in draft.body
        assert "at Unknown Company" not in draft.body
        assert draft.review_required is True
        assert any("Company name missing" in f for f in draft.review_flags)


# ---------------------------------------------------------------------------
# 4. Complete Representative Draft JSON Validation
# ---------------------------------------------------------------------------

class TestRepresentativeDraftInspection:
    def test_representative_drafts_inspectable_and_grounded(self):
        """Generates and inspects 3 distinct representative drafts: Verified AI, Unknown, Referral."""
        
        # 1. Verified AI company (Fitsol style)
        fitsol_pipeline = PipelineResult(
            verification=VerificationResult(
                company_status="VERIFIED",
                contact_status="VERIFIED",
                overall_status=PROCEED,
                evidence_sources=["https://fitsol.green"],
            ),
            match=MatchResult(
                relevance_status=RELEVANT,
                case_type=SPECIFIC_INTERNSHIP,
                matched_areas=["Generative AI", "AI/ML"],
            ),
        )
        draft_ai = draft_outreach_email(
            profile=PROFILE,
            recipient_email="akshay@fitsol.green",
            company_name="Fitsol",
            contact_name="Akshay Tandon",
            target_role="AI/ML Internship",
            verification_pipeline_result=fitsol_pipeline,
            contact_notes="We discussed the internship on LinkedIn last week",
        )
        assert draft_ai.subject == "AI/ML Internship Opportunity — Following Our LinkedIn Conversation"
        assert "Dear Akshay," in draft_ai.body
        assert "Given Fitsol's focus on Generative AI, AI/ML" in draft_ai.body
        assert "a 3–6 month internship" in draft_ai.body

        # 2. Unknown / unverified company
        draft_unknown = draft_outreach_email(
            profile=PROFILE,
            recipient_email="recruiter@ghost.example",
            company_name="",
            contact_name="",
            verification_pipeline_result=None,
        )
        assert draft_unknown.subject == "Inquiry regarding AI/ML Internship Opportunities"
        assert "Dear Hiring Team," in draft_unknown.body
        assert draft_unknown.review_required is True

        # 3. Confirmed Referral (Prodoc style)
        prodoc_pipeline = PipelineResult(
            verification=VerificationResult(
                company_status="VERIFIED",
                contact_status="VERIFIED",
                overall_status=PROCEED,
                evidence_sources=["https://prodoc.ai"],
            ),
            match=MatchResult(
                relevance_status=RELEVANT,
                case_type=SPECIFIC_INTERNSHIP,
                matched_areas=["AI/ML", "Agentic AI"],
            ),
        )
        draft_ref = draft_outreach_email(
            profile=PROFILE,
            recipient_email="hr@prodoc.ai",
            company_name="Prodoc",
            target_role="AI/ML Internship",
            verification_pipeline_result=prodoc_pipeline,
            contact_notes="Referred by Asit Vidyarthi who is the Co-Founder & CEO",
        )
        assert draft_ref.subject == "AI/ML Internship Application — Referred by Asit Vidyarthi"
        assert "on the recommendation of Asit Vidyarthi" in draft_ref.body
        assert "Given Prodoc's focus on AI/ML, Agentic AI" in draft_ref.body
        assert "Co-Founder" not in draft_ref.subject
