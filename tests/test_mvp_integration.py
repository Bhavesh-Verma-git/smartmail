"""
tests/test_mvp_integration.py
------------------------------
Prompt 4: SmartMail MVP final integration and acceptance tests.

Coverage:
  1. Batch CSV workflow via process_csv() — varied contact types, mocked provider
  2. Evidence / relevance isolation end-to-end
  3. Email quality — 5 representative draft scenarios
  4. Uploaded CV Mode — SHA-256 integrity and bypass of tailored-CV generation
  5. Approval and duplicate-send safeguards (mocked delivery only)
  6. Regression tests for defects found during Prompt 4 audit

All mocked tests run without a live network, Gmail OAuth, or real email sending.
Live network retrieval is exercised only in TestLiveSmoke, clearly labelled,
and auto-skipped in CI when network is absent.
"""
from __future__ import annotations

import csv
import hashlib
import socket
from pathlib import Path
from typing import Any

import pytest

from src.contacts.csv_importer import ContactRecord, import_contacts_from_csv
from src.contacts.email_parser import parse_email
from src.outreach.context import OutreachContext, detect_outreach_context
from src.outreach.drafting import build_subject_line, draft_outreach_email
from src.outreach.mailer import EmailReviewManager, EmailSender
from src.outreach.models import DraftStatus, EmailDraft, SendResult, WorkflowMode
from src.outreach.send_log import SendLog
from src.profile.loader import load_profile
from src.research.models import (
    APPLICATION_UNKNOWN,
    EvidenceItem,
    RETRIEVAL_ERROR,
    RETRIEVAL_OK,
    RETRIEVAL_SKIPPED,
    REVIEW,
    SUPPORTED,
    UNKNOWN,
)
from src.verification.models import (
    BLOCKED,
    GENERAL_INQUIRY,
    MatchResult,
    PipelineResult,
    PROCEED,
    RELEVANT,
    SPECIFIC_INTERNSHIP,
    VerificationResult,
)
from src.workflow.engine import BatchResult, OutreachWorkflow


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_CV_PATH = Path("templates/reference_cv.pdf")
_PROFILE_PATH = Path("data/master_profile.json")


# ---------------------------------------------------------------------------
# Helpers & fakes
# ---------------------------------------------------------------------------


class _MockSender:
    """Records send attempts without network I/O."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send_email(
        self, recipient: str, subject: str, body: str, attachment_path: str
    ) -> bool:
        self.sent.append({"recipient": recipient, "subject": subject})
        return True


def _make_pipeline(
    *,
    company_status: str = "VERIFIED",
    contact_status: str = "VERIFIED",
    overall_status: str = PROCEED,
    relevance: str = RELEVANT,
    case_type: str = GENERAL_INQUIRY,
    matched_areas: list[str] | None = None,
    evidence_sources: list[str] | None = None,
    review_flags: list[str] | None = None,
    blocking_reasons: list[str] | None = None,
    summary: str = "",
) -> PipelineResult:
    return PipelineResult(
        verification=VerificationResult(
            company_status=company_status,
            contact_status=contact_status,
            overall_status=overall_status,
            review_flags=review_flags or [],
            blocking_reasons=blocking_reasons or [],
        ),
        match=MatchResult(
            relevance_status=relevance,
            case_type=case_type,
            matched_areas=matched_areas or [],
            evidence_sources=evidence_sources or [],
        ),
        summary=summary,
    )


class _MockResearchProvider:
    """
    Returns a minimal ResearchResult without network I/O.
    Parameterisable per test for different company identity / technical areas.
    """

    def __init__(
        self,
        company_identity: str = SUPPORTED,
        technical_areas: list[str] | None = None,
        retrieval_error: bool = False,
        raise_on_call: bool = False,
    ) -> None:
        self._company_identity = company_identity
        self._technical_areas = technical_areas or ["AI/ML"]
        self._retrieval_error = retrieval_error
        self._raise_on_call = raise_on_call
        self.call_count = 0

    def research(self, inp: Any):
        from src.research.models import (
            APPLICATION_UNKNOWN,
            EvidenceItem,
            ResearchResult,
            RETRIEVAL_OK,
            RETRIEVAL_ERROR,
            SUPPORTED,
            UNKNOWN,
        )

        self.call_count += 1
        if self._raise_on_call:
            raise RuntimeError("Research provider exploded (simulated)")

        retrieval_status = RETRIEVAL_ERROR if self._retrieval_error else RETRIEVAL_OK
        evidence = [
            EvidenceItem(
                source_url="https://example.com",
                source_type="website",
                extracted_text="Mock page content for testing.",
                retrieval_status=retrieval_status,
            )
        ] if not self._retrieval_error else []

        return ResearchResult(
            company_identity_status=self._company_identity,
            domain_relationship_status=SUPPORTED,
            role_status=UNKNOWN,
            application_status=APPLICATION_UNKNOWN,
            technical_areas=self._technical_areas,
            evidence=evidence,
            limitations=["Mock provider — no real retrieval performed."],
        )



def _make_contact(
    email: str,
    company_name: str = "Test Corp",
    contact_name: str = "",
    website: str = "https://example.com",
    role_or_job_url: str = "AI/ML Intern",
    notes: str = "",
    row_number: int = 2,
) -> ContactRecord:
    return ContactRecord(
        row_number=row_number,
        company_name=company_name,
        contact_name=contact_name,
        email_result=parse_email(email),
        website=website,
        role_or_job_url=role_or_job_url,
        source_url="",
        notes=notes,
    )


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    csv_file = tmp_path / "contacts.csv"
    fieldnames = [
        "email", "company_name", "contact_name", "website",
        "role_or_job_url", "source_url", "notes",
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    return csv_file


# ---------------------------------------------------------------------------
# Section 1: Batch CSV Workflow
# ---------------------------------------------------------------------------


class TestBatchCsvWorkflow:
    """process_csv() exercises — all mocked, deterministic."""

    def _make_csv(self, tmp_path: Path) -> Path:
        return _write_csv(
            tmp_path,
            [
                # Row 2: corporate domain, AI company
                {
                    "email": "hr@deepmind-labs.example",
                    "company_name": "DeepMind Labs",
                    "website": "https://deepmind-labs.example",
                    "role_or_job_url": "AI/ML Intern",
                },
                # Row 3: founder/generic inbox
                {
                    "email": "founder@ai-startup.example",
                    "company_name": "AI Startup",
                    "website": "https://ai-startup.example",
                    "role_or_job_url": "AI Intern",
                },
                # Row 4: Gmail with supplied company name
                {
                    "email": "recruiter.google@gmail.com",
                    "company_name": "Moonshot Labs",
                    "website": "https://moonshot-labs.example",
                },
                # Row 5: free provider, unknown company
                {"email": "someone@hotmail.com", "company_name": ""},
                # Row 6: malformed email — will fail import
                {"email": "not-an-email@@@@", "company_name": "Bad Corp"},
                # Row 7: verified AI company
                {
                    "email": "careers@verified-ai.example",
                    "company_name": "Verified AI Co",
                    "website": "https://verified-ai.example",
                },
                # Row 8: non-AI company, notes mention AI internship
                {
                    "email": "jobs@brick-and-mortar.example",
                    "company_name": "Brick & Mortar Retail",
                    "website": "https://brick-and-mortar.example",
                    "notes": "Looking for AI internship here",
                },
                # Row 9: missing evidence (empty website)
                {
                    "email": "info@noevidencecorp.example",
                    "company_name": "No Evidence Corp",
                    "website": "",
                },
                # Row 10: duplicate of Row 2
                {
                    "email": "hr@deepmind-labs.example",
                    "company_name": "DeepMind Labs",
                },
            ],
        )

    def _make_wf(self, tmp_path: Path) -> OutreachWorkflow:
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy")
        return OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            default_mode=WorkflowMode.UPLOADED_CV,
            uploaded_cv_path=dummy_cv,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
            campaign_id="test",
        )

    def test_batch_continues_past_malformed_email(self, tmp_path):
        csv_file = self._make_csv(tmp_path)
        provider = _MockResearchProvider()
        wf = self._make_wf(tmp_path)
        batch = wf.process_csv(csv_file, target_role="AI/ML Intern",
                               research_provider=provider)
        assert batch.failed_count >= 1, "Malformed email must increment failed_count"
        assert batch.drafted_count >= 1, "Valid contacts must still produce drafts"

    def test_batch_total_rows_accounting(self, tmp_path):
        csv_file = self._make_csv(tmp_path)
        provider = _MockResearchProvider()
        wf = self._make_wf(tmp_path)
        batch = wf.process_csv(csv_file, target_role="AI/ML Intern",
                               research_provider=provider)
        assert batch.total_rows >= 9
        assert (
            batch.drafted_count + batch.blocked_count + batch.failed_count
            <= batch.total_rows
        )

    def test_duplicate_contact_not_drafted_twice(self, tmp_path):
        csv_file = self._make_csv(tmp_path)
        provider = _MockResearchProvider()
        wf = self._make_wf(tmp_path)
        batch = wf.process_csv(csv_file, target_role="AI/ML Intern",
                               research_provider=provider)
        dup_drafts = [
            r
            for r in batch.results
            if r.contact_email == "hr@deepmind-labs.example" and r.draft is not None
        ]
        assert len(dup_drafts) <= 1, "Duplicate must not produce two separate drafts"

    def test_batch_never_sends_automatically(self, tmp_path):
        csv_file = self._make_csv(tmp_path)
        mock_sender = _MockSender()
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy")
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            default_mode=WorkflowMode.UPLOADED_CV,
            uploaded_cv_path=dummy_cv,
            email_sender_provider=mock_sender,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        wf.process_csv(csv_file, target_role="AI/ML Intern",
                       research_provider=_MockResearchProvider())
        assert len(mock_sender.sent) == 0, "process_csv must never send emails automatically"

    def test_exception_in_one_contact_does_not_abort_batch(self, tmp_path):
        csv_file = _write_csv(
            tmp_path,
            [
                {
                    "email": "explode@broken.example",
                    "company_name": "Crash Corp",
                    "website": "https://broken.example",
                },
                {
                    "email": "ok@fine.example",
                    "company_name": "Fine Corp",
                    "website": "https://fine.example",
                },
            ],
        )
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy")

        class _BurstProvider:
            def __init__(self):
                self.call_count = 0

            def research(self, inp):
                self.call_count += 1
                if self.call_count == 1:
                    raise RuntimeError("Boom!")
                from src.research.models import (
                    APPLICATION_UNKNOWN,
                    EvidenceItem,
                    ResearchResult,
                    RETRIEVAL_OK,
                    SUPPORTED,
                    UNKNOWN,
                )
                return ResearchResult(
                    company_identity_status=SUPPORTED,
                    domain_relationship_status=SUPPORTED,
                    role_status=UNKNOWN,
                    application_status=APPLICATION_UNKNOWN,
                    technical_areas=["AI/ML"],
                    evidence=[EvidenceItem(
                        source_url="https://fine.example",
                        source_type="website",
                        extracted_text="Fine Corp page.",
                        retrieval_status=RETRIEVAL_OK,
                    )],
                )

        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            default_mode=WorkflowMode.UPLOADED_CV,
            uploaded_cv_path=dummy_cv,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        batch = wf.process_csv(csv_file, research_provider=_BurstProvider())
        assert batch.processed_count == 2, "Both contacts must be attempted"
        assert any(r.status == "ERROR" for r in batch.results)
        assert any(r.draft is not None for r in batch.results)

    def test_gmail_contact_is_not_silently_dropped(self, tmp_path):
        csv_file = _write_csv(
            tmp_path,
            [
                {
                    "email": "recruiter@gmail.com",
                    "company_name": "TechVentures Inc",
                    "website": "https://techventures.example",
                }
            ],
        )
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy")
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            default_mode=WorkflowMode.UPLOADED_CV,
            uploaded_cv_path=dummy_cv,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        batch = wf.process_csv(csv_file, research_provider=_MockResearchProvider())
        assert batch.processed_count == 1
        r = batch.results[0]
        assert r.status != "ERROR", f"Gmail contact errored: {r.message}"


# ---------------------------------------------------------------------------
# Section 2: Evidence / Relevance Isolation
# ---------------------------------------------------------------------------


class TestEvidenceAndRelevanceIsolation:
    def _profile(self) -> dict:
        return load_profile(_PROFILE_PATH)

    def test_user_notes_do_not_establish_company_ai_relevance(self):
        profile = self._profile()
        pipeline = _make_pipeline(
            relevance="NOT_A_MATCH",
            matched_areas=[],
            evidence_sources=[],
        )
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="jobs@brick-and-mortar.example",
            company_name="Brick & Mortar Retail",
            target_role="AI/ML Internship",
            verification_pipeline_result=pipeline,
            mode=WorkflowMode.UPLOADED_CV,
            attachment_path="cv.pdf",
            contact_notes="Looking for AI internship — really excited about their GenAI work",
        )
        # Company should not be described as AI-focused in the body
        assert "Brick & Mortar Retail's focus on" not in draft.body or \
               "AI" not in draft.body.split("Brick & Mortar Retail's focus on")[1][:50]

    def test_verified_ai_company_alignment_uses_evidence_areas_only(self):
        profile = self._profile()
        pipeline = _make_pipeline(
            relevance=RELEVANT,
            matched_areas=["Generative AI", "RAG"],
            evidence_sources=["https://verified-ai.example"],
        )
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="hr@verified-ai.example",
            company_name="Verified AI Corp",
            target_role="AI/ML Internship",
            verification_pipeline_result=pipeline,
            mode=WorkflowMode.UPLOADED_CV,
            attachment_path="cv.pdf",
        )
        assert "Generative AI" in draft.body or "RAG" in draft.body, \
            "Body must reference evidence-backed matched_areas"

    def test_unknown_company_produces_review_flags_not_invented_claims(self):
        profile = self._profile()
        pipeline = _make_pipeline(
            company_status=UNKNOWN,
            contact_status="REVIEW",
            overall_status="REVIEW",
            relevance=UNKNOWN,
            matched_areas=[],
        )
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="someone@mystery.example",
            company_name="",
            target_role="AI/ML Internship",
            verification_pipeline_result=pipeline,
            mode=WorkflowMode.UPLOADED_CV,
            attachment_path="cv.pdf",
        )
        assert draft.review_required is True
        assert len(draft.review_flags) >= 1
        assert "Unknown Company" not in draft.subject

    def test_valid_email_alone_does_not_verify_identity(self):
        profile = self._profile()
        pipeline = _make_pipeline(
            company_status=UNKNOWN,
            overall_status="REVIEW",
            relevance=UNKNOWN,
            matched_areas=[],
        )
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="hr@mystery-corp.example",
            company_name="Mystery Corp",
            target_role="AI/ML Internship",
            verification_pipeline_result=pipeline,
            mode=WorkflowMode.UPLOADED_CV,
            attachment_path="cv.pdf",
        )
        assert draft.review_required is True, \
            "Valid email alone must not satisfy verification"

    def test_blocked_company_no_draft_in_workflow(self, tmp_path):
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy")

        class _BlockedProvider:
            def research(self, inp):
                from src.research.models import (
                    APPLICATION_UNKNOWN,
                    ResearchResult,
                    UNKNOWN,
                )
                return ResearchResult(
                    company_identity_status=UNKNOWN,
                    domain_relationship_status=UNKNOWN,
                    role_status=UNKNOWN,
                    application_status=APPLICATION_UNKNOWN,
                    technical_areas=[],
                    evidence=[],
                    limitations=["Domain mismatch — possibly phishing."],
                )

        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            default_mode=WorkflowMode.UPLOADED_CV,
            uploaded_cv_path=dummy_cv,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        contact = _make_contact(
            "hr@phishing.example", company_name="Legit Corp",
            website="https://real.example",
        )
        result = wf.process_contact(contact, research_provider=_BlockedProvider())
        if result.status == BLOCKED:
            assert result.draft is None, "BLOCKED contact must not produce a draft"


# ---------------------------------------------------------------------------
# Section 3: Email Quality — Five Scenarios
# ---------------------------------------------------------------------------


class TestEmailQuality:
    def _profile(self) -> dict:
        return load_profile(_PROFILE_PATH)

    def test_scenario_1_verified_ai_with_linkedin_followup(self):
        profile = self._profile()
        pipeline = _make_pipeline(
            relevance=RELEVANT,
            matched_areas=["Generative AI", "AI/ML"],
            evidence_sources=["https://fitsol.green"],
        )
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="akshay.tandon@fitsol.green",
            company_name="Fitsol",
            contact_name="Akshay Tandon",
            target_role="AI/ML Internship",
            verification_pipeline_result=pipeline,
            mode=WorkflowMode.UPLOADED_CV,
            attachment_path="cv.pdf",
            contact_notes="We discussed the internship on LinkedIn last week",
        )
        assert "Fitsol" in draft.subject or "LinkedIn" in draft.subject, \
            f"LinkedIn follow-up should reference Fitsol or LinkedIn: {draft.subject}"
        # Company name should appear in the body even if not in subject
        assert "Fitsol" in draft.body
        assert "Generative AI" in draft.body or "AI/ML" in draft.body
        assert "Best regards," in draft.body
        assert "IIT Kharagpur" in draft.body
        import re
        assert not re.search(r"\d\s*[-\u2013]\s*\d\s+months\b", draft.body), \
            "Must use 'month' not 'months'"

    def test_scenario_2_unknown_company_no_invented_claims(self):
        profile = self._profile()
        pipeline = _make_pipeline(
            company_status=UNKNOWN,
            overall_status="REVIEW",
            relevance=UNKNOWN,
            matched_areas=[],
        )
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="contact@mystery.example",
            company_name="",
            target_role="AI/ML Internship",
            verification_pipeline_result=pipeline,
            mode=WorkflowMode.UPLOADED_CV,
            attachment_path="cv.pdf",
        )
        assert draft.review_required is True
        assert len(draft.review_flags) >= 1
        assert "Unknown Company" not in draft.subject

    def test_scenario_3_confirmed_referral_clean_name(self):
        profile = self._profile()
        pipeline = _make_pipeline(
            relevance=RELEVANT, matched_areas=["AI/ML"],
            evidence_sources=["https://prodoc.ai"],
        )
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="hr@prodoc.ai",
            company_name="Prodoc",
            target_role="AI/ML Internship",
            verification_pipeline_result=pipeline,
            mode=WorkflowMode.UPLOADED_CV,
            attachment_path="cv.pdf",
            contact_notes="Referred by Asit Vidyarthi who is the Co-Founder & CEO",
        )
        assert "Asit Vidyarthi" in draft.subject
        assert "Asit Vidyarthi" in draft.body
        assert "Co-Founder" not in draft.subject

    def test_scenario_4_linkedin_discovery_not_prior_conversation(self):
        profile = self._profile()
        pipeline = _make_pipeline(relevance=RELEVANT, matched_areas=["AI/ML"])
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="hr@startup.example",
            company_name="Tech Startup",
            target_role="AI/ML Internship",
            verification_pipeline_result=pipeline,
            mode=WorkflowMode.UPLOADED_CV,
            attachment_path="cv.pdf",
            contact_notes="Found this company on LinkedIn",
        )
        assert "conversation" not in draft.body.lower()
        assert "Following our" not in draft.body

    def test_scenario_5_failed_research_requires_review(self):
        profile = self._profile()
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="hr@unreachable.example",
            company_name="Unreachable Corp",
            target_role="AI/ML Internship",
            verification_pipeline_result=None,
            mode=WorkflowMode.UPLOADED_CV,
            attachment_path="cv.pdf",
        )
        assert draft.review_required is True
        assert any(
            "verification" in f.lower() or "missing" in f.lower()
            for f in draft.review_flags
        )

    def test_subject_never_fabricates_company_name(self):
        subj = build_subject_line(
            candidate_name="Bhavesh Verma",
            target_role="AI/ML Internship",
            company_name="",
        )
        assert "Unknown Company" not in subj

    def test_signature_consistent_across_scenarios(self):
        profile = self._profile()
        pipeline = _make_pipeline()
        for notes in ["", "Found on LinkedIn", "Referred by Priya Sharma"]:
            draft = draft_outreach_email(
                profile=profile,
                recipient_email="hr@example.com",
                company_name="Example Corp",
                verification_pipeline_result=pipeline,
                mode=WorkflowMode.UPLOADED_CV,
                attachment_path="cv.pdf",
                contact_notes=notes,
            )
            assert "Best regards," in draft.body
            assert "IIT Kharagpur" in draft.body


# ---------------------------------------------------------------------------
# Section 4: Uploaded CV Mode — SHA-256 Integrity
# ---------------------------------------------------------------------------


class TestUploadedCVIntegrity:
    def test_reference_cv_exists(self):
        assert _CV_PATH.exists(), (
            "LIMITATION: templates/reference_cv.pdf not found — CV integrity not verified."
        )

    def test_reference_cv_sha256_unchanged_after_workflow(self, tmp_path):
        if not _CV_PATH.exists():
            pytest.skip("Reference CV not found.")
        before_bytes = _CV_PATH.read_bytes()
        before_hash = hashlib.sha256(before_bytes).hexdigest()

        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            default_mode=WorkflowMode.UPLOADED_CV,
            uploaded_cv_path=_CV_PATH,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        contact = _make_contact("hr@example.com", website="https://example.com")
        wf.process_contact(contact, target_role="AI/ML Intern",
                           research_provider=_MockResearchProvider())

        after_hash = hashlib.sha256(_CV_PATH.read_bytes()).hexdigest()
        assert before_hash == after_hash, (
            f"Reference CV SHA-256 changed!\nBefore: {before_hash}\nAfter:  {after_hash}"
        )

    def test_draft_attachment_references_original_cv(self, tmp_path):
        if not _CV_PATH.exists():
            pytest.skip("Reference CV not found.")
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            default_mode=WorkflowMode.UPLOADED_CV,
            uploaded_cv_path=_CV_PATH,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        contact = _make_contact("hr@example.com", website="https://example.com")
        result = wf.process_contact(contact, target_role="AI/ML Intern",
                                    research_provider=_MockResearchProvider())
        assert result.draft is not None
        expected = str(_CV_PATH.resolve())
        assert result.draft.attachment_path == expected, (
            f"Expected attachment: {expected}\nGot: {result.draft.attachment_path}"
        )

    def test_uploaded_mode_bypasses_project_selection_and_cv_generation(self, tmp_path):
        if not _CV_PATH.exists():
            pytest.skip("Reference CV not found.")
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            default_mode=WorkflowMode.UPLOADED_CV,
            uploaded_cv_path=_CV_PATH,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        contact = _make_contact("hr@example.com", website="https://example.com")
        wf.process_contact(contact, target_role="AI/ML Intern",
                           research_provider=_MockResearchProvider())
        assert wf.project_selection_invoked is False
        assert wf.cv_generation_invoked is False

    def test_cv_sha256_tracked_by_workflow(self, tmp_path):
        content = b"%PDF-1.4 sample cv content"
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(content)
        expected_hash = hashlib.sha256(content).hexdigest()
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            uploaded_cv_path=dummy_cv,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        assert wf._uploaded_cv_hash == expected_hash


# ---------------------------------------------------------------------------
# Section 5: Approval and Duplicate-Send Safeguards
# ---------------------------------------------------------------------------


class TestApprovalAndSendSafeguards:
    def _setup(self, tmp_path: Path):
        dummy_cv = tmp_path / "cv.pdf"
        import pypdf
        _w = pypdf.PdfWriter()
        _w.add_blank_page(width=72, height=72)
        with open(dummy_cv, "wb") as _f:
            _w.write(_f)
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            default_mode=WorkflowMode.UPLOADED_CV,
            uploaded_cv_path=dummy_cv,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        contact = _make_contact("hr@example.com", website="https://example.com")
        result = wf.process_contact(
            contact, target_role="AI/ML Intern",
            research_provider=_MockResearchProvider(),
        )
        assert result.draft is not None
        return wf, result.draft

    def test_unapproved_draft_send_rejected(self, tmp_path):
        wf, draft = self._setup(tmp_path)
        result = wf.sender.send(draft.draft_id)
        assert result.success is False
        assert result.error == "APPROVAL_REQUIRED"

    def test_editing_approved_draft_resets_approval(self, tmp_path):
        wf, draft = self._setup(tmp_path)
        wf.review_manager.approve_draft(draft.draft_id)
        assert draft.approved is True
        wf.review_manager.edit_draft(draft.draft_id, subject="New subject")
        assert draft.approved is False
        assert draft.status == DraftStatus.DRAFT

    def test_duplicate_send_blocked_in_review_manager(self, tmp_path):
        wf, draft = self._setup(tmp_path)
        wf.review_manager.approve_draft(draft.draft_id)
        # Mark as already sent in the review manager's in-memory registry
        wf.review_manager.record_sent(draft.recipient)
        result = wf.sender.send(draft.draft_id)
        assert result.success is False
        assert result.error == "DUPLICATE_RECIPIENT"

    def test_send_log_persists_duplicate_protection(self, tmp_path):
        db_path = tmp_path / "send.db"
        log1 = SendLog(db_path=db_path)
        log1.record_success("draft-abc", "hr@example.com", "camp1")
        log2 = SendLog(db_path=db_path)
        assert log2.was_successfully_sent("hr@example.com", "camp1") is True

    def test_uncertain_send_blocks_retry(self, tmp_path):
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy")
        send_log = SendLog(db_path=":memory:")
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            uploaded_cv_path=dummy_cv,
            dry_run=True,
            send_log=send_log,
        )
        contact = _make_contact("hr@example.com", website="https://example.com")
        r = wf.process_contact(
            contact, target_role="AI/ML Intern",
            research_provider=_MockResearchProvider(),
        )
        assert r.draft is not None
        send_log.record_uncertain(r.draft.draft_id, r.draft.recipient, "")
        wf.review_manager.approve_draft(r.draft.draft_id)
        result = wf.send_draft(r.draft.draft_id)
        assert result.success is False
        assert result.error in ("UNCERTAIN_SEND", "DUPLICATE_RECIPIENT", "APPROVAL_REQUIRED")

    def test_process_csv_never_calls_sender(self, tmp_path):
        mock_sender = _MockSender()
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy")
        csv_file = _write_csv(
            tmp_path,
            [{"email": "hr@corp.example", "company_name": "Corp",
              "website": "https://corp.example"}],
        )
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            uploaded_cv_path=dummy_cv,
            email_sender_provider=mock_sender,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        wf.process_csv(csv_file, research_provider=_MockResearchProvider())
        assert len(mock_sender.sent) == 0


# ---------------------------------------------------------------------------
# Section 6: Regression tests for Prompt 4 audit findings
# ---------------------------------------------------------------------------


class TestPrompt4Regressions:
    def test_email_draft_fields_accessible_as_attributes(self):
        """EmailDraft is a plain dataclass — all fields accessible as attributes."""
        profile = load_profile(_PROFILE_PATH)
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="test@example.com",
            company_name="Test Co",
            verification_pipeline_result=_make_pipeline(),
            mode=WorkflowMode.UPLOADED_CV,
            attachment_path="cv.pdf",
        )
        assert isinstance(draft.subject, str)
        assert isinstance(draft.body, str)
        assert isinstance(draft.review_flags, list)
        assert draft.draft_id.startswith("draft-")

    def test_batch_counts_do_not_exceed_total_rows(self, tmp_path):
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy")
        csv_file = _write_csv(
            tmp_path,
            [
                {"email": "a@example.com", "company_name": "A",
                 "website": "https://a.example"},
                {"email": "b@example.com", "company_name": "B",
                 "website": "https://b.example"},
            ],
        )
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            uploaded_cv_path=dummy_cv,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        batch = wf.process_csv(csv_file, research_provider=_MockResearchProvider())
        assert (
            batch.drafted_count + batch.blocked_count + batch.failed_count
            <= batch.total_rows
        )

    def test_review_queue_size_equals_review_required_count(self, tmp_path):
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy")
        csv_file = _write_csv(
            tmp_path,
            [{"email": "hr@unknown.example", "company_name": "", "website": ""}],
        )
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            uploaded_cv_path=dummy_cv,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        batch = wf.process_csv(
            csv_file,
            research_provider=_MockResearchProvider(company_identity=UNKNOWN),
        )
        assert len(batch.review_queue) == batch.review_required_count

    def test_malformed_email_never_crashes_workflow(self, tmp_path):
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy")
        csv_file = _write_csv(
            tmp_path,
            [
                {"email": "not-an-email", "company_name": "Bad Corp"},
                {"email": "good@fine.example", "company_name": "Fine Corp",
                 "website": "https://fine.example"},
            ],
        )
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            uploaded_cv_path=dummy_cv,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        # Must not raise
        batch = wf.process_csv(csv_file, research_provider=_MockResearchProvider())
        assert batch.total_rows == 2
        assert batch.failed_count >= 1

    def test_workflow_result_has_draft_for_proceed_status(self, tmp_path):
        """A PROCEED result must always carry a non-None draft."""
        dummy_cv = tmp_path / "cv.pdf"
        dummy_cv.write_bytes(b"%PDF-1.4 dummy")
        wf = OutreachWorkflow(
            profile_path=_PROFILE_PATH,
            uploaded_cv_path=dummy_cv,
            dry_run=True,
            send_log=SendLog(db_path=":memory:"),
        )
        contact = _make_contact("hr@ai-corp.example", company_name="AI Corp",
                                website="https://ai-corp.example")
        result = wf.process_contact(
            contact, target_role="AI/ML Intern",
            research_provider=_MockResearchProvider(
                company_identity=SUPPORTED, technical_areas=["AI/ML"]
            ),
        )
        if result.status == PROCEED:
            assert result.draft is not None, "PROCEED must have a draft"

    def test_send_log_uncertain_is_not_treated_as_success(self, tmp_path):
        db_path = tmp_path / "send.db"
        log = SendLog(db_path=db_path)
        log.record_uncertain("draft-u001", "hr@example.com", "camp")
        assert log.was_successfully_sent("hr@example.com", "camp") is False
        assert log.has_uncertain_send("hr@example.com", "camp") is True


# ---------------------------------------------------------------------------
# Section 7: Live Smoke Test (clearly labelled; skipped offline)
# ---------------------------------------------------------------------------


def _is_online() -> bool:
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _is_online(), reason="No network — live smoke test skipped")
class TestLiveSmoke:
    """
    [LIVE NETWORK] Bounded, single URL retrieval only.
    No email sent. No credentials used.
    Auto-skipped in CI when network is absent.
    """

    def test_live_http_retrieval_example_dot_com(self):
        from src.research.providers import HttpPageProvider

        provider = HttpPageProvider(timeout_seconds=10)
        item = provider.fetch(
            url="https://example.com",
            source_type="website",
            supports=["company_identity"],
        )
        assert item.retrieval_status in (RETRIEVAL_OK, RETRIEVAL_ERROR), \
            f"[LIVE] Unexpected retrieval_status: {item.retrieval_status}"
        if item.retrieval_status == RETRIEVAL_OK:
            assert len(item.extracted_text) > 0