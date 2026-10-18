"""
tests/test_email_safeguards.py
------------------------------
Tests for email review, approval, duplicate prevention, and verification safeguards.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.contacts.csv_importer import ContactRecord
from src.contacts.email_parser import parse_email
from src.outreach.drafting import draft_outreach_email
from src.outreach.mailer import EmailReviewManager, EmailSender
from src.outreach.models import DraftStatus, EmailDraft, WorkflowMode
from src.profile.loader import load_profile
from src.research.models import (
    APPLICATION_UNKNOWN,
    EvidenceItem,
    ResearchResult,
    REVIEW as RES_REVIEW,
    UNKNOWN,
)
from src.verification.models import BLOCKED, MatchResult, PipelineResult, VerificationResult
from src.workflow.engine import OutreachWorkflow


@pytest.fixture
def profile():
    return load_profile("data/master_profile.json")


@pytest.fixture
def sample_contact():
    return ContactRecord(
        row_number=2,
        company_name="Acme Corp",
        contact_name="Alice Smith",
        email_result=parse_email("alice@acme.example"),
        website="https://acme.example",
        role_or_job_url="AI Intern",
        source_url="https://acme.example/careers",
        notes="",
    )


class MockBlockedResearchProvider:
    """Simulates research where company identity contradicts or domain mismatches."""
    def research(self, inp):
        return ResearchResult(
            company_identity_status=RES_REVIEW,
            domain_relationship_status=RES_REVIEW,
            role_status=UNKNOWN,
            application_status=APPLICATION_UNKNOWN,
            technical_areas=[],
            limitations=["Domain owned by unrelated entity"],
        )


@pytest.fixture
def valid_pdf_file(tmp_path: Path) -> Path:
    """Generate an isolated, valid 1-page PDF fixture for tests without root files."""
    import pypdf
    w = pypdf.PdfWriter()
    w.add_blank_page(width=72, height=72)
    p = tmp_path / "valid_cv.pdf"
    with open(p, "wb") as f:
        w.write(f)
    return p


class MockSendingProvider:
    """Mock email provider to verify calls when explicitly approved."""
    def __init__(self):
        self.sent_emails = []

    def send_email(self, recipient, subject, body, attachment_path):
        self.sent_emails.append({
            "recipient": recipient,
            "subject": subject,
            "body": body,
            "attachment_path": attachment_path,
        })


class TestVerificationEnforcementInWorkflow:
    """Tests ensuring BLOCKED and REVIEW statuses are strictly enforced."""

    def test_blocked_verification_halts_workflow_and_creates_no_draft(
        self, sample_contact, tmp_path
    ):
        dummy_cv = tmp_path / "dummy.pdf"
        dummy_cv.write_bytes(b"dummy")

        workflow = OutreachWorkflow(uploaded_cv_path=dummy_cv)
        result = workflow.process_contact(
            contact=sample_contact,
            research_provider=MockBlockedResearchProvider(),
        )

        assert result.status == BLOCKED
        assert result.draft is None
        assert "Verification failed" in result.message

    def test_review_verification_creates_flagged_draft(self, profile):
        verif = VerificationResult(
            company_status="REVIEW",
            contact_status="REVIEW",
            overall_status="REVIEW",
            review_flags=["Company website did not list contact name."],
        )
        match = MatchResult(
            relevance_status="REVIEW",
            case_type="GENERAL_INQUIRY",
            matched_areas=[],
        )
        pipeline_res = PipelineResult(verification=verif, match=match)

        draft = draft_outreach_email(
            profile=profile,
            recipient_email="recruiter@example.com",
            company_name="Example Corp",
            verification_pipeline_result=pipeline_res,
            mode=WorkflowMode.UPLOADED_CV,
            attachment_path="cv.pdf",
        )

        assert draft.review_required is True
        assert len(draft.review_flags) > 0
        assert "Company website" in draft.review_flags[0]


class TestDraftEditingAndReview:
    """Tests for human review and editing of drafts."""

    def test_edit_draft_fields(self, profile):
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="old@example.com",
            company_name="Old Corp",
            attachment_path="old.pdf",
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)

        # User edits fields
        updated = manager.edit_draft(
            draft.draft_id,
            recipient="new@example.com",
            subject="Updated Subject",
            body="Updated Body Text",
            attachment_path="new.pdf",
        )

        assert updated.recipient == "new@example.com"
        assert updated.subject == "Updated Subject"
        assert updated.body == "Updated Body Text"
        assert updated.attachment_path == "new.pdf"
        assert updated.approved is False
        assert updated.status == DraftStatus.DRAFT


class TestSendingSafeguards:
    """Tests for explicit approval, duplicate prevention, and provider configuration."""

    def test_send_blocked_without_explicit_approval(self, profile, valid_pdf_file):
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="test@example.com",
            company_name="Test Corp",
            attachment_path=str(valid_pdf_file),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)
        sender = EmailSender(review_manager=manager, provider=MockSendingProvider())

        # Attempt to send without approval
        result = sender.send(draft.draft_id)
        assert result.success is False
        assert result.error == "APPROVAL_REQUIRED"
        assert "Explicit user approval is required" in result.message

    def test_send_succeeds_with_explicit_approval_and_provider(self, profile, valid_pdf_file):
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="test@example.com",
            company_name="Test Corp",
            attachment_path=str(valid_pdf_file),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)
        provider = MockSendingProvider()
        sender = EmailSender(review_manager=manager, provider=provider, dry_run=False)

        # Grant explicit approval
        manager.approve_draft(draft.draft_id)
        assert draft.approved is True
        assert draft.status == DraftStatus.APPROVED

        # Send
        result = sender.send(draft.draft_id)
        assert result.success is True
        assert draft.status == DraftStatus.SENT
        assert len(provider.sent_emails) == 1
        assert provider.sent_emails[0]["recipient"] == "test@example.com"

    def test_duplicate_send_is_strictly_blocked(self, profile, valid_pdf_file):
        draft1 = draft_outreach_email(
            profile=profile,
            recipient_email="candidate@target.example",
            company_name="Target Inc",
            attachment_path=str(valid_pdf_file),
        )
        draft2 = draft_outreach_email(
            profile=profile,
            recipient_email="candidate@target.example",  # Same recipient
            company_name="Target Inc",
            attachment_path=str(valid_pdf_file),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft1)
        manager.register_draft(draft2)
        provider = MockSendingProvider()
        sender = EmailSender(review_manager=manager, provider=provider, dry_run=False)

        # Approve and send draft 1
        manager.approve_draft(draft1.draft_id)
        res1 = sender.send(draft1.draft_id)
        assert res1.success is True

        # Approve and attempt to send draft 2 to the same recipient
        manager.approve_draft(draft2.draft_id)
        res2 = sender.send(draft2.draft_id)
        assert res2.success is False
        assert res2.error == "DUPLICATE_RECIPIENT"
        assert "already been sent" in res2.message
        # Provider should still only have 1 sent email
        assert len(provider.sent_emails) == 1

    def test_edit_invalidates_approval_and_resend_is_duplicate(self, profile, valid_pdf_file):
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="unique_test@target.example",
            company_name="Target Inc",
            attachment_path=str(valid_pdf_file),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)
        provider = MockSendingProvider()
        sender = EmailSender(review_manager=manager, provider=provider, dry_run=False)

        # 1. Unapproved send fails
        res1 = sender.send(draft.draft_id)
        assert res1.success is False
        assert res1.error == "APPROVAL_REQUIRED"

        # 2. Approve draft
        manager.approve_draft(draft.draft_id)
        assert draft.approved is True
        assert draft.status == DraftStatus.APPROVED

        # 3. Edit draft invalidates approval
        manager.edit_draft(draft.draft_id, subject="Revised Subject")
        assert draft.approved is False
        assert draft.status == DraftStatus.DRAFT

        # 4. Sending edited draft is blocked
        res2 = sender.send(draft.draft_id)
        assert res2.success is False
        assert res2.error == "APPROVAL_REQUIRED"

        # 5. Re-approve and send
        manager.approve_draft(draft.draft_id)
        res3 = sender.send(draft.draft_id)
        assert res3.success is True
        assert len(provider.sent_emails) == 1

        # 6. Resending already-sent draft is immediately blocked as DUPLICATE_RECIPIENT
        res4 = sender.send(draft.draft_id)
        assert res4.success is False
        assert res4.error == "DUPLICATE_RECIPIENT"

    def test_missing_gmail_configuration_reported_honestly(self, profile, valid_pdf_file, monkeypatch):
        # Clear any environment variables
        monkeypatch.delenv("GMAIL_CREDENTIALS_PATH", raising=False)
        monkeypatch.delenv("GMAIL_TOKEN_PATH", raising=False)

        draft = draft_outreach_email(
            profile=profile,
            recipient_email="hr@unconfigured.example",
            company_name="Unconfigured Corp",
            attachment_path=str(valid_pdf_file),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)
        manager.approve_draft(draft.draft_id)

        # No custom provider injected -> relies on system Gmail configuration
        sender = EmailSender(review_manager=manager, provider=None, dry_run=False)
        result = sender.send(draft.draft_id)

        assert result.success is False
        assert result.error == "GMAIL_NOT_CONFIGURED"
        assert "Gmail integration is not configured" in result.message
        assert "GMAIL_CREDENTIALS_PATH" in result.message

    def test_dry_run_prevents_network_transmission_even_with_provider_injected(self, profile, valid_pdf_file):
        """CRITICAL SAFEGUARD: dry_run=True must NEVER invoke network provider transmission."""
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="safeguard@target.example",
            company_name="Safe Corp",
            attachment_path=str(valid_pdf_file),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)
        manager.approve_draft(draft.draft_id)

        mock_provider = MockSendingProvider()
        sender = EmailSender(review_manager=manager, provider=mock_provider, dry_run=True)

        result = sender.send(draft.draft_id)
        assert result.success is True
        assert "Simulated send succeeded" in result.message
        assert draft.status == DraftStatus.SIMULATED_SENT
        assert len(mock_provider.sent_emails) == 0, "Provider must NEVER be invoked during dry run!"

    def test_persistent_duplicate_protection_across_restarts(
        self, sample_contact, tmp_path
    ):
        from src.outreach.send_log import SendLog
        import pypdf

        dummy_cv = tmp_path / "dummy_cv.pdf"
        _w = pypdf.PdfWriter()
        _w.add_blank_page(width=72, height=72)
        with open(dummy_cv, "wb") as _f:
            _w.write(_f)

        shared_db = tmp_path / "shared_sends.db"
        send_log = SendLog(db_path=shared_db)

        # Instance 1: Processes and sends to sample_contact
        wf1 = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            send_log=send_log,
            email_sender_provider=MockSendingProvider(),
        )
        res1 = wf1.process_contact(
            contact=sample_contact,
            research_provider=None,
        )
        assert res1.status in ("PROCEED", "REVIEW")
        wf1.approve_draft(res1.draft.draft_id)
        send_res1 = wf1.send_draft(res1.draft.draft_id)
        assert send_res1.success is True

        # Instance 2: Simulates a completely new process restart with same db
        send_log2 = SendLog(db_path=shared_db)
        wf2 = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            send_log=send_log2,
            email_sender_provider=MockSendingProvider(),
        )
        res2 = wf2.process_contact(
            contact=sample_contact,
            research_provider=None,
        )
        wf2.approve_draft(res2.draft.draft_id)
        send_res2 = wf2.send_draft(res2.draft.draft_id)

        assert send_res2.success is False
        assert send_res2.error == "DUPLICATE_RECIPIENT"
        assert "already been sent" in send_res2.message

    def test_uncertain_send_outcome_blocks_blind_retry(
        self, sample_contact, tmp_path
    ):
        from src.outreach.send_log import SendLog
        import pypdf

        dummy_cv = tmp_path / "dummy_cv.pdf"
        _w = pypdf.PdfWriter()
        _w.add_blank_page(width=72, height=72)
        with open(dummy_cv, "wb") as _f:
            _w.write(_f)
        shared_db = tmp_path / "shared_sends.db"
        send_log = SendLog(db_path=shared_db)

        # Pre-record an uncertain send for this recipient
        send_log.record_draft("d-uncertain", sample_contact.email_result.normalized_email)
        send_log.record_uncertain("d-uncertain", sample_contact.email_result.normalized_email)

        wf = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            send_log=send_log,
            email_sender_provider=MockSendingProvider(),
        )
        res = wf.process_contact(
            contact=sample_contact,
            research_provider=None,
        )
        wf.approve_draft(res.draft.draft_id)
        send_res = wf.send_draft(res.draft.draft_id)

        assert send_res.success is False
        assert send_res.error == "UNCERTAIN_SEND"
        assert "uncertain outcome" in send_res.message


class TestAttachmentValidation:
    """Comprehensive validation of attachment files per audit requirements."""

    def test_missing_attachment_file_fails(self, profile, tmp_path):
        missing_path = tmp_path / "does_not_exist.pdf"
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="test@example.com",
            company_name="Test Corp",
            attachment_path=str(missing_path),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)
        manager.approve_draft(draft.draft_id)
        sender = EmailSender(review_manager=manager, provider=MockSendingProvider(), dry_run=False)

        res = sender.send(draft.draft_id)
        assert res.success is False
        assert res.error == "ATTACHMENT_NOT_FOUND"
        assert "does not exist on disk" in res.message

    def test_directory_as_attachment_fails(self, profile, tmp_path):
        folder = tmp_path / "not_a_file.pdf"
        folder.mkdir()
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="test@example.com",
            company_name="Test Corp",
            attachment_path=str(folder),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)
        manager.approve_draft(draft.draft_id)
        sender = EmailSender(review_manager=manager, provider=MockSendingProvider(), dry_run=False)

        res = sender.send(draft.draft_id)
        assert res.success is False
        assert res.error == "INVALID_ATTACHMENT"
        assert "not a regular file" in res.message

    def test_non_pdf_extension_fails(self, profile, tmp_path):
        txt_file = tmp_path / "cv.txt"
        txt_file.write_text("plain text resume")
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="test@example.com",
            company_name="Test Corp",
            attachment_path=str(txt_file),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)
        manager.approve_draft(draft.draft_id)
        sender = EmailSender(review_manager=manager, provider=MockSendingProvider(), dry_run=False)

        res = sender.send(draft.draft_id)
        assert res.success is False
        assert res.error == "INVALID_ATTACHMENT"
        assert ".pdf extension required" in res.message

    def test_corrupted_pdf_missing_magic_header_fails(self, profile, tmp_path):
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"NOT A PDF AT ALL 123456789")
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="test@example.com",
            company_name="Test Corp",
            attachment_path=str(bad_pdf),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)
        manager.approve_draft(draft.draft_id)
        sender = EmailSender(review_manager=manager, provider=MockSendingProvider(), dry_run=False)

        res = sender.send(draft.draft_id)
        assert res.success is False
        assert res.error == "INVALID_ATTACHMENT"
        assert "missing %PDF- header" in res.message

    def test_corrupted_pdf_truncated_content_fails(self, profile, tmp_path):
        bad_pdf = tmp_path / "truncated.pdf"
        bad_pdf.write_bytes(b"%PDF-1.4\nTRUNCATED_AND_BROKEN_CONTENT")
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="test@example.com",
            company_name="Test Corp",
            attachment_path=str(bad_pdf),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)
        manager.approve_draft(draft.draft_id)
        sender = EmailSender(review_manager=manager, provider=MockSendingProvider(), dry_run=False)

        res = sender.send(draft.draft_id)
        assert res.success is False
        assert res.error == "INVALID_ATTACHMENT"
        assert "not a parseable PDF" in res.message

    def test_valid_pdf_succeeds(self, profile, valid_pdf_file):
        draft = draft_outreach_email(
            profile=profile,
            recipient_email="test@example.com",
            company_name="Test Corp",
            attachment_path=str(valid_pdf_file),
        )
        manager = EmailReviewManager()
        manager.register_draft(draft)
        manager.approve_draft(draft.draft_id)
        provider = MockSendingProvider()
        sender = EmailSender(review_manager=manager, provider=provider, dry_run=False)

        res = sender.send(draft.draft_id)
        assert res.success is True
        assert res.error is None
        assert len(provider.sent_emails) == 1
