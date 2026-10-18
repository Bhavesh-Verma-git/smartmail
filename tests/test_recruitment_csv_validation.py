"""
tests/test_recruitment_csv_validation.py
----------------------------------------
Tests validating the actual recruitment CSV (smartmail_contacts.csv)
and workflow handling of email-only contacts, row indexing, review flags,
and approval safeguards without live external contact.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from src.contacts.csv_importer import (
    ContactRecord,
    import_contacts_from_csv,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
)
from src.outreach.mailer import EmailReviewManager, EmailSender
from src.outreach.models import DraftStatus, WorkflowMode
from src.outreach.send_log import SendLog
from src.profile.loader import load_profile
from src.research.models import EvidenceItem, RETRIEVAL_OK, RETRIEVAL_ERROR
from src.research.providers import ResearchProvider
from src.verification.models import BLOCKED
from src.workflow.engine import OutreachWorkflow


class MockSendProvider:
    """Mock email sender provider that records sent messages without network calls."""
    def __init__(self):
        self.sent = []

    def send_email(self, recipient: str, subject: str, body: str, attachment_path: str):
        self.sent.append({
            "recipient": recipient,
            "subject": subject,
            "body": body,
            "attachment_path": attachment_path,
        })


class MockResearchProvider(ResearchProvider):
    """
    Deterministic test double that returns pre-configured EvidenceItems
    without making live HTTP requests.  Unregistered URLs return a
    RETRIEVAL_ERROR item so tests remain hermetic.
    """
    def __init__(self, responses: dict[str, EvidenceItem] | None = None) -> None:
        self.responses = responses or {}

    def fetch(self, url: str, source_type: str, supports: list[str]) -> EvidenceItem:
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
            limitations=[f"Mocked test double: URL '{url}' not in mock responses."],
        )


class TestActualRecruitmentCSV:
    """Validation tests for smartmail_contacts.csv."""

    def test_csv_file_exists(self):
        csv_path = Path("smartmail_contacts.csv")
        assert csv_path.exists(), "smartmail_contacts.csv not found in project root"

    def test_csv_columns_compatibility(self):
        assert "email" in REQUIRED_COLUMNS
        assert len(REQUIRED_COLUMNS) == 1
        assert "company_name" in OPTIONAL_COLUMNS
        assert "website" in OPTIONAL_COLUMNS
        assert "contact_name" in OPTIONAL_COLUMNS

    def test_import_smartmail_contacts_count_and_uniqueness(self):
        result = import_contacts_from_csv("smartmail_contacts.csv")
        assert len(result.errors) == 0, f"Unexpected errors during import: {result.errors}"
        assert len(result.contacts) == 59, f"Expected 59 contacts, got {len(result.contacts)}"

        # Verify all email addresses are unique
        emails = [c.email_result.normalized_email for c in result.contacts]
        assert len(emails) == len(set(emails)), "Duplicate emails detected in parsed contacts"

    def test_row_indexing_scheme(self):
        result = import_contacts_from_csv("smartmail_contacts.csv")
        first_contact = result.contacts[0]
        last_contact = result.contacts[-1]

        # In 1-indexed CSV line numbering (where line 1 is header):
        # Line 1: header ("email")
        # Line 2: first contact ("cv@arabianfal.example")
        # Line 60: 59th contact ("hiringnow.ksa@example.com")
        assert first_contact.row_number == 2
        assert first_contact.email_result.normalized_email == "cv@arabianfal.example"

        assert last_contact.row_number == 60
        assert last_contact.email_result.normalized_email == "hiringnow.ksa@example.com"

    def _make_dummy_cv(self, path: Path) -> Path:
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with open(path, "wb") as f:
            writer.write(f)
        return path

    def test_email_only_contact_safely_blocked_without_fabrication(self, tmp_path: Path):
        dummy_cv = self._make_dummy_cv(tmp_path / "cv.pdf")

        result = import_contacts_from_csv("smartmail_contacts.csv")
        c1 = result.contacts[0]

        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            dry_run=True,
        )

        # Processing contact without fabricating company or role details
        wf_res = workflow.process_contact(c1)

        # Must halt safely as BLOCKED
        assert wf_res.status == BLOCKED
        assert wf_res.draft is None
        assert "Verification failed and marked as BLOCKED" in wf_res.message
        assert "Company identity and domain relationship are both unresolved" in wf_res.message

    def test_review_flagged_draft_approval_and_send_lifecycle(self, tmp_path: Path):
        dummy_cv = self._make_dummy_cv(tmp_path / "cv.pdf")

        result = import_contacts_from_csv("smartmail_contacts.csv")
        c1 = result.contacts[0]

        # Enrich contact with verified company context
        enriched_contact = ContactRecord(
            row_number=c1.row_number,
            company_name="Arabian Fal",
            contact_name="",
            email_result=c1.email_result,
            website="https://arabianfal.example",
            role_or_job_url="AI/ML Intern",
            source_url="",
            notes="",
        )

        mock_send_provider = MockSendProvider()
        db_path = tmp_path / "test_send.db"
        send_log = SendLog(db_path=db_path)

        # Research mock: arabianfal.example is reachable but its page text does NOT
        # contain the company name "Arabian Fal".  This deterministically yields
        # identity=REVIEW + domain=SUPPORTED → overall REVIEW, which is what
        # the approval lifecycle test requires.
        research_mock = MockResearchProvider({
            "https://arabianfal.example": EvidenceItem(
                source_url="https://arabianfal.example",
                source_type="official_website",
                title="Welcome",
                extracted_text="We provide professional services and solutions.",
                retrieved_at=EvidenceItem.now_utc(),
                retrieval_status=RETRIEVAL_OK,
                retrieval_note="HTTP 200 (mocked)",
            ),
            "https://www.arabianfal.example": EvidenceItem(
                source_url="https://www.arabianfal.example",
                source_type="official_website",
                title="Welcome",
                extracted_text="We provide professional services and solutions.",
                retrieved_at=EvidenceItem.now_utc(),
                retrieval_status=RETRIEVAL_OK,
                retrieval_note="HTTP 200 (mocked)",
            ),
        })

        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_cv,
            email_sender_provider=mock_send_provider,
            send_log=send_log,
            campaign_id="recruitment_test",
            dry_run=False,
        )

        wf_res = workflow.process_contact(
            enriched_contact,
            target_role="AI/ML Intern",
            research_provider=research_mock,
        )
        assert wf_res.status == "REVIEW"
        assert wf_res.draft is not None

        draft = wf_res.draft
        assert draft.review_required is True
        assert len(draft.review_flags) > 0
        assert draft.approved is False
        assert draft.status == DraftStatus.DRAFT

        # 1. Attempting to send unapproved draft is strictly blocked
        send_unapproved = workflow.send_draft(draft.draft_id)
        assert send_unapproved.success is False
        assert send_unapproved.error == "APPROVAL_REQUIRED"
        assert len(mock_send_provider.sent) == 0

        # 2. Explicit human approval marks draft approved
        approved = workflow.approve_draft(draft.draft_id)
        assert approved.approved is True
        assert approved.status == DraftStatus.APPROVED

        # 3. Sending approved draft succeeds via mock provider
        send_approved = workflow.send_draft(draft.draft_id)
        assert send_approved.success is True
        assert len(mock_send_provider.sent) == 1
        assert mock_send_provider.sent[0]["recipient"] == "cv@arabianfal.example"

        # 4. Duplicate send attempt is blocked
        send_duplicate = workflow.send_draft(draft.draft_id)
        assert send_duplicate.success is False
        assert send_duplicate.error == "DUPLICATE_RECIPIENT"
        assert len(mock_send_provider.sent) == 1
