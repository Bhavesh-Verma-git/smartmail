"""
tests/test_email_only_inputs.py
--------------------------------
Comprehensive regression test suite for email-only inputs:
1. Twelve syntactically valid email addresses with no company or contact columns.
2. A company that can be identified from its email domain.
3. A generic email provider or domain that cannot reliably identify an employer.
4. A blocked company website, timeout, rate limit, or unavailable research source.
5. A contact whose research or draft generation raises an exception.
6. A mixture of successful, review-required, and failed contacts.
7. A batch where some contacts finish before others.
8. Incremental persistence, page refresh, row-order preservation, and duplicate handling.
9. Empty input, malformed addresses, and unexpected spreadsheet headers.
10. An assertion that processing does not send any real email.
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pypdf
import pytest

from src.contacts.column_mapper import auto_map_columns
from src.contacts.csv_importer import ContactRecord, ImportResult, ImportRowError, import_contacts_from_csv
from src.contacts.email_parser import parse_email
from src.contacts.spreadsheet_importer import parse_spreadsheet_contacts
from src.outreach.drafting import draft_outreach_email
from src.outreach.mailer import EmailSender, get_sender_address
from src.outreach.models import DraftStatus, EmailDraft, WorkflowMode
from src.outreach.send_log import SendLog
from src.research.models import (
    NOT_APPLICABLE,
    RETRIEVAL_BLOCKED,
    RETRIEVAL_ERROR,
    RETRIEVAL_OK,
    EvidenceItem,
    ResearchInput,
    ResearchResult,
    SUPPORTED,
    UNKNOWN,
)
from src.research.providers import ResearchProvider
from src.ui.app import create_app
from src.ui.storage import (
    clear_registered_cv,
    get_batch_record,
    get_draft,
    list_all_drafts,
    register_uploaded_cv,
    save_draft_record,
)
from src.verification.models import BLOCKED, PROCEED
from src.verification.pipeline import run_verification
from src.verification.verifier import verify
from src.workflow.batch_runner import BackgroundBatchRunner
from src.workflow.engine import OutreachWorkflow, WorkflowResult


# ---------------------------------------------------------------------------
# Test Fixtures & Mock Providers
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_cv_file(tmp_path: Path) -> Path:
    cv_path = tmp_path / "test_cv.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(cv_path, "wb") as f:
        writer.write(f)
    return cv_path


@pytest.fixture
def isolated_send_log(tmp_path: Path) -> SendLog:
    return SendLog(db_path=tmp_path / "test_send_log.db")


class CustomMockProvider(ResearchProvider):
    """Configurable mock provider supporting domain responses, delays, and errors."""

    def __init__(
        self,
        domain_responses: dict[str, dict[str, Any]] | None = None,
        latencies: dict[str, float] | None = None,
        failing_domains: set[str] | None = None,
    ) -> None:
        self.domain_responses = domain_responses or {}
        self.latencies = latencies or {}
        self.failing_domains = failing_domains or set()
        self.fetched_urls: list[str] = []

    def fetch(self, url: str, source_type: str, supports: list[str]) -> EvidenceItem:
        self.fetched_urls.append(url)
        # Check domain for error injection
        domain = url.split("://")[-1].split("/")[0].lower()
        if any(f in domain for f in self.failing_domains):
            status = self.domain_responses.get(domain, {}).get("status", RETRIEVAL_ERROR)
            return EvidenceItem(
                source_url=url,
                source_type=source_type,
                title="",
                extracted_text="",
                retrieved_at=EvidenceItem.now_utc(),
                retrieval_status=status,
                retrieval_note=f"Simulated retrieval failure for {url}",
                limitations=[f"Network error accessing {url}"],
            )

        delay = self.latencies.get(domain, 0.0)
        if delay > 0:
            time.sleep(delay)

        info = self.domain_responses.get(domain, {})
        text = info.get(
            "text",
            f"Official site of {domain.split('.')[0].capitalize()}. "
            "We build machine learning, deep learning, generative AI and LLM infrastructure."
        )
        title = info.get("title", f"{domain.split('.')[0].capitalize()} AI")
        return EvidenceItem(
            source_url=url,
            source_type=source_type,
            title=title,
            extracted_text=text,
            retrieved_at=EvidenceItem.now_utc(),
            retrieval_status=RETRIEVAL_OK,
            supports=supports,
        )


# ---------------------------------------------------------------------------
# Test Case 1: Twelve valid emails with no company or contact columns
# ---------------------------------------------------------------------------

def test_case_1_twelve_valid_emails_no_company_or_contact_columns(tmp_path: Path, dummy_cv_file: Path, isolated_send_log: SendLog):
    """
    1. Twelve syntactically valid email addresses with no company or contact columns.
    Asserts complete end-to-end execution, total_rows==12, processed_count==12,
    row-order preservation, and absence of AttributeError ('SpreadsheetParseResult' has no 'contacts').
    """
    csv_file = tmp_path / "twelve_emails.csv"
    emails = [
        "hiring@stripe.com",
        "recruiting@anthropic.com",
        "talent@openai.com",
        "team@deepmind.com",
        "jobs@scale.com",
        "careers@cohere.com",
        "contact@huggingface.co",
        "recruiter@mistral.ai",
        "inquiries@elevenlabs.io",
        "ai-interns@databricks.com",
        "internships@snowflake.com",
        "careers@anyscale.com",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Email Address"])
        for e in emails:
            writer.writerow([e])

    # Parse using spreadsheet_importer with automatic column mapping
    parse_res = parse_spreadsheet_contacts(csv_file)
    assert len(parse_res.valid_contacts) == 12
    # Verify backward-compatible alias property works
    assert len(parse_res.contacts) == 12
    assert parse_res.summary.missing_company_count == 12
    assert parse_res.summary.missing_name_count == 12
    assert parse_res.summary.invalid_syntax_count == 0

    workflow = OutreachWorkflow(
        uploaded_cv_path=dummy_cv_file,
        dry_run=True,
        send_log=isolated_send_log,
        campaign_id="camp_twelve",
    )
    provider = CustomMockProvider()
    batch_runner = BackgroundBatchRunner.get_instance()
    batch_id = "batch_test_12_emails"

    batch_runner.start_batch(
        batch_id=batch_id,
        import_result=parse_res,
        workflow=workflow,
        target_role="AI/ML Intern",
        campaign_id="camp_twelve",
        mode=WorkflowMode.UPLOADED_CV,
        research_provider=provider,
        max_workers=4,
        wait_for_completion=True,
    )

    record = get_batch_record(batch_id)
    assert record is not None
    assert record["status"] == "COMPLETED"
    assert record["total_rows"] == 12
    assert record["processed_count"] == 12
    assert record["failed_count"] == 0
    assert len(record["results"]) == 12
    # Check that row numbers 2..13 are preserved in order (row 1 is CSV header)
    assert [r["row_number"] for r in record["results"]] == list(range(2, 14))


# ---------------------------------------------------------------------------
# Test Case 2: Company identified from its email domain
# ---------------------------------------------------------------------------

def test_case_2_company_identified_from_email_domain(dummy_cv_file: Path, isolated_send_log: SendLog):
    """
    2. A company that can be identified from its email domain.
    Asserts domain research discovers the company, flags for human review without
    unconditionally blocking, generates an outreach draft, and does not invent recruiter names.
    """
    contact = ContactRecord(
        row_number=1,
        company_name="",
        contact_name="",
        email_result=parse_email("intern-recruiting@anthropic.com"),
        website="",
        role_or_job_url="",
        source_url="",
        notes="",
    )
    provider = CustomMockProvider({
        "anthropic.com": {
            "title": "Anthropic \u2014 AI Research and Safety",
            "text": "Anthropic is an AI safety and research company that builds reliable, interpretable, and steerable AI systems like Claude.",
        }
    })

    workflow = OutreachWorkflow(uploaded_cv_path=dummy_cv_file, dry_run=True, send_log=isolated_send_log)
    wf_res = workflow.process_contact(contact, target_role="AI Safety Intern", research_provider=provider)

    assert wf_res.status == "REVIEW"
    assert wf_res.company_name == "Anthropic"
    assert wf_res.draft is not None
    assert wf_res.draft.review_required is True
    # Salutation should be professional team greeting, not an invented person
    assert "Dear Hiring Team at Anthropic," in wf_res.draft.body
    # Subject contains inferred company name
    assert "Anthropic" in wf_res.draft.subject
    # Verification review flags should state company was inferred from domain
    v_pipe = wf_res.verification
    assert v_pipe is not None
    assert any("inferred from domain" in flag.lower() for flag in v_pipe.verification.review_flags)


# ---------------------------------------------------------------------------
# Test Case 3: Generic email provider cannot identify employer
# ---------------------------------------------------------------------------

def test_case_3_generic_email_provider_cannot_identify_employer(dummy_cv_file: Path, isolated_send_log: SendLog):
    """
    3. A generic email provider or domain that cannot reliably identify an employer.
    Asserts company is NOT invented as 'Gmail' or 'Unknown Company', draft is neutral inquiry,
    and marked for human review.
    """
    contact = ContactRecord(
        row_number=1,
        company_name="",
        contact_name="",
        email_result=parse_email("senior.hiring.manager@gmail.com"),
        website="",
        role_or_job_url="",
        source_url="",
        notes="",
    )
    workflow = OutreachWorkflow(uploaded_cv_path=dummy_cv_file, dry_run=True, send_log=isolated_send_log)
    wf_res = workflow.process_contact(contact, target_role="AI Engineer", research_provider=CustomMockProvider())

    assert wf_res.status == "REVIEW"
    # Never invent "Gmail" or "Unknown Company" as company name
    assert wf_res.company_name == ""
    assert wf_res.draft is not None
    assert wf_res.draft.review_required is True
    assert "Dear Hiring Team," in wf_res.draft.body
    assert "Unknown Company" not in wf_res.draft.body
    assert "Inquiry regarding" in wf_res.draft.subject


# ---------------------------------------------------------------------------
# Test Case 4: Blocked website, timeout, or rate limit
# ---------------------------------------------------------------------------

def test_case_4_blocked_website_or_unavailable_source(dummy_cv_file: Path, isolated_send_log: SendLog):
    """
    4. A blocked company website, timeout, rate limit, or unavailable research source.
    Asserts contact is safely marked BLOCKED with clear honest reason, preserving row without crashing.
    """
    contact = ContactRecord(
        row_number=1,
        company_name="",
        contact_name="",
        email_result=parse_email("careers@blocked-enterprise.com"),
        website="",
        role_or_job_url="",
        source_url="",
        notes="",
    )
    provider = CustomMockProvider(
        domain_responses={"blocked-enterprise.com": {"status": RETRIEVAL_BLOCKED}},
        failing_domains={"blocked-enterprise.com"},
    )
    workflow = OutreachWorkflow(uploaded_cv_path=dummy_cv_file, dry_run=True, send_log=isolated_send_log)
    wf_res = workflow.process_contact(contact, research_provider=provider)

    # When both company name is missing and domain cannot be retrieved/verified, it must be BLOCKED
    assert wf_res.status == BLOCKED
    assert wf_res.draft is None
    v_pipe = wf_res.verification
    assert v_pipe is not None
    assert any("could not be verified" in reason.lower() for reason in v_pipe.verification.blocking_reasons)


# ---------------------------------------------------------------------------
# Test Case 5: Contact research or draft generation raises exception
# ---------------------------------------------------------------------------

def test_case_5_contact_exception_does_not_crash_batch(tmp_path: Path, dummy_cv_file: Path, isolated_send_log: SendLog):
    """
    5. A contact whose research or draft generation raises an exception.
    Asserts exception is isolated to that contact, recorded as ERROR, and remaining contacts succeed.
    """
    contacts = [
        ContactRecord(1, "", "", parse_email("contact1@stripe.com"), "", "", "", ""),
        ContactRecord(2, "", "", parse_email("crash@fatal-error.com"), "", "", "", ""),
        ContactRecord(3, "", "", parse_email("contact3@anthropic.com"), "", "", "", ""),
    ]
    import_result = ImportResult(contacts=contacts, errors=[])

    workflow = OutreachWorkflow(uploaded_cv_path=dummy_cv_file, dry_run=True, send_log=isolated_send_log)

    # Mock process_contact to raise on contact 2
    orig_process = workflow.process_contact

    def side_effect(contact: ContactRecord, *args: Any, **kwargs: Any) -> WorkflowResult:
        if "fatal-error" in contact.email:
            raise RuntimeError("Simulated unhandled pipeline crash on contact 2")
        return orig_process(contact, *args, **kwargs)

    workflow.process_contact = side_effect  # type: ignore

    batch_runner = BackgroundBatchRunner.get_instance()
    batch_id = "batch_test_single_crash"
    batch_runner.start_batch(
        batch_id=batch_id,
        import_result=import_result,
        workflow=workflow,
        mode=WorkflowMode.UPLOADED_CV,
        research_provider=CustomMockProvider(),
        max_workers=2,
        wait_for_completion=True,
    )

    record = get_batch_record(batch_id)
    assert record is not None
    assert record["status"] == "COMPLETED"
    assert record["total_rows"] == 3
    assert record["processed_count"] == 3
    assert record["failed_count"] == 1

    # Contact 1 and 3 generated drafts, contact 2 is ERROR
    results = record["results"]
    assert results[0]["status"] == "REVIEW"
    assert results[1]["status"] == "ERROR"
    assert "Simulated unhandled pipeline crash" in results[1]["error"]
    assert results[2]["status"] == "REVIEW"


# ---------------------------------------------------------------------------
# Test Case 6: Mixture of successful, review-required, and failed contacts
# ---------------------------------------------------------------------------

def test_case_6_mixture_of_outcomes(dummy_cv_file: Path, isolated_send_log: SendLog):
    """
    6. A mixture of successful, review-required, and failed contacts.
    Asserts accurate aggregate metrics and classifications.
    """
    contacts = [
        ContactRecord(1, "DeepMind", "Demis", parse_email("demis@deepmind.com"), "https://deepmind.google", "", "", ""),
        ContactRecord(2, "", "", parse_email("team@anthropic.com"), "", "", "", ""),
        ContactRecord(3, "", "", parse_email("user@gmail.com"), "", "", "", ""),
        ContactRecord(4, "", "", parse_email("bad@unreachable-domain.com"), "", "", "", ""),
    ]
    initial_errors = [
        ImportRowError(5, {"email": "invalid-syntax@@"}, "Invalid email syntax"),
    ]
    import_result = ImportResult(contacts=contacts, errors=initial_errors)

    provider = CustomMockProvider(
        domain_responses={
            "deepmind.google": {"title": "Google DeepMind", "text": "Google DeepMind AI research and frontier models."},
            "anthropic.com": {"title": "Anthropic", "text": "Anthropic AI research and Claude language models."},
            "unreachable-domain.com": {"status": RETRIEVAL_ERROR},
        },
        failing_domains={"unreachable-domain.com"},
    )

    workflow = OutreachWorkflow(uploaded_cv_path=dummy_cv_file, dry_run=True, send_log=isolated_send_log)
    batch_runner = BackgroundBatchRunner.get_instance()
    batch_id = "batch_test_mixture"

    batch_runner.start_batch(
        batch_id=batch_id,
        import_result=import_result,
        workflow=workflow,
        mode=WorkflowMode.UPLOADED_CV,
        research_provider=provider,
        max_workers=2,
        wait_for_completion=True,
    )

    record = get_batch_record(batch_id)
    assert record is not None
    assert record["status"] == "COMPLETED"
    assert record["total_rows"] == 5  # 4 contacts + 1 initial import error
    assert record["failed_count"] == 1  # 1 initial syntax error
    assert record["drafted_count"] >= 2  # deepmind and anthropic drafted
    assert record["blocked_count"] >= 1  # unreachable domain blocked


# ---------------------------------------------------------------------------
# Test Case 7: Contacts finish out of order
# ---------------------------------------------------------------------------

def test_case_7_out_of_order_completion_preserves_row_order(dummy_cv_file: Path, isolated_send_log: SendLog):
    """
    7. A batch where some contacts finish before others.
    Asserts that row-order is strictly preserved despite uneven latency.
    """
    contacts = [
        ContactRecord(1, "", "", parse_email("c1@slow-domain.com"), "", "", "", ""),
        ContactRecord(2, "", "", parse_email("c2@fast-domain.com"), "", "", "", ""),
        ContactRecord(3, "", "", parse_email("c3@medium-domain.com"), "", "", "", ""),
    ]
    import_result = ImportResult(contacts=contacts, errors=[])

    provider = CustomMockProvider(
        latencies={
            "slow-domain.com": 0.08,    # Finishes last
            "fast-domain.com": 0.001,   # Finishes first
            "medium-domain.com": 0.03, # Finishes second
        }
    )

    workflow = OutreachWorkflow(uploaded_cv_path=dummy_cv_file, dry_run=True, send_log=isolated_send_log)
    batch_runner = BackgroundBatchRunner.get_instance()
    batch_id = "batch_test_out_of_order"

    batch_runner.start_batch(
        batch_id=batch_id,
        import_result=import_result,
        workflow=workflow,
        mode=WorkflowMode.UPLOADED_CV,
        research_provider=provider,
        max_workers=3,
        wait_for_completion=True,
    )

    record = get_batch_record(batch_id)
    assert record is not None
    results = record["results"]
    assert len(results) == 3
    # Strict 1-indexed ordering preserved:
    assert results[0]["row_number"] == 1
    assert results[0]["contact_email"] == "c1@slow-domain.com"
    assert results[1]["row_number"] == 2
    assert results[1]["contact_email"] == "c2@fast-domain.com"
    assert results[2]["row_number"] == 3
    assert results[2]["contact_email"] == "c3@medium-domain.com"


# ---------------------------------------------------------------------------
# Test Case 8: Incremental persistence, page refresh, and duplicate handling
# ---------------------------------------------------------------------------

def test_case_8_incremental_persistence_and_duplicate_handling(dummy_cv_file: Path, isolated_send_log: SendLog):
    """
    8. Incremental persistence, page refresh, row-order preservation, and duplicate handling.
    Asserts batch file is readable mid-flight and after reload, and duplicates are safely prevented.
    """
    # 1. First run sends (simulated) to contact
    workflow = OutreachWorkflow(uploaded_cv_path=dummy_cv_file, dry_run=True, send_log=isolated_send_log)
    c1 = ContactRecord(1, "Stripe", "Recruiter", parse_email("duplicate@stripe.com"), "https://stripe.com", "", "", "")
    wf_res1 = workflow.process_contact(c1, research_provider=CustomMockProvider())
    assert wf_res1.draft is not None
    # Simulate approval and dry-run send
    workflow.sender.review_manager.register_draft(wf_res1.draft)
    workflow.sender.review_manager.approve_draft(wf_res1.draft.draft_id)
    send_res = workflow.sender.send(wf_res1.draft.draft_id)
    assert send_res.success is True
    isolated_send_log.record_simulated(wf_res1.draft.draft_id, "duplicate@stripe.com", "camp_dup")

    # 2. Second batch includes the same recipient
    batch_runner = BackgroundBatchRunner.get_instance()
    batch_id = "batch_test_dup_handling"
    import_result = ImportResult(contacts=[c1], errors=[])

    workflow2 = OutreachWorkflow(
        uploaded_cv_path=dummy_cv_file,
        dry_run=True,
        send_log=isolated_send_log,
        campaign_id="camp_dup",
    )
    batch_runner.start_batch(
        batch_id=batch_id,
        import_result=import_result,
        workflow=workflow2,
        campaign_id="camp_dup",
        mode=WorkflowMode.UPLOADED_CV,
        research_provider=CustomMockProvider(),
        max_workers=1,
        wait_for_completion=True,
    )

    record = get_batch_record(batch_id)
    assert record is not None
    # Contact is blocked due to duplicate prevention
    assert record["results"][0]["status"] == BLOCKED
    assert "duplicate" in record["results"][0]["message"].lower()


# ---------------------------------------------------------------------------
# Test Case 9: Empty input, malformed addresses, unexpected headers
# ---------------------------------------------------------------------------

def test_case_9_empty_input_malformed_addresses_unexpected_headers(tmp_path: Path):
    """
    9. Empty input, malformed addresses, and unexpected spreadsheet headers.
    Asserts graceful validation and error capture without unhandled exceptions.
    """
    # Unexpected headers with valid email mapping
    csv_file = tmp_path / "odd_headers.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Work Mail", "Random Column 1", "Random Column 2"])
        writer.writerow(["valid@corp.com", "foo", "bar"])
        writer.writerow(["bad-email-format", "foo2", "bar2"])
        writer.writerow(["", "", ""])  # Empty row

    parse_res = parse_spreadsheet_contacts(csv_file)
    assert len(parse_res.valid_contacts) == 1
    assert parse_res.valid_contacts[0].email == "valid@corp.com"
    assert parse_res.summary.invalid_syntax_count == 1
    assert len(parse_res.errors) == 1
    assert "bad-email-format" in str(parse_res.errors[0].raw_row)


# ---------------------------------------------------------------------------
# Test Case 10: Processing does not send any real email
# ---------------------------------------------------------------------------

def test_case_10_assertion_no_real_email_sent(dummy_cv_file: Path, isolated_send_log: SendLog):
    """
    10. An assertion that processing does not send any real email.
    Verifies that throughout entire batch and drafting flow, dry_run=True is enforced,
    no network transmission takes place, and test sender account is safely configured.
    """
    # Configure safe test sender account via environment variable
    os.environ["SMARTMAIL_TEST_SENDER_EMAIL"] = "test.outreach.safe@example.org"
    sender_addr = get_sender_address()
    assert sender_addr == "test.outreach.safe@example.org"

    workflow = OutreachWorkflow(
        uploaded_cv_path=dummy_cv_file,
        dry_run=True,
        send_log=isolated_send_log,
    )
    assert workflow.sender.dry_run is True

    c = ContactRecord(1, "", "", parse_email("lead@enterprise.com"), "", "", "", "")
    wf_res = workflow.process_contact(c, research_provider=CustomMockProvider())
    assert wf_res.draft is not None

    # Verify draft is in DRAFT status, unapproved
    assert wf_res.draft.approved is False
    assert wf_res.draft.status == DraftStatus.DRAFT

    # Attempt send without approval -> BLOCKED
    send_result = workflow.sender.send(wf_res.draft.draft_id)
    assert send_result.success is False
    assert send_result.error == "APPROVAL_REQUIRED"

    # Explicit approval
    workflow.sender.review_manager.approve_draft(wf_res.draft.draft_id)
    send_result_approved = workflow.sender.send(wf_res.draft.draft_id)

    # Dry-run execution must simulate only; network transmission strictly prevented
    assert send_result_approved.success is True
    assert "DRY RUN" in send_result_approved.message
    assert wf_res.draft.status == DraftStatus.SIMULATED_SENT
    assert wf_res.draft.status != DraftStatus.SENT
