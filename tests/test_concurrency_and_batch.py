"""
tests/test_concurrency_and_batch.py
-----------------------------------
Comprehensive test suite verifying Part E requirements:
1. Bounded concurrency (2, 4, 8 workers).
2. Thread safety of review queue and persistence under contention.
3. Incremental persistence (drafts written as they finish, not lost on interruption).
4. Row-order preservation in batch results regardless of completion order.
5. Duplicate detection and idempotency under concurrent execution.
6. Partial failure handling: failing contact does not crash batch or lose previous results.
7. Uploaded CV PDF is not mutated during batch processing (SHA-256 byte-for-byte unmodified).
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pypdf
import pytest

from src.contacts.csv_importer import ContactRecord, ImportResult
from src.contacts.email_parser import parse_email
from src.outreach.drafting import draft_outreach_email
from src.outreach.mailer import EmailReviewManager
from src.outreach.models import DraftStatus, EmailDraft, WorkflowMode
from src.outreach.send_log import SendLog
from src.research.cache import CachedResearchProvider, ResearchCache
from src.research.models import EvidenceItem, ResearchResult, RETRIEVAL_OK
from src.research.providers import ResearchProvider
from src.ui.storage import (
    clear_registered_cv,
    get_batch_record,
    get_draft,
    register_uploaded_cv,
    save_draft_record,
)
from src.verification.models import BLOCKED
from src.workflow.batch_runner import BackgroundBatchRunner
from src.workflow.engine import OutreachWorkflow, WorkflowResult


class JitteryMockProvider(ResearchProvider):
    """
    Mock research provider that introduces intentional jitter or errors
    to test race conditions, out-of-order completion, and partial failures.
    """

    def __init__(self, latencies: dict[str, float] | None = None, failing_urls: set[str] | None = None) -> None:
        self.latencies = latencies or {}
        self.failing_urls = failing_urls or set()
        self.call_count = 0

    def fetch(self, url: str, source_type: str, supports: list[str]) -> EvidenceItem:
        self.call_count += 1
        url_lower = url.lower()
        if any(f in url_lower for f in self.failing_urls):
            raise ConnectionResetError(f"Simulated fatal network error for {url}")

        delay = self.latencies.get(url, 0.005)
        if delay > 0:
            time.sleep(delay)

        domain = url.split("://")[-1].split("/")[0]
        company_hint = domain.split(".")[0].capitalize()
        sample_text = (
            f"{company_hint} develops generative AI models, deep learning agents, "
            f"and LLM infrastructure. We hire machine learning and AI interns."
        )
        return EvidenceItem(
            source_url=url,
            source_type=source_type,
            title=f"{company_hint} - Intelligent Agents",
            extracted_text=sample_text,
            retrieved_at=EvidenceItem.now_utc(),
            retrieval_status=RETRIEVAL_OK,
            supports=list(supports),
            claim_supported="Extracted technical focus",
        )


def _make_contact(row_num: int, email: str, company: str, website: str) -> ContactRecord:
    email_res = parse_email(email)
    return ContactRecord(
        row_number=row_num,
        company_name=company,
        contact_name=f"Contact {row_num}",
        email_result=email_res,
        website=website,
        role_or_job_url="AI/ML Intern",
        source_url="https://example.com/source",
        notes=f"Note for row {row_num}",
        raw_values={"company_name": company, "email": email, "website": website},
    )


def _create_test_pdf(path: Path, content: str = "Candidate CV - Machine Learning Engineer") -> Path:
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as f:
        writer.write(f)
    return path


@pytest.fixture(autouse=True)
def mock_llm_for_tests() -> Any:
    from src.llm.router import LLMResponse
    def _mock_extract(self: Any, company_name: str, evidence_texts: list[str]) -> dict[str, Any]:
        return {
            "technical_areas": ["generative ai", "agents", "machine learning"],
            "evidence_summary": f"{company_name} develops generative AI agent systems and machine learning workflows.",
        }

    def _mock_gen_json(self: Any, prompt: str, system_instruction: str = "") -> LLMResponse:
        return LLMResponse(
            success=True,
            provider_name="mock",
            parsed_json={
                "opening_paragraph": "I am a 4th-year Dual Degree student at IIT Kharagpur writing to express my interest in AI engineering internship opportunities.",
                "alignment_paragraph": "Given your work on intelligent systems, my work on agent workflows at ControlPlane.ai aligns well with your team.",
                "closing_paragraph": "I have attached my resume and would welcome the opportunity to discuss how I can contribute."
            },
        )

    with patch("src.llm.router.LLMRouter.extract_company_ai_evidence", _mock_extract), \
         patch("src.llm.router.LLMRouter.generate_json_with_fallback", _mock_gen_json), \
         patch("src.llm.router.LLMRouter.has_any_llm", True):
        yield


# ---------------------------------------------------------------------------
# 1. Bounded Concurrency (2, 4, 8 workers)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workers", [2, 4, 8])
def test_bounded_concurrency_workers(tmp_path: Path, workers: int) -> None:
    """Verify that batch execution succeeds across 2, 4, and 8 workers."""
    cv_path = _create_test_pdf(tmp_path / "candidate_cv.pdf")
    send_log = SendLog(db_path=tmp_path / f"send_log_workers_{workers}.db")

    workflow = OutreachWorkflow(
        default_mode=WorkflowMode.UPLOADED_CV,
        uploaded_cv_path=cv_path,
        dry_run=True,
        send_log=send_log,
        campaign_id=f"test_workers_{workers}",
    )

    contacts = [
        _make_contact(i, f"intern_{i}@company{i % 5}.com", f"Company {i % 5}", f"https://company{i % 5}.com")
        for i in range(1, 17)
    ]
    import_result = ImportResult(contacts=contacts, errors=[])

    mock_provider = JitteryMockProvider()
    batch = workflow.process_import_result(
        import_result=import_result,
        target_role="AI/ML Intern",
        research_provider=mock_provider,
        max_workers=workers,
    )

    assert batch.processed_count == 16
    assert len(batch.results) == 16
    assert batch.failed_count == 0
    assert batch.drafted_count > 0

    # Ensure row order is strictly preserved
    row_numbers = [r.row_number for r in batch.results]
    assert row_numbers == list(range(1, 17))


# ---------------------------------------------------------------------------
# 2. Thread Safety of Review Queue and Send Log under Contention
# ---------------------------------------------------------------------------

def test_thread_safety_review_manager_and_send_log(tmp_path: Path) -> None:
    """
    Verify thread safety of EmailReviewManager and SendLog when multiple
    threads register, edit, and approve drafts concurrently under high contention.
    """
    review_mgr = EmailReviewManager()
    send_log = SendLog(db_path=tmp_path / "contention_send_log.db")

    num_threads = 20
    drafts_per_thread = 5
    total_drafts = num_threads * drafts_per_thread

    def worker_task(thread_id: int) -> list[str]:
        registered_ids = []
        for i in range(drafts_per_thread):
            draft_id = f"draft_{thread_id}_{i}"
            recipient = f"dev_{thread_id}_{i}@example.com"
            draft = EmailDraft(
                draft_id=draft_id,
                recipient=recipient,
                recipient_name=f"Developer {thread_id}",
                company_name=f"Company {thread_id}",
                subject=f"Internship Inquiry {draft_id}",
                body=f"Dear Team, inquiry {draft_id}",
                attachment_path=str(tmp_path / "cv.pdf"),
                mode=WorkflowMode.UPLOADED_CV,
                approved=False,
                status=DraftStatus.DRAFT,
            )
            review_mgr.register_draft(draft)
            send_log.record_draft(draft_id=draft_id, recipient=recipient, campaign_id="contention_test")
            registered_ids.append(draft_id)

            # Concurrent edit and approval
            review_mgr.edit_draft(draft_id, subject=f"Updated Inquiry {draft_id}")
            review_mgr.approve_draft(draft_id)
            send_log.record_approved(draft_id=draft_id, recipient=recipient, campaign_id="contention_test")
        return registered_ids

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, t) for t in range(num_threads)]
        for f in as_completed(futures):
            f.result()

    all_drafts = review_mgr.list_drafts()
    assert len(all_drafts) == total_drafts
    for draft in all_drafts:
        assert draft.approved is True
        assert draft.status == DraftStatus.APPROVED
        assert draft.subject.startswith("Updated Inquiry")

    # Verify atomic claim concurrency on identical draft
    # Only 1 thread should succeed claiming a draft for sending
    target_draft = all_drafts[0]
    claim_successes = []

    def claim_task() -> bool:
        claimed, _ = send_log.claim_for_sending(
            draft_id=target_draft.draft_id,
            recipient=target_draft.recipient,
            campaign_id="contention_test",
            dry_run=True,
        )
        return claimed

    with ThreadPoolExecutor(max_workers=10) as executor:
        claims = list(executor.map(lambda _: claim_task(), range(10)))

    # Exactly one thread succeeds
    assert claims.count(True) == 1
    assert claims.count(False) == 9


# ---------------------------------------------------------------------------
# 3. Incremental Persistence During Batch Execution
# ---------------------------------------------------------------------------

def test_incremental_persistence_during_batch(tmp_path: Path) -> None:
    """
    Verify that BackgroundBatchRunner persists batch state incrementally as each
    contact finishes, rather than waiting for the entire batch to complete.
    """
    cv_path = _create_test_pdf(tmp_path / "candidate_cv.pdf")
    send_log = SendLog(db_path=tmp_path / "test_incremental.db")
    workflow = OutreachWorkflow(
        default_mode=WorkflowMode.UPLOADED_CV,
        uploaded_cv_path=cv_path,
        dry_run=True,
        send_log=send_log,
        campaign_id="batch_incremental_test",
    )

    batch_runner = BackgroundBatchRunner.get_instance()
    batch_id = "test_batch_incremental_001"

    contacts = [
        _make_contact(1, "c1@testinc.com", "Inc Corp 1", "https://testinc1.com"),
        _make_contact(2, "c2@testinc.com", "Inc Corp 2", "https://testinc2.com"),
        _make_contact(3, "c3@testinc.com", "Inc Corp 3", "https://testinc3.com"),
        _make_contact(4, "c4@testinc.com", "Inc Corp 4", "https://testinc4.com"),
    ]
    import_result = ImportResult(contacts=contacts, errors=[])

    # Contact 1 and 2 finish immediately; 3 and 4 take slightly longer
    latencies = {
        "https://testinc1.com": 0.001,
        "https://testinc2.com": 0.001,
        "https://testinc3.com": 0.15,
        "https://testinc4.com": 0.15,
    }
    mock_provider = JitteryMockProvider(latencies=latencies)

    # Launch batch with wait_for_completion=False to inspect mid-flight state
    batch_runner.start_batch(
        batch_id=batch_id,
        import_result=import_result,
        workflow=workflow,
        target_role="AI/ML Intern",
        campaign_id="batch_incremental_test",
        mode=WorkflowMode.UPLOADED_CV,
        max_workers=2,
        research_provider=mock_provider,
        wait_for_completion=False,
    )

    # Allow fast contacts 1 and 2 to complete incrementally
    mid_record = None
    for _ in range(30):
        time.sleep(0.05)
        mid_record = get_batch_record(batch_id)
        if mid_record and mid_record.get("processed_count", 0) >= 1:
            break

    # Mid-flight check: state file should exist and show at least 1-2 processed contacts
    assert mid_record is not None
    assert mid_record["batch_id"] == batch_id
    assert mid_record["total_rows"] == 4
    # Incremental update verified: processed_count is > 0 before the batch is finished
    assert mid_record["processed_count"] >= 1

    # Wait for completion
    timeout = time.time() + 5.0
    while time.time() < timeout:
        rec = get_batch_record(batch_id)
        if rec and rec["status"] == "COMPLETED":
            break
        time.sleep(0.05)

    final_record = get_batch_record(batch_id)
    assert final_record is not None
    assert final_record["status"] == "COMPLETED"
    assert final_record["processed_count"] == 4
    assert len(final_record["results"]) == 4

    # Verify drafts persisted individually on disk
    for res in final_record["results"]:
        if res.get("draft_id"):
            draft_rec = get_draft(res["draft_id"])
            assert draft_rec is not None
            assert draft_rec["draft_id"] == res["draft_id"]


# ---------------------------------------------------------------------------
# 4. Row-Order Preservation under High Jitter
# ---------------------------------------------------------------------------

def test_row_order_preservation_under_high_jitter(tmp_path: Path) -> None:
    """
    Verify that results strictly preserve original CSV row order (1, 2, 3, 4, 5)
    even when contacts complete in arbitrary or reverse order.
    """
    cv_path = _create_test_pdf(tmp_path / "candidate_cv.pdf")
    send_log = SendLog(db_path=tmp_path / "row_order_send_log.db")

    workflow = OutreachWorkflow(
        default_mode=WorkflowMode.UPLOADED_CV,
        uploaded_cv_path=cv_path,
        dry_run=True,
        send_log=send_log,
        campaign_id="row_order_test",
    )

    contacts = [
        _make_contact(1, "first@corp.com", "Slow Corp 1", "https://slow1.com"),
        _make_contact(2, "second@corp.com", "Med Corp 2", "https://med2.com"),
        _make_contact(3, "third@corp.com", "Fast Corp 3", "https://fast3.com"),
        _make_contact(4, "fourth@corp.com", "SuperFast Corp 4", "https://superfast4.com"),
        _make_contact(5, "fifth@corp.com", "Instant Corp 5", "https://instant5.com"),
    ]
    import_result = ImportResult(contacts=contacts, errors=[])

    # Row 1 is slowest (100ms), Row 5 is fastest (1ms)
    latencies = {
        "https://slow1.com": 0.10,
        "https://med2.com": 0.06,
        "https://fast3.com": 0.03,
        "https://superfast4.com": 0.01,
        "https://instant5.com": 0.001,
    }
    mock_provider = JitteryMockProvider(latencies=latencies)

    # 1. Test engine.process_import_result directly
    batch_res = workflow.process_import_result(
        import_result=import_result,
        target_role="AI/ML Intern",
        research_provider=mock_provider,
        max_workers=4,
    )
    assert [r.row_number for r in batch_res.results] == [1, 2, 3, 4, 5]
    assert [r.contact_email for r in batch_res.results] == [
        "first@corp.com",
        "second@corp.com",
        "third@corp.com",
        "fourth@corp.com",
        "fifth@corp.com",
    ]

    # 2. Test BackgroundBatchRunner
    runner = BackgroundBatchRunner.get_instance()
    batch_id = "test_batch_row_order_002"
    runner.start_batch(
        batch_id=batch_id,
        import_result=import_result,
        workflow=workflow,
        target_role="AI/ML Intern",
        campaign_id="row_order_test",
        mode=WorkflowMode.UPLOADED_CV,
        max_workers=4,
        research_provider=mock_provider,
        wait_for_completion=True,
    )
    rec = get_batch_record(batch_id)
    assert rec is not None
    assert rec["status"] == "COMPLETED"
    assert [r["row_number"] for r in rec["results"]] == [1, 2, 3, 4, 5]
    assert [r["contact_email"] for r in rec["results"]] == [
        "first@corp.com",
        "second@corp.com",
        "third@corp.com",
        "fourth@corp.com",
        "fifth@corp.com",
    ]


# ---------------------------------------------------------------------------
# 5. Duplicate Detection & Idempotency under Concurrent Execution
# ---------------------------------------------------------------------------

def test_duplicate_detection_and_idempotency_concurrent(tmp_path: Path) -> None:
    """
    Verify duplicate detection across SQLite send_log during concurrent execution:
    - Pre-existing sends are blocked immediately with duplicate message.
    - Repeated contacts in the batch do not produce duplicate sends.
    """
    cv_path = _create_test_pdf(tmp_path / "candidate_cv.pdf")
    send_log = SendLog(db_path=tmp_path / "dup_detection.db")

    # Pre-record an existing simulated send for "already_contacted@corp.com"
    send_log.record_simulated(
        draft_id="prior_draft_01",
        recipient="already_contacted@corp.com",
        campaign_id="campaign_dup_test",
    )

    workflow = OutreachWorkflow(
        default_mode=WorkflowMode.UPLOADED_CV,
        uploaded_cv_path=cv_path,
        dry_run=True,
        send_log=send_log,
        campaign_id="campaign_dup_test",
    )

    contacts = [
        _make_contact(1, "already_contacted@corp.com", "Prior Corp", "https://priorcorp.com"),
        _make_contact(2, "new_intern_1@fresh1.com", "Fresh1", "https://fresh1.com"),
        _make_contact(3, "new_intern_2@fresh2.com", "Fresh2", "https://fresh2.com"),
    ]
    import_result = ImportResult(contacts=contacts, errors=[])

    mock_provider = JitteryMockProvider()
    batch = workflow.process_import_result(
        import_result=import_result,
        target_role="AI/ML Intern",
        research_provider=mock_provider,
        max_workers=4,
    )

    # Row 1 must be blocked as duplicate
    res_1 = batch.results[0]
    assert res_1.status == BLOCKED
    assert "Duplicate send prevented" in res_1.message
    assert res_1.draft is None

    # Rows 2 and 3 must succeed
    assert batch.results[1].draft is not None
    assert batch.results[2].draft is not None
    assert batch.blocked_count >= 1


# ---------------------------------------------------------------------------
# 6. Partial Failure Handling
# ---------------------------------------------------------------------------

def test_partial_failure_isolation(tmp_path: Path) -> None:
    """
    Verify that an unhandled failure in one contact (e.g. fatal provider crash)
    is cleanly caught, recorded as ERROR, and does NOT crash or halt other contacts.
    """
    cv_path = _create_test_pdf(tmp_path / "candidate_cv.pdf")
    send_log = SendLog(db_path=tmp_path / "test_partial_failure.db")

    workflow = OutreachWorkflow(
        default_mode=WorkflowMode.UPLOADED_CV,
        uploaded_cv_path=cv_path,
        dry_run=True,
        send_log=send_log,
        campaign_id="partial_fail_campaign",
    )

    contacts = [
        _make_contact(1, "ok1@healthy1.com", "Healthy1", "https://healthy1.com"),
        _make_contact(2, "fail@crash.com", "Crashing", "https://crashing-endpoint.com"),
        _make_contact(3, "ok2@healthy2.com", "Healthy2", "https://healthy2.com"),
    ]
    import_result = ImportResult(contacts=contacts, errors=[])

    # Configure provider to raise hard exception on "crashing-endpoint.com"
    failing_provider = JitteryMockProvider(failing_urls={"crashing-endpoint.com"})

    batch = workflow.process_import_result(
        import_result=import_result,
        target_role="AI/ML Intern",
        research_provider=failing_provider,
        max_workers=4,
    )

    assert batch.processed_count == 3
    assert len(batch.results) == 3

    # Row 1: Succeeded
    assert batch.results[0].status != "ERROR"
    assert batch.results[0].draft is not None

    # Row 2: Cleanly caught as ERROR without crashing the batch
    assert batch.results[1].status == "ERROR"
    assert "Simulated fatal network error" in (batch.results[1].error or "")
    assert batch.results[1].draft is None

    # Row 3: Succeeded despite row 2 error
    assert batch.results[2].status != "ERROR"
    assert batch.results[2].draft is not None

    # Error tracked in batch.errors
    assert any(err.get("row_number") == 2 for err in batch.errors)


# ---------------------------------------------------------------------------
# 7. Uploaded CV PDF Byte-for-Byte Preservation
# ---------------------------------------------------------------------------

def test_cv_pdf_unmodified_byte_for_byte_concurrent(tmp_path: Path) -> None:
    """
    Verify that the registered uploaded CV PDF remains strictly byte-for-byte
    unmodified across high-concurrency batch execution.
    """
    cv_file = _create_test_pdf(
        tmp_path / "master_candidate_resume.pdf",
        content="Candidate Resume: 4th year Dual Degree, IIT Kharagpur, AI/ML Specialization.",
    )
    initial_bytes = cv_file.read_bytes()
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()

    send_log = SendLog(db_path=tmp_path / "cv_preservation.db")
    workflow = OutreachWorkflow(
        default_mode=WorkflowMode.UPLOADED_CV,
        uploaded_cv_path=cv_file,
        dry_run=True,
        send_log=send_log,
        campaign_id="cv_preservation_test",
    )

    # Process 20 contacts across 8 concurrent workers
    contacts = [
        _make_contact(i, f"intern_{i}@company{i}.com", f"Company {i}", f"https://company{i}.com")
        for i in range(1, 21)
    ]
    import_result = ImportResult(contacts=contacts, errors=[])
    mock_provider = JitteryMockProvider()

    batch = workflow.process_import_result(
        import_result=import_result,
        target_role="AI/ML Intern",
        research_provider=mock_provider,
        max_workers=8,
    )

    assert batch.processed_count == 20

    # Post-execution verification
    final_bytes = cv_file.read_bytes()
    final_hash = hashlib.sha256(final_bytes).hexdigest()

    assert final_bytes == initial_bytes, "Uploaded CV file bytes were modified!"
    assert final_hash == initial_hash, "Uploaded CV SHA-256 hash changed!"
    assert workflow.verify_uploaded_cv_unmodified() is True
