"""
tests/test_send_log.py
----------------------
Tests for the persistent, SQLite-backed send log (src/outreach/send_log.py).
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from src.outreach.send_log import (
    SendLog,
    STATUS_DRAFT,
    STATUS_APPROVED,
    STATUS_SUCCESS,
    STATUS_SIMULATED,
    STATUS_FAILED,
    STATUS_UNCERTAIN,
    STATUS_SENDING,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_send_log.sqlite3"


@pytest.fixture
def send_log(temp_db: Path) -> SendLog:
    return SendLog(db_path=temp_db)


class TestSendLogInitialization:
    def test_creates_db_file_and_table(self, temp_db: Path):
        assert not temp_db.exists()
        log = SendLog(db_path=temp_db)
        assert temp_db.exists()
        assert len(log.list_all()) == 0

    def test_table_has_expected_schema(self, temp_db: Path, send_log: SendLog):
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(send_log)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {
            "id", "draft_id", "recipient", "campaign_id",
            "status", "created_at", "updated_at",
        }
        assert expected.issubset(cols)


class TestSendLogStateTransitions:
    def test_record_draft(self, send_log: SendLog):
        send_log.record_draft("draft-001", "test@example.com", campaign_id="camp-1")
        assert send_log.get_status("draft-001") == STATUS_DRAFT
        entries = send_log.list_all()
        assert len(entries) == 1
        assert entries[0]["recipient"] == "test@example.com"
        assert entries[0]["campaign_id"] == "camp-1"

    def test_record_approval(self, send_log: SendLog):
        send_log.record_draft("draft-002", "alice@example.com")
        send_log.record_approved("draft-002", "alice@example.com")
        assert send_log.get_status("draft-002") == STATUS_APPROVED

    def test_record_success(self, send_log: SendLog):
        send_log.record_draft("draft-003", "bob@example.com")
        send_log.record_approved("draft-003", "bob@example.com")
        send_log.record_success("draft-003", "bob@example.com")
        assert send_log.get_status("draft-003") == STATUS_SUCCESS
        assert send_log.was_successfully_sent("bob@example.com")

    def test_record_failure(self, send_log: SendLog):
        send_log.record_draft("draft-004", "charlie@example.com")
        send_log.record_failed("draft-004", "charlie@example.com")
        assert send_log.get_status("draft-004") == STATUS_FAILED
        assert not send_log.was_successfully_sent("charlie@example.com")

    def test_record_uncertain(self, send_log: SendLog):
        send_log.record_draft("draft-005", "dave@example.com")
        send_log.record_uncertain("draft-005", "dave@example.com")
        assert send_log.get_status("draft-005") == STATUS_UNCERTAIN
        assert send_log.has_uncertain_send("dave@example.com")
        assert not send_log.was_successfully_sent("dave@example.com")

    def test_record_simulated(self, send_log: SendLog):
        send_log.record_draft("draft-006", "sim@example.com")
        send_log.record_simulated("draft-006", "sim@example.com", campaign_id="camp-dry")
        assert send_log.get_status("draft-006") == STATUS_SIMULATED
        assert send_log.was_simulated("sim@example.com", campaign_id="camp-dry")
        # Simulated send must NOT be treated as confirmed live send
        assert not send_log.was_successfully_sent("sim@example.com", campaign_id="camp-dry")
        # But has_any_send detects both
        assert send_log.has_any_send("sim@example.com", campaign_id="camp-dry")


class TestDuplicatePrevention:
    def test_not_sent_initially(self, send_log: SendLog):
        assert not send_log.was_successfully_sent("target@example.com")
        assert not send_log.has_uncertain_send("target@example.com")

    def test_draft_or_approved_does_not_count_as_sent(self, send_log: SendLog):
        send_log.record_draft("d-1", "target@example.com")
        assert not send_log.was_successfully_sent("target@example.com")
        send_log.record_approved("d-1", "target@example.com")
        assert not send_log.was_successfully_sent("target@example.com")

    def test_sent_success_blocks_duplicate(self, send_log: SendLog):
        send_log.record_draft("d-2", "Target@Example.COM")
        send_log.record_success("d-2", "Target@Example.COM")
        # Case insensitive check
        assert send_log.was_successfully_sent("target@example.com")
        assert send_log.was_successfully_sent("TARGET@EXAMPLE.COM")

    def test_campaign_scoped_duplicate_check(self, send_log: SendLog):
        send_log.record_draft("d-3", "user@corp.com", campaign_id="camp-A")
        send_log.record_success("d-3", "user@corp.com", campaign_id="camp-A")
        assert send_log.was_successfully_sent("user@corp.com", campaign_id="camp-A")
        assert not send_log.was_successfully_sent("user@corp.com", campaign_id="camp-B")

    def test_uncertain_state_tracked_separately(self, send_log: SendLog):
        send_log.record_draft("d-4", "risk@corp.com")
        send_log.record_uncertain("d-4", "risk@corp.com")
        assert send_log.has_uncertain_send("risk@corp.com")
        assert not send_log.was_successfully_sent("risk@corp.com")


class TestPersistenceAcrossInstances:
    def test_data_persists_across_new_instances(self, temp_db: Path):
        log1 = SendLog(db_path=temp_db)
        log1.record_draft("d-persist", "persist@example.com")
        log1.record_success("d-persist", "persist@example.com")

        # Create separate instance reading from same SQLite file
        log2 = SendLog(db_path=temp_db)
        assert len(log2.list_all()) == 1
        assert log2.was_successfully_sent("persist@example.com")
        assert log2.get_status("d-persist") == STATUS_SUCCESS


class TestConcurrencyThreadSafety:
    def test_concurrent_writes_do_not_corrupt_database(self, temp_db: Path):
        log = SendLog(db_path=temp_db)
        errors: list[Exception] = []

        def worker(worker_id: int):
            try:
                for i in range(10):
                    did = f"w{worker_id}-{i}"
                    email = f"user{worker_id}_{i}@concurrent.com"
                    log.record_draft(did, email)
                    log.record_approved(did, email)
                    log.record_success(did, email)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(log.list_all()) == 40


class TestDispatchReservationAndConcurrency:
    def test_single_thread_claim_and_release_success(self, send_log: SendLog):
        """Standard lifecycle: claim -> send succeeds -> record_success."""
        claimed, reason = send_log.claim_for_sending("d-claim-1", "user@test.com", "camp-1", dry_run=False)
        assert claimed is True
        assert reason == "CLAIMED"
        assert send_log.get_status("d-claim-1") == STATUS_SENDING

        # Re-claiming while sending must be rejected
        c2, r2 = send_log.claim_for_sending("d-claim-1", "user@test.com", "camp-1", dry_run=False)
        assert c2 is False
        assert r2 == "SEND_IN_PROGRESS"

        # Record success
        send_log.record_success("d-claim-1", "user@test.com", "camp-1")
        assert send_log.get_status("d-claim-1") == STATUS_SUCCESS
        assert send_log.was_successfully_sent("user@test.com", "camp-1")

        # Claiming after success must be rejected as DUPLICATE_RECIPIENT
        c3, r3 = send_log.claim_for_sending("d-claim-1", "user@test.com", "camp-1", dry_run=False)
        assert c3 is False
        assert r3 == "DUPLICATE_RECIPIENT"

    def test_concurrent_claims_only_one_succeeds(self, send_log: SendLog):
        """Two concurrent threads claiming the exact same recipient; only one succeeds."""
        results: list[tuple[bool, str]] = []
        barrier = threading.Barrier(2)

        def attempt_claim(worker_id: int):
            barrier.wait()
            res = send_log.claim_for_sending(f"draft-{worker_id}", "contested@test.com", "camp-race", dry_run=False)
            results.append(res)

        t1 = threading.Thread(target=attempt_claim, args=(1,))
        t2 = threading.Thread(target=attempt_claim, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        successes = [r for r in results if r[0] is True]
        failures = [r for r in results if r[0] is False]
        assert len(successes) == 1
        assert len(failures) == 1
        assert successes[0][1] == "CLAIMED"
        assert failures[0][1] in ("SEND_IN_PROGRESS", "ALREADY_SENT", "DUPLICATE_RECIPIENT")

    def test_claim_release_on_failure_allows_reclaim(self, send_log: SendLog):
        """If sending fails, release_claim(STATUS_FAILED) clears SENDING and allows retry."""
        claimed, _ = send_log.claim_for_sending("d-fail", "fail@test.com", "camp-retry", dry_run=False)
        assert claimed is True
        assert send_log.get_status("d-fail") == STATUS_SENDING

        # Provider fails
        send_log.release_claim("d-fail", STATUS_FAILED)
        assert send_log.get_status("d-fail") == STATUS_FAILED
        assert not send_log.was_successfully_sent("fail@test.com", "camp-retry")

        # Can now be claimed again for retry
        reclaimed, r_reason = send_log.claim_for_sending("d-fail", "fail@test.com", "camp-retry", dry_run=False)
        assert reclaimed is True
        assert r_reason == "CLAIMED"

    def test_claim_release_on_uncertain_blocks_further_live_sends(self, send_log: SendLog):
        """If send outcome is uncertain, release_claim(STATUS_UNCERTAIN) blocks live sends to avoid duplicates."""
        claimed, _ = send_log.claim_for_sending("d-unc", "uncertain@test.com", "camp-unc", dry_run=False)
        assert claimed is True

        send_log.release_claim("d-unc", STATUS_UNCERTAIN)
        assert send_log.get_status("d-unc") == STATUS_UNCERTAIN
        assert send_log.has_uncertain_send("uncertain@test.com", "camp-unc")

        # Future live claims are blocked because status is UNCERTAIN
        c_next, r_next = send_log.claim_for_sending("d-unc-2", "uncertain@test.com", "camp-unc", dry_run=False)
        assert c_next is False
        assert r_next == "UNCERTAIN_SEND"

    def test_dry_run_claim_does_not_block_simulations(self, send_log: SendLog):
        """A simulated send must not be treated as a confirmed live send, and must not block a subsequent live send."""
        c1, _ = send_log.claim_for_sending("d-sim-1", "dry@test.com", "camp-dry", dry_run=True)
        assert c1 is True
        send_log.record_simulated("d-sim-1", "dry@test.com", "camp-dry")
        assert send_log.was_simulated("dry@test.com", "camp-dry")
        # Critical rule: Simulated send must NEVER be counted as a successful live send
        assert not send_log.was_successfully_sent("dry@test.com", "camp-dry")

        # A future live send claim (dry_run=False) for a different draft to this recipient is NOT blocked by simulation
        c_live, r_live = send_log.claim_for_sending("d-live-1", "dry@test.com", "camp-dry", dry_run=False)
        assert c_live is True
        assert r_live == "CLAIMED"

    def test_changed_recipient_updates_send_log_record(self, send_log: SendLog):
        """When recipient is edited on a draft, upsert updates the recipient and allows the old one to be free."""
        send_log.record_draft("d-edit", "original@corp.com", "camp-edit")
        assert send_log.get_status("d-edit") == STATUS_DRAFT

        # Edit recipient
        send_log.record_draft("d-edit", "updated@corp.com", "camp-edit")
        entries = send_log.list_all()
        assert len(entries) == 1
        assert entries[0]["recipient"] == "updated@corp.com"

