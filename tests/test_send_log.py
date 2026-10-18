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
