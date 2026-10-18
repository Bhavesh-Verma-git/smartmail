"""
src/outreach/send_log.py
------------------------
Persistent, SQLite-backed send log for duplicate-send prevention.

Policy
------
- A send is recorded only after the provider confirms success.
- A timed-out or exception-bearing send is stored as UNCERTAIN, not SUCCESS.
- Retry of an UNCERTAIN send requires explicit user action.
- DRAFT and APPROVED records are tracked for audit; only SUCCESS blocks resend.
- All operations run inside a transaction to prevent race conditions.
- No credentials, tokens, or full email body text are persisted.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_DDL = """
CREATE TABLE IF NOT EXISTS send_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id        TEXT    NOT NULL,
    recipient       TEXT    NOT NULL,
    campaign_id     TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL,       -- DRAFT | APPROVED | SUCCESS | FAILED | UNCERTAIN
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_recipient_campaign
    ON send_log (recipient, campaign_id)
    WHERE status = 'SUCCESS';

CREATE INDEX IF NOT EXISTS idx_recipient ON send_log (recipient);
CREATE INDEX IF NOT EXISTS idx_draft_id  ON send_log (draft_id);
"""

# Status constants
STATUS_DRAFT = "DRAFT"
STATUS_APPROVED = "APPROVED"
STATUS_SENDING = "SENDING"
STATUS_SUCCESS = "SUCCESS"
STATUS_SIMULATED = "SIMULATED"
STATUS_FAILED = "FAILED"
STATUS_UNCERTAIN = "UNCERTAIN"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SendLog:
    """
    Thread-safe, SQLite-backed registry of outreach send attempts.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Pass ``":memory:"`` for an
        in-memory database (useful in tests).
    """

    def __init__(self, db_path: str | Path = "data/send_log.db") -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        # For ':memory:' databases each sqlite3.connect() call creates a fresh
        # empty DB, so we keep a single shared connection alive for the
        # lifetime of this instance.
        self._shared_conn: sqlite3.Connection | None = None
        if self._db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.execute("PRAGMA foreign_keys=ON;")
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        if self._shared_conn is not None:
            # In-memory mode: reuse the single persistent connection.
            # No close() — connection lives for the lifetime of the instance.
            try:
                yield self._shared_conn
                self._shared_conn.commit()
            except Exception:
                self._shared_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Write operations (all serialised via threading.Lock)
    # ------------------------------------------------------------------

    def record_draft(self, draft_id: str, recipient: str, campaign_id: str = "") -> None:
        """Record that a draft was created."""
        self._upsert(draft_id, recipient, campaign_id, STATUS_DRAFT)

    def record_approved(self, draft_id: str, recipient: str, campaign_id: str = "") -> None:
        """Record that a draft was explicitly approved."""
        self._upsert(draft_id, recipient, campaign_id, STATUS_APPROVED)

    def record_success(self, draft_id: str, recipient: str, campaign_id: str = "") -> None:
        """Record a confirmed successful send."""
        self._upsert(draft_id, recipient, campaign_id, STATUS_SUCCESS)

    def record_simulated(self, draft_id: str, recipient: str, campaign_id: str = "") -> None:
        """Record a simulated send (dry run — no actual transmission)."""
        self._upsert(draft_id, recipient, campaign_id, STATUS_SIMULATED)

    def record_failed(self, draft_id: str, recipient: str, campaign_id: str = "") -> None:
        """Record a definitively failed send (provider rejected it)."""
        self._upsert(draft_id, recipient, campaign_id, STATUS_FAILED)

    def record_uncertain(self, draft_id: str, recipient: str, campaign_id: str = "") -> None:
        """
        Record a send attempt whose outcome is unknown (e.g. network timeout).
        Must NOT be treated as SUCCESS until reconciled manually.
        """
        self._upsert(draft_id, recipient, campaign_id, STATUS_UNCERTAIN)

    def claim_for_sending(
        self,
        draft_id: str,
        recipient: str,
        campaign_id: str = "",
        dry_run: bool = True,
    ) -> tuple[bool, str]:
        """
        Atomically reserve a draft and recipient for sending.
        Prevents concurrent requests from dispatching the same recipient/campaign.
        Returns (True, "CLAIMED") on success, or (False, reason) if blocked.
        """
        now = _now_iso()
        recipient_norm = recipient.strip().lower()

        with self._lock:
            with self._connect() as conn:
                # 1. Duplicate check: Confirmed live send exists
                row_success = conn.execute(
                    """
                    SELECT id FROM send_log
                    WHERE recipient = ? AND campaign_id = ? AND status = ?
                    LIMIT 1
                    """,
                    (recipient_norm, campaign_id, STATUS_SUCCESS),
                ).fetchone()
                if row_success:
                    return False, "DUPLICATE_RECIPIENT"

                # 2. Simulated duplicate check (only in dry-run mode)
                if dry_run:
                    row_sim = conn.execute(
                        """
                        SELECT id FROM send_log
                        WHERE recipient = ? AND campaign_id = ? AND status = ?
                        LIMIT 1
                        """,
                        (recipient_norm, campaign_id, STATUS_SIMULATED),
                    ).fetchone()
                    if row_sim:
                        return False, "DUPLICATE_RECIPIENT"

                # 3. Check for unresolved uncertain send
                row_unc = conn.execute(
                    """
                    SELECT id FROM send_log
                    WHERE recipient = ? AND campaign_id = ? AND status = ?
                    LIMIT 1
                    """,
                    (recipient_norm, campaign_id, STATUS_UNCERTAIN),
                ).fetchone()
                if row_unc:
                    return False, "UNCERTAIN_SEND"

                # 4. Check for active concurrent in-flight claim
                row_sending = conn.execute(
                    """
                    SELECT id, draft_id FROM send_log
                    WHERE recipient = ? AND campaign_id = ? AND status = ?
                    LIMIT 1
                    """,
                    (recipient_norm, campaign_id, STATUS_SENDING),
                ).fetchone()
                if row_sending:
                    return False, "SEND_IN_PROGRESS"

                # 5. Acquire reservation/claim
                row_draft = conn.execute(
                    "SELECT id FROM send_log WHERE draft_id = ?", (draft_id,)
                ).fetchone()
                if row_draft:
                    conn.execute(
                        "UPDATE send_log SET status = ?, recipient = ?, campaign_id = ?, updated_at = ? WHERE draft_id = ?",
                        (STATUS_SENDING, recipient_norm, campaign_id, now, draft_id),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO send_log
                            (draft_id, recipient, campaign_id, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (draft_id, recipient_norm, campaign_id, STATUS_SENDING, now, now),
                    )
                return True, "CLAIMED"

    def release_claim(
        self,
        draft_id: str,
        status: str = STATUS_APPROVED,
    ) -> None:
        """Release an in-flight send reservation back to approved (or draft)."""
        now = _now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE send_log SET status = ?, updated_at = ? WHERE draft_id = ? AND status = ?",
                    (status, now, draft_id, STATUS_SENDING),
                )

    def _upsert(
        self,
        draft_id: str,
        recipient: str,
        campaign_id: str,
        status: str,
    ) -> None:
        now = _now_iso()
        recipient_norm = recipient.strip().lower()
        with self._lock:
            with self._connect() as conn:
                # Check if row already exists for this draft_id
                row = conn.execute(
                    "SELECT id FROM send_log WHERE draft_id = ?", (draft_id,)
                ).fetchone()
                if row:
                    if campaign_id:
                        conn.execute(
                            "UPDATE send_log SET status=?, recipient=?, updated_at=?, campaign_id=? WHERE draft_id=?",
                            (status, recipient_norm, now, campaign_id, draft_id),
                        )
                    else:
                        conn.execute(
                            "UPDATE send_log SET status=?, recipient=?, updated_at=? WHERE draft_id=?",
                            (status, recipient_norm, now, draft_id),
                        )
                else:
                    conn.execute(
                        """
                        INSERT INTO send_log
                            (draft_id, recipient, campaign_id, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (draft_id, recipient_norm, campaign_id, status, now, now),
                    )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def was_successfully_sent(self, recipient: str, campaign_id: str = "") -> bool:
        """
        Return True if a successful send to this recipient (and optional campaign)
        has been recorded.  This is the primary duplicate-prevention check.
        """
        recipient_norm = recipient.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM send_log
                WHERE recipient = ? AND campaign_id = ? AND status = ?
                LIMIT 1
                """,
                (recipient_norm, campaign_id, STATUS_SUCCESS),
            ).fetchone()
        return row is not None

    def was_simulated(self, recipient: str, campaign_id: str = "") -> bool:
        """
        Return True if a simulated send to this recipient (and optional campaign)
        has been recorded.
        """
        recipient_norm = recipient.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM send_log
                WHERE recipient = ? AND campaign_id = ? AND status = ?
                LIMIT 1
                """,
                (recipient_norm, campaign_id, STATUS_SIMULATED),
            ).fetchone()
        return row is not None

    def has_any_send(self, recipient: str, campaign_id: str = "") -> bool:
        """Return True if either a successful live send or a simulated send was recorded."""
        recipient_norm = recipient.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM send_log
                WHERE recipient = ? AND campaign_id = ? AND status IN (?, ?)
                LIMIT 1
                """,
                (recipient_norm, campaign_id, STATUS_SUCCESS, STATUS_SIMULATED),
            ).fetchone()
        return row is not None

    def has_uncertain_send(self, recipient: str, campaign_id: str = "") -> bool:
        """Return True if there is an unreconciled uncertain send to this recipient."""
        recipient_norm = recipient.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM send_log
                WHERE recipient = ? AND campaign_id = ? AND status = ?
                LIMIT 1
                """,
                (recipient_norm, campaign_id, STATUS_UNCERTAIN),
            ).fetchone()
        return row is not None

    def get_status(self, draft_id: str) -> str | None:
        """Return the current status for a specific draft_id, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM send_log WHERE draft_id = ?", (draft_id,)
            ).fetchone()
        return row["status"] if row else None

    def list_all(self, limit: int = 200) -> list[dict]:
        """Return recent send log entries (no sensitive body content)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT draft_id, recipient, campaign_id, status, created_at, updated_at
                FROM send_log
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_for_testing(self) -> None:
        """Drop and recreate tables. Only safe for in-memory or test databases."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DROP TABLE IF EXISTS send_log;")
            self._init_db()
