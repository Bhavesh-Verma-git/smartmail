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
    status          TEXT    NOT NULL,       -- DRAFT | APPROVED | SUCCESS | FAILED | UNCERTAIN | SENDING
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_recipient_campaign
    ON send_log (recipient, campaign_id)
    WHERE status = 'SUCCESS';

CREATE UNIQUE INDEX IF NOT EXISTS uq_recipient_campaign_sending
    ON send_log (recipient, campaign_id)
    WHERE status = 'SENDING';

CREATE INDEX IF NOT EXISTS idx_recipient ON send_log (recipient);
CREATE INDEX IF NOT EXISTS idx_draft_id  ON send_log (draft_id);

-- Persistent opt-out / suppression list. Checked before every send
-- (see claim_for_sending). A suppressed recipient is never sent to,
-- in dry-run or live mode, until explicitly un-suppressed by the operator.
CREATE TABLE IF NOT EXISTS suppression (
    recipient     TEXT PRIMARY KEY,
    reason        TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT 'manual',
    suppressed_at TEXT NOT NULL
);
"""

# Status constants
STATUS_DRAFT = "DRAFT"
STATUS_APPROVED = "APPROVED"
STATUS_SENDING = "SENDING"
STATUS_SUCCESS = "SUCCESS"
STATUS_SIMULATED = "SIMULATED"
STATUS_FAILED = "FAILED"
STATUS_UNCERTAIN = "UNCERTAIN"
STATUS_REJECTED = "REJECTED"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SendLog:
    """
    Thread-safe and cross-instance, SQLite-backed registry of outreach send attempts.

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
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
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
            try:
                yield self._shared_conn
            except Exception:
                raise
        else:
            conn = sqlite3.connect(self._db_path, timeout=15.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA busy_timeout=15000;")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    @contextmanager
    def _immediate_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Acquires an immediate write transaction (BEGIN IMMEDIATE) on an autocommit connection.
        Ensures cross-instance, cross-connection, and multi-thread serialization at the SQLite lock level.
        """
        if self._shared_conn is not None:
            with self._lock:
                self._shared_conn.execute("BEGIN IMMEDIATE;")
                try:
                    yield self._shared_conn
                    self._shared_conn.execute("COMMIT;")
                except Exception:
                    try:
                        self._shared_conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                    raise
        else:
            conn = sqlite3.connect(
                self._db_path,
                timeout=15.0,
                check_same_thread=False,
                isolation_level=None,  # Manual transaction management with explicit BEGIN IMMEDIATE
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA busy_timeout=15000;")
            conn.execute("BEGIN IMMEDIATE;")
            try:
                yield conn
                conn.execute("COMMIT;")
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
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

    def record_rejected(self, draft_id: str, recipient: str, campaign_id: str = "") -> None:
        """
        Record that a draft was rejected or placed on hold by the reviewer.
        Clears previous approval in SQLite send log.
        """
        self._upsert(draft_id, recipient, campaign_id, STATUS_REJECTED)

    def claim_for_sending(
        self,
        draft_id: str,
        recipient: str,
        campaign_id: str = "",
        dry_run: bool = True,
    ) -> tuple[bool, str]:
        """
        Atomically reserve a draft and recipient for sending across independent
        SQLite connections, instances, and worker threads.
        Prevents concurrent requests from dispatching the same recipient/campaign.
        Returns (True, "CLAIMED") on success, or (False, reason) if blocked.
        """
        now = _now_iso()
        recipient_norm = recipient.strip().lower()

        try:
            with self._immediate_conn() as conn:
                # 0. Suppression check: opt-out always wins, before any claim.
                row_sup = conn.execute(
                    "SELECT recipient FROM suppression WHERE recipient = ? LIMIT 1",
                    (recipient_norm,),
                ).fetchone()
                if row_sup:
                    return False, "SUPPRESSED"

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

                # 4. Check for active concurrent in-flight claim for this recipient/campaign
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

                # 5. Check if THIS draft already has an in-flight or terminal status
                row_draft = conn.execute(
                    "SELECT id, status FROM send_log WHERE draft_id = ? LIMIT 1", (draft_id,)
                ).fetchone()
                if row_draft:
                    d_status = row_draft["status"]
                    if d_status == STATUS_SENDING:
                        return False, "SEND_IN_PROGRESS"
                    if d_status == STATUS_SUCCESS:
                        return False, "ALREADY_SENT"
                    if d_status == STATUS_UNCERTAIN:
                        return False, "UNCERTAIN_SEND"
                    if dry_run and d_status == STATUS_SIMULATED:
                        return False, "ALREADY_SENT"

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
        except sqlite3.IntegrityError:
            # Caught DB-level unique constraint violation (e.g. uq_recipient_campaign_sending)
            return False, "SEND_IN_PROGRESS"
        except sqlite3.OperationalError:
            # Contention / lock busy timeout
            return False, "CONTENTION_ERROR"

    def release_claim(
        self,
        draft_id: str,
        status: str = STATUS_APPROVED,
    ) -> None:
        """Release an in-flight send reservation back to approved (or draft/failed)."""
        now = _now_iso()
        try:
            with self._immediate_conn() as conn:
                conn.execute(
                    "UPDATE send_log SET status = ?, updated_at = ? WHERE draft_id = ? AND status = ?",
                    (status, now, draft_id, STATUS_SENDING),
                )
        except Exception:
            pass

    def _upsert(
        self,
        draft_id: str,
        recipient: str,
        campaign_id: str,
        status: str,
    ) -> None:
        now = _now_iso()
        recipient_norm = recipient.strip().lower()
        # Terminal real-send states. SIMULATED is deliberately NOT in
        # this set: a dry-run sent nothing, so resetting a simulated draft
        # to DRAFT on edit is safe (and an existing test pins it). SUCCESS
        # (a real send happened), UNCERTAIN (outcome unknown) and SENDING
        # (in-flight claim) must never be erased by an edit.
        _TERMINAL = {"SUCCESS", "UNCERTAIN", "SENDING"}
        with self._immediate_conn() as conn:
            # Check if row already exists for this draft_id
            row = conn.execute(
                "SELECT id, status FROM send_log WHERE draft_id = ?", (draft_id,)
            ).fetchone()
            if row and row["status"] in _TERMINAL and status in ("DRAFT", "APPROVED"):
                # Never downgrade a terminal/send-state record back to
                # DRAFT/APPROVED: editing a sent/simulated/uncertain draft
                # must not erase its send history (previously _upsert
                # overwrote unconditionally, contradicting
                # remove_draft_record which preserves SUCCESS).
                return
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

    # ------------------------------------------------------------------
    # Suppression / opt-out list (persistent, checked before every send)
    # ------------------------------------------------------------------

    def suppress(self, recipient: str, reason: str = "", source: str = "manual") -> None:
        """Add a recipient to the persistent opt-out list (idempotent)."""
        recipient_norm = recipient.strip().lower()
        if not recipient_norm:
            return
        with self._immediate_conn() as conn:
            conn.execute(
                """INSERT INTO suppression (recipient, reason, source, suppressed_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(recipient) DO UPDATE SET reason=excluded.reason,
                       source=excluded.source, suppressed_at=excluded.suppressed_at""",
                (recipient_norm, reason, source, _now_iso()),
            )

    def unsuppress(self, recipient: str) -> bool:
        """Remove a recipient from the opt-out list. Returns True if removed."""
        recipient_norm = recipient.strip().lower()
        with self._immediate_conn() as conn:
            cur = conn.execute("DELETE FROM suppression WHERE recipient = ?", (recipient_norm,))
            return cur.rowcount > 0

    def is_suppressed(self, recipient: str) -> bool:
        recipient_norm = recipient.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT recipient FROM suppression WHERE recipient = ? LIMIT 1",
                (recipient_norm,),
            ).fetchone()
        return row is not None

    def list_suppressed(self, limit: int = 500) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT recipient, reason, source, suppressed_at FROM suppression "
                "ORDER BY suppressed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_success_on_day(self, day_prefix: str) -> int:
        """Count confirmed live SUCCESS sends whose updated_at starts with day_prefix (YYYY-MM-DD)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM send_log WHERE status = ? AND updated_at LIKE ?",
                (STATUS_SUCCESS, day_prefix + "%"),
            ).fetchone()
        return int(row["n"]) if row else 0

    def has_any_campaign_send(self, recipient: str) -> bool:
        """True if recipient has SUCCESS/SIMULATED under ANY campaign.

        The duplicate rule itself stays campaign-scoped (a deliberate
        product rule), but callers use this to warn when re-contacting
        someone already emailed under a different campaign.
        """
        recipient_norm = recipient.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                """SELECT id FROM send_log
                   WHERE recipient = ? AND status IN (?, ?) LIMIT 1""",
                (recipient_norm, STATUS_SUCCESS, STATUS_SIMULATED),
            ).fetchone()
        return row is not None

    def remove_draft_record(self, draft_id: str) -> bool:
        """
        Remove draft from send_log UNLESS its status is SUCCESS.
        Preserves SUCCESS records to prevent duplicate sends and maintain audit history.
        Returns True if a record was removed.
        """
        with self._immediate_conn() as conn:
            cur = conn.execute(
                "DELETE FROM send_log WHERE draft_id = ? AND status != 'SUCCESS';",
                (draft_id,),
            )
            return cur.rowcount > 0

    def remove_batch_records(self, campaign_id: str) -> int:
        """
        Remove drafts belonging to a campaign UNLESS status is SUCCESS.
        Returns count of removed records.
        """
        with self._immediate_conn() as conn:
            cur = conn.execute(
                "DELETE FROM send_log WHERE campaign_id = ? AND status != 'SUCCESS';",
                (campaign_id,),
            )
            return cur.rowcount

    def clear_draft_records(self, keep_success: bool = True) -> int:
        """
        Clear draft records from send_log.
        By default, preserves all SUCCESS records for audit and compliance.
        """
        with self._immediate_conn() as conn:
            if keep_success:
                cur = conn.execute("DELETE FROM send_log WHERE status != 'SUCCESS';")
            else:
                cur = conn.execute("DELETE FROM send_log;")
            return cur.rowcount

    def clear_for_testing(self) -> None:
        """Drop and recreate tables. Only safe for in-memory or test databases."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DROP TABLE IF EXISTS send_log;")
            self._init_db()
