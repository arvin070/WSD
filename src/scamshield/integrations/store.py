"""
Conversation state for live message streams.

The analyser reasons about a *thread* — escalation, per-sender profiles, burst
timing — but a webhook delivers one message at a time. This keeps enough
recent history per contact to rebuild that thread on every inbound message.

SQLite via the standard library: no service to run, safe across restarts, and
the write volume of a single business number is nowhere near its limits. A
connection is opened per operation rather than shared, because FastAPI runs
handlers on a thread pool and SQLite connections are not thread-safe.

Message ids are the primary key, which makes the store the deduplication
point as well: Meta retries a webhook until it gets a 2xx, so the same
message routinely arrives more than once and must be analysed only the first
time.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id      TEXT PRIMARY KEY,
    contact_id      TEXT NOT NULL,
    phone_number_id TEXT,
    direction       TEXT NOT NULL DEFAULT 'inbound',
    ts              INTEGER NOT NULL,
    text            TEXT NOT NULL,
    risk            REAL,
    verdict         TEXT,
    category        TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_contact ON messages(contact_id, ts);

CREATE TABLE IF NOT EXISTS contacts (
    contact_id     TEXT PRIMARY KEY,
    profile_name   TEXT,
    first_seen     INTEGER NOT NULL,
    last_seen      INTEGER NOT NULL,
    message_count  INTEGER NOT NULL DEFAULT 0,
    max_risk       REAL NOT NULL DEFAULT 0,
    alert_count    INTEGER NOT NULL DEFAULT 0,
    last_alert_ts  INTEGER
);
"""


@dataclass(frozen=True)
class StoredMessage:
    message_id: str
    contact_id: str
    direction: str
    ts: datetime
    text: str
    risk: float | None = None
    verdict: str | None = None


@dataclass(frozen=True)
class ContactProfile:
    contact_id: str
    profile_name: str
    message_count: int
    max_risk: float
    alert_count: int
    last_alert_ts: datetime | None


def _now() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


class ConversationStore:
    """Append-only history keyed by contact, with per-message deduplication."""

    def __init__(self, path: Path | str, history_limit: int = 100):
        self.path = Path(path)
        self.history_limit = history_limit
        self._lock = threading.Lock()
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            # WAL lets the reader in a concurrent request proceed during writes.
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- writes ------------------------------------------------------------
    def record_inbound(self, message_id: str, contact_id: str, text: str,
                       ts: datetime, phone_number_id: str = "",
                       profile_name: str = "") -> bool:
        """
        Store a message. Returns True if it is new, False if already seen.

        The caller uses the return value to skip re-analysing Meta's retries.
        """
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO messages "
                "(message_id, contact_id, phone_number_id, direction, ts, text) "
                "VALUES (?, ?, ?, 'inbound', ?, ?)",
                (message_id, contact_id, phone_number_id, int(ts.timestamp()), text))
            if cur.rowcount == 0:
                return False
            now = _now()
            conn.execute(
                "INSERT INTO contacts (contact_id, profile_name, first_seen, last_seen, message_count) "
                "VALUES (?, ?, ?, ?, 1) "
                "ON CONFLICT(contact_id) DO UPDATE SET "
                "  last_seen=excluded.last_seen, "
                "  message_count=contacts.message_count+1, "
                "  profile_name=COALESCE(NULLIF(excluded.profile_name,''), contacts.profile_name)",
                (contact_id, profile_name, now, now))
            return True

    def record_verdict(self, message_id: str, contact_id: str, risk: float,
                       verdict: str, category: str | None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE messages SET risk=?, verdict=?, category=? WHERE message_id=?",
                         (risk, verdict, category, message_id))
            conn.execute("UPDATE contacts SET max_risk=MAX(max_risk, ?) WHERE contact_id=?",
                         (risk, contact_id))

    def record_alert(self, contact_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE contacts SET alert_count=alert_count+1, last_alert_ts=? "
                         "WHERE contact_id=?", (_now(), contact_id))

    def record_outbound(self, message_id: str, contact_id: str, text: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO messages "
                "(message_id, contact_id, direction, ts, text) VALUES (?, ?, 'outbound', ?, ?)",
                (message_id, contact_id, _now(), text))

    # -- reads -------------------------------------------------------------
    def history(self, contact_id: str, limit: int | None = None) -> list[StoredMessage]:
        """Most recent messages for a contact, oldest first."""
        with self._connect() as conn:
            # rowid is implicit on the base table but not carried through a
            # subquery, so it is selected explicitly to break ties on ts.
            rows = conn.execute(
                "SELECT * FROM (SELECT *, rowid AS _rid FROM messages WHERE contact_id=? "
                "ORDER BY ts DESC, _rid DESC LIMIT ?) ORDER BY ts ASC, _rid ASC",
                (contact_id, limit or self.history_limit)).fetchall()
        return [StoredMessage(
            message_id=r["message_id"], contact_id=r["contact_id"], direction=r["direction"],
            ts=datetime.fromtimestamp(r["ts"], tz=timezone.utc), text=r["text"],
            risk=r["risk"], verdict=r["verdict"]) for r in rows]

    def contact(self, contact_id: str) -> ContactProfile | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM contacts WHERE contact_id=?", (contact_id,)).fetchone()
        if not r:
            return None
        return ContactProfile(
            contact_id=r["contact_id"], profile_name=r["profile_name"] or "",
            message_count=r["message_count"], max_risk=r["max_risk"],
            alert_count=r["alert_count"],
            last_alert_ts=(datetime.fromtimestamp(r["last_alert_ts"], tz=timezone.utc)
                           if r["last_alert_ts"] else None))

    def stats(self) -> dict:
        with self._connect() as conn:
            msgs = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
            contacts = conn.execute("SELECT COUNT(*) c FROM contacts").fetchone()["c"]
            flagged = conn.execute(
                "SELECT COUNT(*) c FROM messages WHERE verdict IN "
                "('likely_scam','dangerous')").fetchone()["c"]
            alerts = conn.execute("SELECT COALESCE(SUM(alert_count),0) c FROM contacts").fetchone()["c"]
        return {"messages": msgs, "contacts": contacts, "flagged": flagged, "alerts_sent": alerts}

    def prune(self, keep_per_contact: int | None = None) -> int:
        """Drop history beyond the retention window. Returns rows removed."""
        keep = keep_per_contact or self.history_limit
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM messages WHERE rowid NOT IN ("
                "  SELECT rowid FROM messages m WHERE m.contact_id = messages.contact_id"
                "  ORDER BY ts DESC LIMIT ?)", (keep,))
            return cur.rowcount
