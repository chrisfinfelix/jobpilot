"""
application_tracker.py
-----------------------
Step 6: a persistent record of applications, so sending a cover letter
isn't a one-and-done action that vanishes once the modal closes.

Design decision: this uses its OWN SQLite file (app/data/applications.db),
separate from the research cache's database. Reasoning: the research
cache is disposable (7-day TTL, safe to wipe and re-fetch), but tracker
data is the whole point of this feature — a user's application history.
Keeping it in its own file makes that distinction obvious at a glance and
means clearing the research cache can never accidentally touch it.

Per the current design (auto-log only), a row is created exactly once,
right after a successful Gmail send in main.py's /api/gmail/send handler.
There is no create-manually path yet — every row here corresponds to a
real email JobPilot actually sent.
"""

import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "applications.db")

VALID_STATUSES = ["applied", "interviewing", "offer", "rejected", "withdrawn"]


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_tracker_db() -> None:
    """Call once at startup — safe to call repeatedly (CREATE TABLE IF NOT EXISTS)."""
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                role TEXT NOT NULL,
                match_score INTEGER,
                status TEXT NOT NULL DEFAULT 'applied',
                recipient_email TEXT NOT NULL,
                gmail_message_id TEXT,
                cover_letter TEXT,
                applied_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_application(
    company: str,
    role: str,
    recipient_email: str,
    gmail_message_id: str | None = None,
    match_score: int | None = None,
    cover_letter: str | None = None,
) -> int:
    """
    Inserts one row after a successful send. Returns the new row's id.
    Called from main.py's /api/gmail/send handler — see that function for
    why a logging failure there doesn't fail the send response itself
    (the email already went out; losing the tracker row is a lesser
    problem than pretending the send failed when it didn't).
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO applications
                (company, role, match_score, status, recipient_email, gmail_message_id, cover_letter, applied_at, updated_at)
            VALUES (?, ?, ?, 'applied', ?, ?, ?, ?, ?)
            """,
            (company, role, match_score, recipient_email, gmail_message_id, cover_letter, now, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_applications(status: str | None = None) -> list[dict]:
    """Returns applications newest-first, optionally filtered by status."""
    conn = _connect()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM applications WHERE status = ? ORDER BY applied_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM applications ORDER BY applied_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_status(application_id: int, new_status: str) -> bool:
    """Returns True if a row was updated, False if the id didn't exist."""
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{new_status}'. Must be one of: {VALID_STATUSES}")

    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE applications SET status = ?, updated_at = ? WHERE id = ?",
            (new_status, now, application_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_stats() -> dict:
    """Simple counts per status, plus a total — powers the dashboard header."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT status, COUNT(*) as n FROM applications GROUP BY status").fetchall()
        counts = {s: 0 for s in VALID_STATUSES}
        for r in rows:
            counts[r["status"]] = r["n"]
        counts["total"] = sum(counts.values())
        return counts
    finally:
        conn.close()
