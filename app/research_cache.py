"""
research_cache.py
------------------
A simple file-based cache so we don't re-research the same company/role
for every single user. Uses SQLite (a single local file, zero setup,
zero cost) instead of Redis — same idea, simpler infrastructure.

Cache key = normalized "company|role". Entries expire after CACHE_TTL_DAYS
(default 7, matching the original design doc: "company info doesn't change
every day, so cache it for about a week").
"""

import sqlite3
import json
import os
import time
from contextlib import contextmanager

DB_PATH = os.environ.get("JOBPILOT_CACHE_DB", "research_cache.db")
CACHE_TTL_DAYS = 7
CACHE_TTL_SECONDS = CACHE_TTL_DAYS * 24 * 60 * 60


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Create the cache table if it doesn't exist yet. Call once at startup."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS research_cache (
                cache_key   TEXT PRIMARY KEY,
                company     TEXT NOT NULL,
                role        TEXT NOT NULL,
                data_json   TEXT NOT NULL,
                fetched_at  REAL NOT NULL
            )
        """)
        conn.commit()


def _make_key(company: str, role: str) -> str:
    """Normalize so 'Google'/'google '/'GOOGLE' all hit the same cache entry."""
    return f"{company.strip().lower()}|{role.strip().lower()}"


def get_cached_research(company: str, role: str) -> dict | None:
    """
    Returns the cached research dict if a fresh (< 7 day old) entry exists,
    otherwise returns None (meaning: go fetch it fresh).
    """
    key = _make_key(company, role)
    with _connect() as conn:
        row = conn.execute(
            "SELECT data_json, fetched_at FROM research_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()

    if row is None:
        return None

    data_json, fetched_at = row
    age_seconds = time.time() - fetched_at
    if age_seconds > CACHE_TTL_SECONDS:
        return None  # stale — caller should re-fetch

    result = json.loads(data_json)
    result["_cache_hit"] = True
    result["_cached_age_hours"] = round(age_seconds / 3600, 1)
    return result


def set_cached_research(company: str, role: str, data: dict) -> None:
    """Store (or overwrite) the research result for this company/role."""
    key = _make_key(company, role)
    # Don't persist the meta fields we add on read
    clean_data = {k: v for k, v in data.items() if not k.startswith("_")}

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO research_cache (cache_key, company, role, data_json, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                data_json = excluded.data_json,
                fetched_at = excluded.fetched_at
            """,
            (key, company, role, json.dumps(clean_data), time.time()),
        )
        conn.commit()
