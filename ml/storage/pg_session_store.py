"""PostgreSQL session metadata store for the ML backend.

Writes session records directly to the shared Neon PostgreSQL database
using the same `eeg_sessions` table defined in shared/schema.ts.

Falls back silently to no-op if DATABASE_URL is not set or psycopg2 is
unavailable — the local JSON files remain the source of truth in that case.
"""

import json
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _get_conn():
    """Return a new psycopg2 connection or None if unavailable."""
    if not _DATABASE_URL:
        return None
    try:
        import psycopg2
        return psycopg2.connect(_DATABASE_URL)
    except Exception as exc:
        logger.warning(f"pg_session_store: cannot connect to DB: {exc}")
        return None


def upsert_session(
    session_id: str,
    user_id: str,
    session_type: str,
    start_time: Optional[float],
    end_time: Optional[float],
    status: str,
    summary: Optional[Dict],
    signal_r2_key: Optional[str],
) -> bool:
    """Insert or update a session row in eeg_sessions.

    Returns True on success, False on error / no DB.
    """
    conn = _get_conn()
    if conn is None:
        return False

    sql = """
        INSERT INTO eeg_sessions
            (session_id, user_id, session_type, status, start_time, end_time, summary, signal_r2_key)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (session_id) DO UPDATE SET
            status       = EXCLUDED.status,
            end_time     = EXCLUDED.end_time,
            summary      = EXCLUDED.summary,
            signal_r2_key = EXCLUDED.signal_r2_key
    """
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    session_id,
                    user_id,
                    session_type,
                    status,
                    start_time,
                    end_time,
                    json.dumps(summary) if summary else None,
                    signal_r2_key,
                ))
        return True
    except Exception as exc:
        logger.error(f"pg_session_store upsert failed: {exc}")
        return False
    finally:
        conn.close()


def delete_session(session_id: str) -> bool:
    """Delete a session row from eeg_sessions."""
    conn = _get_conn()
    if conn is None:
        return False
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM eeg_sessions WHERE session_id = %s", (session_id,)
                )
        return True
    except Exception as exc:
        logger.error(f"pg_session_store delete failed: {exc}")
        return False
    finally:
        conn.close()
