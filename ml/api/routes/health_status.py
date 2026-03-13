"""Health sync status endpoint.

GET /health/sync-status — returns aggregated connection status for all 6
health data sources: Apple HealthKit, Google Health Connect, Oura Ring,
Garmin, Whoop, and Muse 2 EEG.

Response schema per source:
  {
    "source":      str,                        # canonical source key
    "connected":   bool,
    "last_sync":   str | null,                 # ISO 8601 datetime or null
    "data_types":  list[str],                  # metric names available
    "freshness":   "fresh"|"stale"|"old"|"disconnected"
  }

Freshness bands (from last_sync timestamp):
  fresh       < 1 hour
  stale       1 – 24 hours
  old         > 24 hours (but connected)
  disconnected not connected or never synced
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

log = logging.getLogger(__name__)

router = APIRouter()

# ── Canonical source definitions ──────────────────────────────────────────────

_SOURCE_DATA_TYPES: Dict[str, List[str]] = {
    "apple_health": ["HR", "HRV", "sleep", "steps"],
    "google_health": ["steps", "HR", "calories"],
    "oura": ["sleep", "readiness", "activity"],
    "garmin": ["Body Battery", "stress", "HRV"],
    "whoop": ["recovery", "strain", "sleep"],
    "muse_eeg": ["connection", "battery", "signal"],
}

_SOURCE_ORDER = ["apple_health", "google_health", "oura", "garmin", "whoop", "muse_eeg"]


# ── Freshness helper ───────────────────────────────────────────────────────────

def _freshness(last_sync_iso: Optional[str], connected: bool) -> str:
    """Classify freshness based on last_sync timestamp and connection state."""
    if not connected or not last_sync_iso:
        return "disconnected"
    try:
        last_dt = datetime.fromisoformat(last_sync_iso.replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
        if age_hours < 1:
            return "fresh"
        if age_hours < 24:
            return "stale"
        return "old"
    except Exception:
        return "disconnected"


# ── Token / connection probes ──────────────────────────────────────────────────
# Each probe checks whether an access token is available (in env or a lightweight
# in-memory store). A missing token means disconnected. Actual OAuth flows are
# handled by the dedicated oauth route modules (oura.py, garmin.py, whoop.py).

def _check_apple_health() -> Dict[str, Any]:
    """Apple HealthKit status — driven by uploaded export files."""
    try:
        from health.correlation_engine import HealthBrainDB
        db = HealthBrainDB()
        conn = db._get_conn()
        cur = conn.execute(
            "SELECT MAX(timestamp) FROM health_samples WHERE source = 'apple_health'"
        )
        row = cur.fetchone()
        last_sync = row[0] if row and row[0] else None
        if last_sync and isinstance(last_sync, (int, float)):
            last_sync = datetime.fromtimestamp(last_sync, tz=timezone.utc).isoformat()
        connected = last_sync is not None
        return {"connected": connected, "last_sync": last_sync}
    except Exception as exc:
        log.debug("apple_health probe failed: %s", exc)
        return {"connected": False, "last_sync": None}


def _check_google_health() -> Dict[str, Any]:
    """Google Health Connect status — driven by uploaded export files."""
    try:
        from health.correlation_engine import HealthBrainDB
        db = HealthBrainDB()
        conn = db._get_conn()
        cur = conn.execute(
            "SELECT MAX(timestamp) FROM health_samples WHERE source = 'google_fit'"
        )
        row = cur.fetchone()
        last_sync = row[0] if row and row[0] else None
        if last_sync and isinstance(last_sync, (int, float)):
            last_sync = datetime.fromtimestamp(last_sync, tz=timezone.utc).isoformat()
        connected = last_sync is not None
        return {"connected": connected, "last_sync": last_sync}
    except Exception as exc:
        log.debug("google_health probe failed: %s", exc)
        return {"connected": False, "last_sync": None}


def _check_oura() -> Dict[str, Any]:
    """Oura Ring — connected if OURA_ACCESS_TOKEN is set in env."""
    token = os.environ.get("OURA_ACCESS_TOKEN") or os.environ.get("OURA_CLIENT_ID")
    if not token:
        return {"connected": False, "last_sync": None}
    # Try to read the last stored Oura sample timestamp
    try:
        from health.correlation_engine import HealthBrainDB
        db = HealthBrainDB()
        conn = db._get_conn()
        cur = conn.execute(
            "SELECT MAX(timestamp) FROM health_samples WHERE source = 'oura'"
        )
        row = cur.fetchone()
        last_sync = row[0] if row and row[0] else None
        if last_sync and isinstance(last_sync, (int, float)):
            last_sync = datetime.fromtimestamp(last_sync, tz=timezone.utc).isoformat()
        return {"connected": True, "last_sync": last_sync}
    except Exception:
        return {"connected": True, "last_sync": None}


def _check_garmin() -> Dict[str, Any]:
    """Garmin — connected if GARMIN_ACCESS_TOKEN is set in env."""
    token = os.environ.get("GARMIN_ACCESS_TOKEN") or os.environ.get("GARMIN_CONSUMER_KEY")
    if not token:
        return {"connected": False, "last_sync": None}
    try:
        from health.correlation_engine import HealthBrainDB
        db = HealthBrainDB()
        conn = db._get_conn()
        cur = conn.execute(
            "SELECT MAX(timestamp) FROM health_samples WHERE source = 'garmin'"
        )
        row = cur.fetchone()
        last_sync = row[0] if row and row[0] else None
        if last_sync and isinstance(last_sync, (int, float)):
            last_sync = datetime.fromtimestamp(last_sync, tz=timezone.utc).isoformat()
        return {"connected": True, "last_sync": last_sync}
    except Exception:
        return {"connected": True, "last_sync": None}


def _check_whoop() -> Dict[str, Any]:
    """Whoop — connected if WHOOP_ACCESS_TOKEN is set in env."""
    token = os.environ.get("WHOOP_ACCESS_TOKEN") or os.environ.get("WHOOP_CLIENT_ID")
    if not token:
        return {"connected": False, "last_sync": None}
    try:
        from health.correlation_engine import HealthBrainDB
        db = HealthBrainDB()
        conn = db._get_conn()
        cur = conn.execute(
            "SELECT MAX(timestamp) FROM health_samples WHERE source = 'whoop'"
        )
        row = cur.fetchone()
        last_sync = row[0] if row and row[0] else None
        if last_sync and isinstance(last_sync, (int, float)):
            last_sync = datetime.fromtimestamp(last_sync, tz=timezone.utc).isoformat()
        return {"connected": True, "last_sync": last_sync}
    except Exception:
        return {"connected": True, "last_sync": None}


def _check_muse_eeg() -> Dict[str, Any]:
    """Muse 2 EEG — connected if the BrainFlow device manager reports a live stream."""
    try:
        from hardware.brainflow_manager import BrainFlowManager
        mgr = BrainFlowManager()
        status = mgr.get_status() if hasattr(mgr, "get_status") else {}
        connected = status.get("connected", False) or status.get("streaming", False)
        last_sync: Optional[str] = None
        if connected:
            last_sync = datetime.now(timezone.utc).isoformat()
        return {"connected": bool(connected), "last_sync": last_sync}
    except Exception as exc:
        log.debug("muse_eeg probe failed: %s", exc)
        return {"connected": False, "last_sync": None}


_PROBES = {
    "apple_health": _check_apple_health,
    "google_health": _check_google_health,
    "oura": _check_oura,
    "garmin": _check_garmin,
    "whoop": _check_whoop,
    "muse_eeg": _check_muse_eeg,
}


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.get("/health/sync-status")
async def get_health_sync_status() -> Dict[str, Any]:
    """Return aggregated sync status for all 6 health data sources.

    Returns:
        {
            "sources": [
                {
                    "source": "apple_health",
                    "connected": true,
                    "last_sync": "2026-03-12T10:00:00+00:00",
                    "data_types": ["HR", "HRV", "sleep", "steps"],
                    "freshness": "fresh"
                },
                ...
            ],
            "fetched_at": "2026-03-12T11:30:00+00:00"
        }
    """
    sources: List[Dict[str, Any]] = []

    for key in _SOURCE_ORDER:
        probe = _PROBES.get(key)
        if probe is None:
            state: Dict[str, Any] = {"connected": False, "last_sync": None}
        else:
            try:
                state = probe()
            except Exception as exc:
                log.warning("Health probe %s raised: %s", key, exc)
                state = {"connected": False, "last_sync": None}

        connected: bool = bool(state.get("connected", False))
        last_sync: Optional[str] = state.get("last_sync")
        freshness = _freshness(last_sync, connected)

        sources.append({
            "source": key,
            "connected": connected,
            "last_sync": last_sync,
            "data_types": _SOURCE_DATA_TYPES.get(key, []),
            "freshness": freshness,
        })

    return {
        "sources": sources,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
