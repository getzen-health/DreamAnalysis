"""Apple Health / Google Fit / Health Connect integration endpoints."""

import os
import sqlite3
from typing import Optional

from fastapi import APIRouter, HTTPException

from ._shared import (
    _numpy_safe,
    HealthDataPayload, BrainSessionPayload,
)

router = APIRouter()

# Health DB singleton — TimescaleDB when DATABASE_URL is set, SQLite otherwise
from health.correlation_engine import HealthBrainDB, TimescaleHealthDB
from health.apple_health import parse_healthkit_payload, format_brain_data_for_healthkit
from health.google_fit import parse_google_fit_payload, parse_health_connect_payload

try:
    if os.environ.get("DATABASE_URL"):
        _health_db = TimescaleHealthDB()
    else:
        _health_db = HealthBrainDB()
except Exception as _e:
    import logging
    logging.getLogger(__name__).warning("TimescaleHealthDB init failed, using SQLite: %s", _e)
    _health_db = HealthBrainDB()


@router.post("/health/ingest")
async def ingest_health_data(payload: HealthDataPayload):
    """Ingest health data from Apple Health, Google Fit, or Health Connect."""
    try:
        if payload.source == "apple_health":
            parsed = parse_healthkit_payload(payload.data)
        elif payload.source == "google_fit":
            parsed = parse_google_fit_payload(payload.data)
        elif payload.source == "health_connect":
            parsed = parse_health_connect_payload(payload.data)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source '{payload.source}'. Use 'apple_health', 'google_fit', or 'health_connect'.",
            )

        _health_db.store_health_samples(
            user_id=payload.user_id,
            metric=parsed["metric"],
            samples=parsed["samples"],
        )

        return {
            "status": "stored",
            "metric": parsed["metric"],
            "samples_stored": parsed["count"],
            "source": payload.source,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health/brain-session")
async def store_brain_session(payload: BrainSessionPayload):
    """Store a brain state session for correlation with health data."""
    try:
        session_data = {
            "session_id": payload.session_id,
            "start_time": payload.start_time,
            "end_time": payload.end_time,
            "duration_seconds": payload.duration_seconds,
            **payload.analysis,
        }
        _health_db.store_brain_session(payload.user_id, session_data)
        return {"status": "stored", "session_id": payload.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/daily-summary/{user_id}")
async def daily_summary(user_id: str, date: Optional[str] = None):
    """Get combined brain + health summary for a day."""
    try:
        return _health_db.get_daily_summary(user_id, date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/insights/{user_id}")
async def get_insights(user_id: str, days: int = 30):
    """Generate personalized brain-health correlation insights."""
    try:
        insights = _health_db.generate_insights(user_id, days)
        return {"user_id": user_id, "period_days": days, "insights": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/trends/{user_id}")
async def brain_trends(user_id: str, days: int = 30):
    """Get brain state trends over time."""
    try:
        return _health_db.get_brain_trends(user_id, days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health/export-to-healthkit/{user_id}")
async def export_to_healthkit(user_id: str):
    """Export brain data formatted for Apple HealthKit."""
    try:
        with sqlite3.connect(_health_db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sessions = conn.execute(
                """SELECT * FROM brain_sessions
                   WHERE user_id = ? ORDER BY start_time DESC LIMIT 100""",
                (user_id,),
            ).fetchall()

        exports = []
        for session in sessions:
            session_dict = dict(session)
            brain_session = {
                "start_time": session_dict.get("start_time"),
                "end_time": session_dict.get("end_time"),
                "flow_state": {
                    "state": session_dict.get("flow_state"),
                    "flow_score": session_dict.get("flow_score"),
                },
                "sleep_stage": {
                    "stage": session_dict.get("sleep_stage"),
                    "confidence": session_dict.get("sleep_confidence"),
                },
            }
            exports.extend(format_brain_data_for_healthkit(brain_session))

        return {
            "user_id": user_id,
            "healthkit_samples": exports,
            "count": len(exports),
            "instructions": "Pass these samples to HKHealthStore.save() on iOS",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/supported-metrics")
async def supported_metrics():
    """List all supported health metrics from Apple Health and Google Fit."""
    from health.apple_health import HEALTHKIT_TYPE_MAP, EXPORT_TYPES
    from health.google_fit import GOOGLE_FIT_TYPE_MAP, HEALTH_CONNECT_TYPE_MAP

    return {
        "apple_health": {
            "import": HEALTHKIT_TYPE_MAP,
            "export": EXPORT_TYPES,
        },
        "google_fit": GOOGLE_FIT_TYPE_MAP,
        "health_connect": HEALTH_CONNECT_TYPE_MAP,
        "brain_metrics": {
            "flow_state": ["no_flow", "micro_flow", "flow", "deep_flow"],
            "creativity": ["analytical", "transitional", "creative", "insight"],
            "memory_encoding": ["poor_encoding", "weak_encoding", "active_encoding", "deep_encoding"],
            "sleep_staging": ["Wake", "N1", "N2", "N3", "REM"],
            "emotions": ["happy", "sad", "angry", "fearful", "relaxed", "focused"],
            "dream_detection": ["dreaming", "not_dreaming"],
            "drowsiness": ["alert", "drowsy", "sleepy"],
            "cognitive_load": ["low", "moderate", "high"],
            "attention": ["distracted", "passive", "focused", "hyperfocused"],
            "stress": ["relaxed", "mild", "moderate", "high"],
            "lucid_dream": ["non_lucid", "pre_lucid", "lucid", "controlled"],
            "meditation": ["surface", "light", "moderate", "deep", "transcendent"],
        },
    }
