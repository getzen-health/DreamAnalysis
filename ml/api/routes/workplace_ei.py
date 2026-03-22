"""Workplace emotional intelligence API.

POST /workplace-ei/daily/{user_id}          — log daily emotion + HRV snapshot
POST /workplace-ei/meeting/{user_id}        — log meeting pre/post check-in
GET  /workplace-ei/report/{user_id}         — full burnout risk + productivity report
GET  /workplace-ei/meetings/{user_id}       — recent meeting climate history
DELETE /workplace-ei/reset/{user_id}        — clear all data
GET  /workplace-ei/status                   — health check
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from ._shared import sanitize_id

router = APIRouter(tags=["Workplace EI"])


class DailySnapshot(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)
    stress_index: float = Field(0.5, ge=0.0, le=1.0)
    focus_index: float = Field(0.5, ge=0.0, le=1.0)
    relaxation_index: float = Field(0.5, ge=0.0, le=1.0)
    hrv_rmssd: Optional[float] = Field(None, ge=0.0, description="HRV RMSSD in ms (optional)")
    work_hours: Optional[float] = Field(None, ge=0.0, le=24.0, description="Hours worked today")
    date: Optional[str] = Field(None, description="ISO date string (defaults to today)")


class EmotionState(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)
    stress_index: float = Field(0.5, ge=0.0, le=1.0)
    focus_index: float = Field(0.5, ge=0.0, le=1.0)
    relaxation_index: float = Field(0.5, ge=0.0, le=1.0)


class MeetingCheckin(BaseModel):
    meeting_name: str = Field(..., description="Short name or title of the meeting")
    pre_state: EmotionState
    post_state: EmotionState


@router.post("/workplace-ei/daily/{user_id}")
def log_daily(user_id: str, snapshot: DailySnapshot) -> dict:
    """Log a daily emotion + HRV snapshot for burnout risk tracking.

    Call once per day (or after each work session) with the user's current
    emotional state and optional HRV data. After 7+ days, burnout risk
    becomes meaningful (based on progressive HRV decline + stress escalation).
    """
    sanitize_id(user_id, "user_id")
    try:
        from models.workplace_ei import get_tracker

        return get_tracker().log_daily(user_id, snapshot.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/workplace-ei/meeting/{user_id}")
def log_meeting(user_id: str, checkin: MeetingCheckin) -> dict:
    """Log a meeting with pre/post emotional check-ins.

    Computes meeting climate (energizing/neutral/mixed/draining) from the
    change in valence, stress, and focus across the meeting duration.

    Use case: voice micro-check-in before and after each meeting.
    """
    sanitize_id(user_id, "user_id")
    try:
        from models.workplace_ei import get_tracker

        return get_tracker().log_meeting(
            user_id,
            checkin.meeting_name,
            checkin.pre_state.dict(),
            checkin.post_state.dict(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/workplace-ei/report/{user_id}")
def get_report(user_id: str) -> dict:
    """Return full workplace EI report including burnout risk and productivity insights.

    Burnout risk (0-1) is computed from:
    - HRV RMSSD trend over last 14 days (decline = risk)
    - Average stress_index elevation
    - Focus_index decline
    - Valence trend

    Requires 7+ daily snapshots for meaningful burnout risk.
    """
    sanitize_id(user_id, "user_id")
    try:
        from models.workplace_ei import get_tracker

        return get_tracker().get_report(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/workplace-ei/meetings/{user_id}")
def get_meeting_history(user_id: str, limit: int = 20) -> dict:
    """Return recent meeting climate history.

    Returns the last `limit` meetings with climate label (energizing/draining/neutral/mixed)
    and emotional delta (valence/stress/energy change).
    """
    sanitize_id(user_id, "user_id")
    try:
        from models.workplace_ei import get_tracker

        meetings = get_tracker().get_meeting_history(user_id, limit)
        return {"meetings": meetings, "total": len(meetings)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/workplace-ei/reset/{user_id}")
def reset(user_id: str) -> dict:
    """Clear all workplace EI data for a user."""
    sanitize_id(user_id, "user_id")
    try:
        from models.workplace_ei import get_tracker

        return get_tracker().reset(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/workplace-ei/status")
def status() -> dict:
    return {
        "status": "ready",
        "features": [
            "burnout_risk_detection",
            "meeting_climate_analysis",
            "productivity_emotion_correlation",
            "hrv_baseline_tracking",
        ],
        "burnout_markers": [
            "progressive HRV decline (Golkar 2014)",
            "sustained stress elevation",
            "focus deterioration",
            "negative valence trend",
        ],
        "note": "Requires 7+ daily snapshots for meaningful burnout risk. Not a medical device.",
    }
