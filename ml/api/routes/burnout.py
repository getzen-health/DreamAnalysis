"""Predictive burnout detection API routes (issue #418).

POST /burnout/analyze    — analyze emotional trajectory for burnout risk
GET  /burnout/status     — model availability
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/burnout", tags=["burnout"])


class DailySnapshotInput(BaseModel):
    date: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    mean_valence: float = Field(..., ge=-1.0, le=1.0)
    max_valence: float = Field(..., ge=-1.0, le=1.0)
    min_valence: float = Field(..., ge=-1.0, le=1.0)
    mean_arousal: float = Field(..., ge=0.0, le=1.0)
    arousal_variance: float = Field(0.0, ge=0.0)
    mean_stress: float = Field(0.0, ge=0.0, le=1.0)
    mean_focus: float = Field(0.0, ge=0.0, le=1.0)
    sleep_quality: Optional[float] = Field(None, ge=0.0, le=100.0)
    check_in_count: int = Field(0, ge=0)
    is_weekend: bool = False


class BurnoutRequest(BaseModel):
    user_id: str
    daily_snapshots: List[DailySnapshotInput]


class BurnoutSignalResponse(BaseModel):
    name: str
    description: str
    value: float
    trend_slope: float
    weeks_trending: int
    severity: str


class BurnoutResponse(BaseModel):
    user_id: str
    risk_score: int
    risk_level: str
    signals: List[BurnoutSignalResponse]
    warning_signals_count: int
    primary_concern: str
    recommended_actions: List[str]
    data_weeks: int
    confidence: float


@router.post("/analyze", response_model=BurnoutResponse)
def analyze_burnout(req: BurnoutRequest) -> Dict[str, Any]:
    """Analyze emotional trajectory for burnout risk indicators.

    Requires daily emotional snapshots spanning at least 2 weeks.
    Best results with 4-8 weeks of continuous data.
    """
    try:
        from models.burnout_detector import (
            DailySnapshot,
            analyze_burnout_trajectory,
            assessment_to_dict,
        )
    except ImportError as exc:
        log.error("burnout_detector import failed: %s", exc)
        raise HTTPException(503, "Burnout detector unavailable") from exc

    if len(req.daily_snapshots) < 7:
        raise HTTPException(422, "At least 7 days of data required")

    snapshots = [
        DailySnapshot(
            date=s.date,
            mean_valence=s.mean_valence,
            max_valence=s.max_valence,
            min_valence=s.min_valence,
            mean_arousal=s.mean_arousal,
            arousal_variance=s.arousal_variance,
            mean_stress=s.mean_stress,
            mean_focus=s.mean_focus,
            sleep_quality=s.sleep_quality,
            check_in_count=s.check_in_count,
            is_weekend=s.is_weekend,
        )
        for s in req.daily_snapshots
    ]

    try:
        assessment = analyze_burnout_trajectory(snapshots)
    except Exception as exc:
        log.exception("Burnout analysis failed: %s", exc)
        raise HTTPException(500, f"Analysis error: {exc}") from exc

    result = assessment_to_dict(assessment)
    result["user_id"] = req.user_id
    return result


@router.get("/status")
def burnout_status() -> Dict[str, Any]:
    """Return availability status of the burnout detector."""
    try:
        from models.burnout_detector import analyze_burnout_trajectory  # noqa: F401
        available = True
    except ImportError:
        available = False

    return {
        "available": available,
        "model_type": "multi-signal trajectory analysis",
        "description": (
            "Analyzes weeks-to-months of emotional data to detect burnout onset. "
            "Monitors 6 signals: emotional range compression, valence drift, "
            "arousal flattening, recovery failure, engagement decay, and "
            "sleep-mood decoupling."
        ),
        "minimum_data": "7 days (2+ weeks recommended)",
        "risk_levels": ["green (0-29)", "yellow (30-59)", "orange (60-79)", "red (80-100)"],
    }
