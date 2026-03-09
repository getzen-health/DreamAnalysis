"""Digital phenotyping API endpoints.

Endpoints:
    POST /phenotype/daily-score       — Compute wellness score for one day
    POST /phenotype/weekly-trend      — Compute 7-day trend from daily scores
    POST /phenotype/monthly-report    — 30-day aggregate report
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Digital Phenotyping"])


# ── Request models ─────────────────────────────────────────────────────────────

class EEGSessionSummary(BaseModel):
    valence: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    arousal: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stress_index: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    focus_index: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    session_minutes: Optional[float] = Field(default=None, ge=0)


class HealthData(BaseModel):
    steps: Optional[int] = Field(default=None, ge=0)
    resting_hr: Optional[float] = Field(default=None, gt=0)
    hrv: Optional[float] = Field(default=None, gt=0, description="HRV in ms")
    sleep_hours: Optional[float] = Field(default=None, ge=0, le=24)
    active_kcal: Optional[float] = Field(default=None, ge=0)


class DailyScoreRequest(BaseModel):
    eeg_sessions: List[EEGSessionSummary] = Field(
        default=[], description="EEG session summaries for the day"
    )
    health_data: Optional[HealthData] = Field(
        default=None, description="Apple Health metrics for the day"
    )
    date_label: str = Field(default="today", description="Human-readable date")


class WeeklyTrendRequest(BaseModel):
    daily_scores: List[float] = Field(
        ..., description="Daily wellness scores (0-100), most-recent last",
        min_length=1,
    )


class MonthlyReportRequest(BaseModel):
    daily_records: List[Dict] = Field(
        ..., description="List of daily score result dicts",
        min_length=1,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/phenotype/daily-score")
async def compute_daily_score(request: DailyScoreRequest):
    """Compute mental wellness score (0-100) for one day.

    Fuses EEG biomarkers (valence, stress, focus) with Apple Health metrics
    (sleep, HRV, steps, resting HR) into a single wellness score.
    Missing data reduces the score's weight rather than causing errors.
    """
    try:
        from models.digital_phenotyper import get_digital_phenotyper
        sessions = [s.model_dump(exclude_none=True) for s in request.eeg_sessions]
        health = request.health_data.model_dump(exclude_none=True) if request.health_data else {}
        result = get_digital_phenotyper().compute_daily_score(
            eeg_sessions=sessions,
            health_data=health,
            date_label=request.date_label,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/phenotype/weekly-trend")
async def compute_weekly_trend(request: WeeklyTrendRequest):
    """Compute 7-day trend from daily wellness scores.

    Returns trend label (improving/stable/declining), slope per day,
    and summary statistics.
    """
    try:
        from models.digital_phenotyper import get_digital_phenotyper
        result = get_digital_phenotyper().compute_weekly_trend(request.daily_scores)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/phenotype/monthly-report")
async def compute_monthly_report(request: MonthlyReportRequest):
    """Aggregate up to 30 days of daily records into a monthly summary.

    Provides mean/min/max scores, trend direction, total sessions,
    and the most frequent risk flags over the period.
    """
    try:
        from models.digital_phenotyper import get_digital_phenotyper
        result = get_digital_phenotyper().compute_monthly_report(request.daily_records)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
