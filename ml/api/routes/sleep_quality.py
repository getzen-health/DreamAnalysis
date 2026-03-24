"""Sleep quality prediction API.

GitHub issue: #122
"""
from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ._shared import _numpy_safe
from models.sleep_quality_predictor import SleepQualityPredictor
from models.sleep_staging import compute_waso, compute_sleep_stats

router = APIRouter(tags=["sleep-quality"])

_predictor = SleepQualityPredictor()


class SleepInput(BaseModel):
    n3_pct: float = Field(0.0, ge=0.0, le=1.0, description="Deep sleep (N3) percentage (0-1)")
    rem_pct: float = Field(0.0, ge=0.0, le=1.0, description="REM sleep percentage (0-1)")
    n2_pct: float = Field(0.0, ge=0.0, le=1.0, description="N2 sleep percentage (0-1)")
    n1_pct: float = Field(0.0, ge=0.0, le=1.0, description="N1 sleep percentage (0-1)")
    wake_pct: float = Field(0.0, ge=0.0, le=1.0, description="Wake percentage (0-1)")
    sleep_efficiency: float = Field(0.85, ge=0.0, le=1.0, description="Time asleep / time in bed (0-1)")
    spindle_density: float = Field(0.0, ge=0.0, description="Sleep spindles per minute")
    total_sleep_hours: float = Field(7.0, ge=0.0, description="Total hours of sleep")
    waso_minutes: float = Field(0.0, ge=0.0, description="Wake after sleep onset (minutes)")
    hrv_ms: Optional[float] = Field(None, description="HRV RMSSD during sleep (optional)")
    user_id: str = Field(..., description="User identifier")


@router.post("/sleep-quality/predict")
async def predict_sleep_quality(data: SleepInput):
    """Predict sleep quality score and next-day readiness from sleep architecture.

    Returns quality_score (0-100), quality_label, readiness_forecast,
    component scores (deep, REM, efficiency, duration, continuity, spindles),
    and improvement suggestions.
    """
    result = _predictor.predict(
        n3_pct=data.n3_pct,
        rem_pct=data.rem_pct,
        n2_pct=data.n2_pct,
        n1_pct=data.n1_pct,
        wake_pct=data.wake_pct,
        sleep_efficiency=data.sleep_efficiency,
        spindle_density=data.spindle_density,
        total_sleep_hours=data.total_sleep_hours,
        waso_minutes=data.waso_minutes,
        hrv_ms=data.hrv_ms,
        user_id=data.user_id,
    )
    return _numpy_safe(result)


@router.get("/sleep-quality/trend")
async def get_sleep_trend(user_id: str, window: int = 7):
    """Get sleep quality trend over the last N nights."""
    result = _predictor.get_trend(user_id=user_id, window=window)
    result["user_id"] = user_id
    return _numpy_safe(result)


@router.get("/sleep-quality/history")
async def get_sleep_history(user_id: str, last_n: int = 30):
    """Get sleep quality history for a user."""
    history = _predictor.get_history(user_id=user_id, last_n=last_n)
    return _numpy_safe({"history": history, "user_id": user_id, "count": len(history)})


@router.post("/sleep-quality/reset")
async def reset_sleep_quality(user_id: str):
    """Clear sleep quality history for a user."""
    _predictor.reset(user_id=user_id)
    return {"status": "ok", "message": "Sleep quality history cleared.", "user_id": user_id}


class StagedEpoch(BaseModel):
    stage: str = Field(..., description="Sleep stage: Wake, N1, N2, N3, REM, or artifact")
    stage_index: int = Field(..., description="Stage index: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, -1=artifact")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Classification confidence")


class SleepStatsInput(BaseModel):
    epochs: List[StagedEpoch] = Field(..., description="Staged epoch sequence from predict_sequence()")
    epoch_duration_s: float = Field(30.0, gt=0, description="Duration of each epoch in seconds")
    user_id: str = Field("default", description="User identifier")


@router.post("/sleep-quality/stats")
async def get_sleep_stats(data: SleepStatsInput):
    """Compute comprehensive sleep statistics from staged epochs.

    Accepts a sequence of staged epochs (output of predict_sequence) and
    returns all clinical sleep metrics: stage percentages, WASO, number
    of awakenings, sleep onset latency, sleep efficiency, and a quality
    prediction -- all auto-computed.

    This is the preferred endpoint for post-session sleep analysis. Instead
    of manually computing stage percentages and WASO, pass the raw staged
    epoch list and get everything in one call.
    """
    epoch_dicts = [e.model_dump() for e in data.epochs]

    stats = compute_sleep_stats(epoch_dicts, epoch_duration_s=data.epoch_duration_s)
    if stats is None:
        return {"error": "No valid epochs provided", "stats": None, "quality": None}

    # Auto-compute quality prediction from the derived stats
    quality = _predictor.predict(
        n3_pct=stats["n3_pct"],
        rem_pct=stats["rem_pct"],
        n2_pct=stats["n2_pct"],
        n1_pct=stats["n1_pct"],
        wake_pct=stats["wake_pct"],
        sleep_efficiency=stats["sleep_efficiency"],
        total_sleep_hours=stats["total_sleep_minutes"] / 60.0,
        waso_minutes=stats["waso_minutes"],
        user_id=data.user_id,
    )

    return _numpy_safe({
        "stats": stats,
        "quality": quality,
    })
