"""Sleep-to-mood predictor API routes.

POST /sleep-mood/predict            — predict next-day mood from last night's sleep data
GET  /sleep-mood/history/{user_id}  — retrieve past predictions for a user
GET  /sleep-mood/status             — model availability check
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/sleep-mood", tags=["sleep-mood"])

# ── Lazy singleton ─────────────────────────────────────────────────────────────

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        try:
            from models.sleep_mood_predictor import get_sleep_mood_predictor  # type: ignore
            _predictor = get_sleep_mood_predictor()
        except Exception as exc:
            log.warning("SleepMoodPredictor unavailable: %s", exc)
    return _predictor


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class SleepMoodRequest(BaseModel):
    user_id: str = Field("default", description="User identifier")
    total_sleep_hours: Optional[float] = Field(
        None,
        description="Total hours of sleep (default 7.0)",
        ge=0.0,
        le=14.0,
    )
    deep_sleep_pct: Optional[float] = Field(
        None,
        description="Fraction of sleep in N3 deep sleep, 0–1 (default 0.18)",
        ge=0.0,
        le=1.0,
    )
    rem_pct: Optional[float] = Field(
        None,
        description="Fraction of sleep in REM, 0–1 (default 0.22)",
        ge=0.0,
        le=1.0,
    )
    sleep_efficiency: Optional[float] = Field(
        None,
        description="Time asleep / time in bed, 0–1 (default 0.85)",
        ge=0.0,
        le=1.0,
    )
    waso_minutes: Optional[float] = Field(
        None,
        description="Wake-after-sleep-onset in minutes (default 0)",
        ge=0.0,
        le=300.0,
    )
    sleep_onset_latency: Optional[float] = Field(
        None,
        description="Minutes to fall asleep (default 15)",
        ge=0.0,
        le=120.0,
    )
    bedtime_regularity: Optional[float] = Field(
        None,
        description="Std-dev of bedtime in hours (default 0.5). Higher = more irregular.",
        ge=0.0,
        le=6.0,
    )
    hrv_ms: Optional[float] = Field(
        None,
        description="Overnight RMSSD HRV in milliseconds (optional)",
        ge=0.0,
        le=300.0,
    )
    resting_hr_during_sleep: Optional[float] = Field(
        None,
        description="Mean heart rate during sleep in bpm (optional)",
        ge=20.0,
        le=120.0,
    )


class SleepMoodResponse(BaseModel):
    user_id: str
    predicted_valence: float
    predicted_arousal: float
    predicted_stress_risk: float
    predicted_focus_score: float
    predicted_focus_window: str
    confidence: float
    key_factor: str
    mood_label: str
    sleep_score: int
    timestamp: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=SleepMoodResponse)
def predict_sleep_mood(req: SleepMoodRequest) -> Dict[str, Any]:
    """Predict next-day emotional state from last night's sleep metrics.

    All sleep fields are optional — sensible population-average defaults are
    applied when not provided.  Returns valence, arousal, stress risk, focus
    score, best focus window, confidence, and an overall sleep score (0–100).
    """
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(
            503,
            "SleepMoodPredictor unavailable — check server logs for import errors",
        )

    # Build sleep_data dict from request, only forwarding explicitly set values
    sleep_data: Dict[str, Any] = {}
    if req.total_sleep_hours is not None:
        sleep_data["total_sleep_hours"] = req.total_sleep_hours
    if req.deep_sleep_pct is not None:
        sleep_data["deep_sleep_pct"] = req.deep_sleep_pct
    if req.rem_pct is not None:
        sleep_data["rem_pct"] = req.rem_pct
    if req.sleep_efficiency is not None:
        sleep_data["sleep_efficiency"] = req.sleep_efficiency
    if req.waso_minutes is not None:
        sleep_data["waso_minutes"] = req.waso_minutes
    if req.sleep_onset_latency is not None:
        sleep_data["sleep_onset_latency"] = req.sleep_onset_latency
    if req.bedtime_regularity is not None:
        sleep_data["bedtime_regularity"] = req.bedtime_regularity
    if req.hrv_ms is not None:
        sleep_data["hrv_ms"] = req.hrv_ms
    if req.resting_hr_during_sleep is not None:
        sleep_data["resting_hr_during_sleep"] = req.resting_hr_during_sleep

    try:
        result = predictor.predict_next_day(sleep_data=sleep_data, user_id=req.user_id)
    except Exception as exc:
        log.exception("SleepMoodPredictor.predict_next_day failed: %s", exc)
        raise HTTPException(500, f"Prediction error: {exc}")

    return {"user_id": req.user_id, **result}


@router.get("/history/{user_id}")
def get_sleep_mood_history(
    user_id: str,
    last_n: int = 14,
) -> Dict[str, Any]:
    """Return the last N sleep-to-mood predictions for a user (default 14 days)."""
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(
            503,
            "SleepMoodPredictor unavailable — check server logs",
        )

    if last_n < 1 or last_n > 90:
        raise HTTPException(422, "last_n must be between 1 and 90")

    history = predictor.get_history(user_id=user_id, last_n=last_n)
    return {
        "user_id": user_id,
        "count": len(history),
        "predictions": history,
    }


@router.get("/status")
def sleep_mood_status() -> Dict[str, Any]:
    """Return availability status of the SleepMoodPredictor."""
    predictor = _get_predictor()
    available = predictor is not None
    return {
        "available": available,
        "model_type": "evidence-based heuristics (no ML weights required)",
        "description": (
            "Forecasts next-day valence, arousal, stress risk, and focus score "
            "from last night's sleep metrics using npj Digital Medicine (2024) "
            "derived correlations."
        ),
        "inputs": [
            "total_sleep_hours",
            "deep_sleep_pct",
            "rem_pct",
            "sleep_efficiency",
            "waso_minutes",
            "sleep_onset_latency",
            "bedtime_regularity",
            "hrv_ms (optional)",
            "resting_hr_during_sleep (optional)",
        ],
        "outputs": [
            "predicted_valence",
            "predicted_arousal",
            "predicted_stress_risk",
            "predicted_focus_score",
            "predicted_focus_window",
            "confidence",
            "key_factor",
            "mood_label",
            "sleep_score",
        ],
    }
