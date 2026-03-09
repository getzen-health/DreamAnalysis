"""Sleep-to-next-day mood prediction API endpoints.

Endpoints
---------
POST /sleep-mood/predict
    Predict next-day emotional state from last night's sleep metrics.

GET  /sleep-mood/predict/{user_id}
    Predict using the user's most recent Apple Health sleep entry
    (falls back to request body if no stored data).

GET  /sleep-mood/circadian-baseline
    Return the circadian emotion baseline for a given hour (no sleep data needed).
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.sleep_mood_predictor import SleepMoodPredictor
from models.context_prior import ContextPrior, _circadian_valence, _circadian_arousal, _circadian_stress

router = APIRouter(prefix="/sleep-mood", tags=["Sleep Mood Predictor"])

_predictor = SleepMoodPredictor()
_context = ContextPrior()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class SleepMetrics(BaseModel):
    """Sleep metrics from Apple Health, Oura, or EEG sleep session."""
    total_sleep_hours: Optional[float] = Field(
        default=None, description="Total hours slept"
    )
    deep_sleep_pct: Optional[float] = Field(
        default=None,
        description="Fraction of total sleep in N3 deep sleep (0-1)",
    )
    rem_pct: Optional[float] = Field(
        default=None,
        description="Fraction of total sleep in REM (0-1)",
    )
    sleep_efficiency: Optional[float] = Field(
        default=None,
        description="Time asleep / time in bed (0-1)",
    )
    wake_after_sleep_onset: Optional[float] = Field(
        default=None,
        description="Minutes of waking after sleep onset (WASO)",
    )
    sleep_onset_latency: Optional[float] = Field(
        default=None,
        description="Minutes to fall asleep after lights out",
    )
    bedtime_regularity: Optional[float] = Field(
        default=None,
        description="Std dev of bedtime in minutes over last 7 days",
    )
    hrv_during_sleep: Optional[float] = Field(
        default=None,
        description="Mean RMSSD in ms during sleep",
    )
    resting_hr_during_sleep: Optional[float] = Field(
        default=None,
        description="Resting heart rate during sleep (BPM)",
    )
    sleep_debt: Optional[float] = Field(
        default=None,
        description="Cumulative sleep debt in hours over last 7 days",
    )


class SleepMoodRequest(BaseModel):
    """Request body for sleep-to-mood prediction."""
    user_id: str = Field(default="default", description="User identifier")
    sleep: SleepMetrics = Field(
        default_factory=SleepMetrics,
        description="Last night's sleep metrics",
    )
    include_circadian: bool = Field(
        default=True,
        description="Include circadian baseline in the response",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/predict")
async def predict_next_day_mood(req: SleepMoodRequest):
    """Predict next-day emotional state from last night's sleep.

    Uses evidence-based heuristics calibrated to published effect sizes:
    - npj Digital Medicine (2024): 36 sleep/circadian features predict mood episodes
    - MDPI (2024): HRV + heart rate during sleep → next-day state
    - Established sleep architecture research (Hobson 2005, Dang-Vu 2010)

    At least 1 sleep metric is required; more metrics = higher confidence.
    Recommended: total_sleep_hours + deep_sleep_pct + rem_pct + hrv_during_sleep.

    Returns predicted valence, arousal, stress risk, focus window, confidence,
    key influencing factor, and human-readable interpretation.
    """
    sleep_dict = req.sleep.model_dump(exclude_none=True)

    prediction = _predictor.predict_next_day(
        sleep_data=sleep_dict,
        user_id=req.user_id,
    )

    response = {
        "user_id": req.user_id,
        "prediction": prediction,
        "n_sleep_features": len(sleep_dict),
        "predicted_at": time.time(),
    }

    if req.include_circadian:
        # Add circadian context for a typical morning hour (8am)
        morning_ctx = _context.compute_prior(hour=8.0)
        response["circadian_morning_baseline"] = {
            "valence_offset": morning_ctx["valence_offset"],
            "arousal_offset": morning_ctx["arousal_offset"],
            "stress_offset": morning_ctx["stress_offset"],
            "note": "Circadian baseline at 8am — independent of sleep quality",
        }

    return response


@router.get("/circadian-baseline")
async def get_circadian_baseline(hour: float = 9.0):
    """Return the circadian emotion baseline for a given hour of day.

    No sleep data required — this is the population-average emotion
    trajectory driven purely by time of day (circadian rhythm).

    Sources: Stone et al. (2006) experience sampling; Monk (2005) post-lunch dip.

    Args:
        hour: Hour of day (0-24). Default: 9.0 (9am).

    Returns:
        Expected valence/arousal/stress offsets at this hour,
        relative to a neutral baseline (0.0).
    """
    if hour < 0 or hour > 24:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="hour must be between 0 and 24")

    ctx = _context.compute_prior(hour=hour)
    valence_c = _circadian_valence(hour)
    arousal_c = _circadian_arousal(hour)
    stress_c = _circadian_stress(hour)

    # Identify time-of-day label
    if 5 <= hour < 9:
        period = "early morning"
    elif 9 <= hour < 12:
        period = "mid-morning (peak alertness)"
    elif 12 <= hour < 14:
        period = "midday"
    elif 14 <= hour < 17:
        period = "afternoon / post-lunch dip"
    elif 17 <= hour < 21:
        period = "evening"
    else:
        period = "night"

    return {
        "hour": hour,
        "period": period,
        "circadian_valence_offset": round(valence_c, 4),
        "circadian_arousal_offset": round(arousal_c, 4),
        "circadian_stress_offset": round(stress_c, 4),
        "full_context_prior": ctx,
        "note": (
            "These are population-average offsets from neutral. "
            "Individual circadian timing varies by ~1-2 hours from the average."
        ),
    }
