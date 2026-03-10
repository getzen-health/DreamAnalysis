"""Biometric snapshot endpoint.

Clients call POST /biometrics/update to push Apple Health / Google Fit /
wearable data into the per-user BiometricSnapshot cache.  The WebSocket
and /analyze-eeg paths then automatically read this cache when running
MultimodalEmotionFusion.fuse(), enriching every prediction with whatever
signals are available.

All fields are optional — send only what you have.
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ._shared import get_biometric_snapshot, update_biometric_snapshot

router = APIRouter(prefix="/biometrics", tags=["biometrics"])


class BiometricUpdateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    # HRV / Heart
    hrv_sdnn: Optional[float] = None
    hrv_rmssd: Optional[float] = None
    hrv_lf_hf_ratio: Optional[float] = None
    hrv_hf_power: Optional[float] = None
    resting_heart_rate: Optional[float] = None
    current_heart_rate: Optional[float] = None
    heart_rate_variability_coherence: Optional[float] = None
    # Respiratory
    respiratory_rate: Optional[float] = None
    breath_coherence: Optional[float] = None
    # Blood oxygen
    spo2: Optional[float] = None
    # Skin temperature
    skin_temperature_deviation: Optional[float] = None
    # Sleep (previous night)
    sleep_total_hours: Optional[float] = None
    sleep_rem_hours: Optional[float] = None
    sleep_deep_hours: Optional[float] = None
    sleep_efficiency: Optional[float] = None
    sleep_onset_latency_min: Optional[float] = None
    hours_since_wake: Optional[float] = None
    # Activity
    steps_today: Optional[float] = None
    active_energy_kcal: Optional[float] = None
    exercise_minutes_today: Optional[float] = None
    days_since_last_workout: Optional[float] = None
    # Nutrition / food
    minutes_since_last_meal: Optional[float] = None
    estimated_glucose_load: Optional[float] = None
    caffeine_minutes_ago: Optional[float] = None


@router.post("/update")
async def update_biometrics(req: BiometricUpdateRequest):
    """Push latest biometric readings into the per-user fusion cache.

    Call this whenever wearable data changes (e.g. on Apple Health
    background fetch, after a meal log, or at session start).
    """
    fields = req.model_dump(exclude={"user_id"}, exclude_none=True)
    snap = update_biometric_snapshot(req.user_id, fields)
    filled = sum(1 for v in vars(snap).values() if v is not None)
    return {"ok": True, "signals_cached": filled}


@router.get("/status/{user_id}")
async def biometric_status(user_id: str):
    """Return the current cached biometric snapshot for a user."""
    snap = get_biometric_snapshot(user_id)
    data = {k: v for k, v in vars(snap).items() if v is not None}
    return {"user_id": user_id, "signals_cached": len(data), "snapshot": data}
