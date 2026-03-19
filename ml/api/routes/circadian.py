"""Circadian neural signature API routes.

POST /circadian/compute          — compute circadian profile from feature streams
GET  /circadian/status           — model availability check

Issue #410: Personalized brain clock from multi-modal fusion.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/circadian", tags=["circadian"])


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class FeaturePoint(BaseModel):
    time_h: float = Field(..., ge=0.0, le=24.0, description="Time of day in fractional hours (0–24)")
    value: float = Field(..., description="Feature value (raw or z-scored)")


class CircadianRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    feature_streams: Dict[str, List[FeaturePoint]] = Field(
        ...,
        description=(
            "Map of stream name to time-stamped values. "
            "Expected streams: alpha_beta_ratio, valence, arousal, hrv, stress, focus"
        ),
    )
    current_hour: float = Field(
        12.0,
        ge=0.0,
        le=24.0,
        description="Current time of day in fractional hours",
    )
    baseline_acrophase: Optional[float] = Field(
        None,
        ge=0.0,
        le=24.0,
        description="Previous learned acrophase for phase-shift detection",
    )


class CosinorFitResponse(BaseModel):
    mesor: float
    amplitude: float
    acrophase_h: float
    period_h: float
    r_squared: float
    p_value: float
    n_samples: int


class CircadianResponse(BaseModel):
    user_id: str
    chronotype: str
    chronotype_confidence: float
    acrophase_h: float
    amplitude: float
    period_h: float
    phase_stability: float
    current_phase: float
    predicted_focus_window: str
    predicted_slump_window: str
    phase_shift_hours: float
    fits: Dict[str, CosinorFitResponse]
    data_days: int
    minimum_days_met: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/compute", response_model=CircadianResponse)
def compute_circadian(req: CircadianRequest) -> Dict[str, Any]:
    """Compute personalized circadian profile from multi-modal feature streams.

    Fits a cosinor model (gold standard in chronobiology) to each feature stream
    and produces a weighted consensus acrophase, chronotype, focus/slump windows,
    and phase-shift detection.

    Requires 7–14 days of data for reliable results; returns best-effort with less.
    """
    try:
        from models.circadian_model import compute_circadian_profile, profile_to_dict
    except ImportError as exc:
        log.error("circadian_model import failed: %s", exc)
        raise HTTPException(503, "Circadian model unavailable") from exc

    # Convert Pydantic models to plain dicts for the model
    streams: Dict[str, list] = {}
    total_points = 0
    for name, points in req.feature_streams.items():
        streams[name] = [{"time_h": p.time_h, "value": p.value} for p in points]
        total_points += len(points)

    if total_points < 6:
        raise HTTPException(
            422,
            f"Insufficient data: {total_points} points provided, minimum 6 required",
        )

    try:
        profile = compute_circadian_profile(
            feature_streams=streams,
            current_hour=req.current_hour,
            baseline_acrophase=req.baseline_acrophase,
        )
    except Exception as exc:
        log.exception("Circadian profile computation failed: %s", exc)
        raise HTTPException(500, f"Computation error: {exc}") from exc

    result = profile_to_dict(profile)
    result["user_id"] = req.user_id
    result["minimum_days_met"] = profile.data_days >= 7

    return result


@router.get("/status")
def circadian_status() -> Dict[str, Any]:
    """Return availability status of the circadian model."""
    try:
        from models.circadian_model import compute_circadian_profile  # noqa: F401
        available = True
    except ImportError:
        available = False

    return {
        "available": available,
        "model_type": "cosinor oscillator (Refinetti et al. 2007)",
        "description": (
            "Fits personalized circadian rhythm model to multi-modal neural data "
            "(EEG alpha/beta, valence, arousal, HRV). Discovers true chronotype, "
            "predicts focus/slump windows, and detects phase shifts."
        ),
        "minimum_data": "7 days recommended, 14 days for reliable phase-shift detection",
        "inputs": [
            "alpha_beta_ratio (EEG — most weighted)",
            "valence (EEG/voice)",
            "arousal (EEG/voice)",
            "hrv (Apple Health / wearable)",
            "stress (composite)",
            "focus (composite)",
        ],
        "outputs": [
            "chronotype (early_bird / night_owl / intermediate)",
            "acrophase_h (peak time in hours)",
            "predicted_focus_window",
            "predicted_slump_window",
            "phase_stability (0–1)",
            "phase_shift_hours (drift from baseline)",
            "per-stream cosinor fits (mesor, amplitude, acrophase, R², p-value)",
        ],
    }
