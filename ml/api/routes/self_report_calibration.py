"""Self-report calibration API routes (issue #411).

POST /calibration/self-report/observe   — record a paired observation
POST /calibration/self-report/compute   — compute calibration profile
GET  /calibration/self-report/status    — model availability
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/calibration/self-report", tags=["self-report-calibration"])


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ObservationInput(BaseModel):
    reported_valence: float = Field(..., ge=-1.0, le=1.0)
    reported_arousal: float = Field(..., ge=0.0, le=1.0)
    measured_valence: float = Field(..., ge=-1.0, le=1.0)
    measured_arousal: float = Field(..., ge=0.0, le=1.0)
    eeg_valence: Optional[float] = Field(None, ge=-1.0, le=1.0)
    voice_valence: Optional[float] = Field(None, ge=-1.0, le=1.0)
    channel_agreement: float = Field(0.0, ge=-1.0, le=1.0)
    time_h: float = Field(12.0, ge=0.0, le=24.0)
    context: str = ""


class CalibrationRequest(BaseModel):
    user_id: str
    observations: List[ObservationInput]
    current_measured_valence: Optional[float] = Field(None, ge=-1.0, le=1.0)
    current_measured_arousal: Optional[float] = Field(None, ge=0.0, le=1.0)


class CalibrationResponse(BaseModel):
    user_id: str
    reporter_type: str
    reporter_confidence: float
    emotional_awareness_score: float
    gap_statistics: Dict[str, Any]
    bias_model: Optional[Dict[str, float]]
    valence_blind_spots: List[str]
    trend_direction: str
    calibrated_valence: Optional[float]
    calibrated_arousal: Optional[float]
    n_observations: int
    sufficient_data: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/compute", response_model=CalibrationResponse)
def compute_calibration(req: CalibrationRequest) -> Dict[str, Any]:
    """Compute self-report calibration profile from paired observations.

    Detects systematic biases between what users report feeling and what
    EEG/voice objectively measures. Classifies reporter type (accurate,
    suppressor, amplifier, inconsistent) and provides awareness scoring.
    """
    try:
        from models.self_report_calibration import (
            CalibrationObservation,
            compute_calibration_profile,
            profile_to_dict,
        )
    except ImportError as exc:
        log.error("self_report_calibration import failed: %s", exc)
        raise HTTPException(503, "Self-report calibration model unavailable") from exc

    if not req.observations:
        raise HTTPException(422, "At least one observation is required")

    observations = [
        CalibrationObservation(
            reported_valence=o.reported_valence,
            reported_arousal=o.reported_arousal,
            measured_valence=o.measured_valence,
            measured_arousal=o.measured_arousal,
            eeg_valence=o.eeg_valence,
            voice_valence=o.voice_valence,
            channel_agreement=o.channel_agreement,
            time_h=o.time_h,
            context=o.context,
        )
        for o in req.observations
    ]

    try:
        profile = compute_calibration_profile(
            observations,
            current_measured_valence=req.current_measured_valence,
            current_measured_arousal=req.current_measured_arousal,
        )
    except Exception as exc:
        log.exception("Calibration computation failed: %s", exc)
        raise HTTPException(500, f"Computation error: {exc}") from exc

    result = profile_to_dict(profile)
    result["user_id"] = req.user_id
    result["sufficient_data"] = len(observations) >= 10

    return result


@router.get("/status")
def calibration_status() -> Dict[str, Any]:
    """Return availability status of the self-report calibration model."""
    try:
        from models.self_report_calibration import compute_calibration_profile  # noqa: F401
        available = True
    except ImportError:
        available = False

    return {
        "available": available,
        "model_type": "linear bias model + reporter type classifier",
        "description": (
            "Detects systematic discrepancies between self-reported and objectively "
            "measured emotions. Classifies users as accurate reporters, suppressors, "
            "amplifiers, or inconsistent. Computes Emotional Awareness Score (0-100)."
        ),
        "minimum_data": "10 paired observations for bias model, 20 for trend detection",
        "reporter_types": [
            "accurate — |gap| < 0.15 consistently",
            "suppressor — reports better than measured (common pattern)",
            "amplifier — reports worse than measured (clinical anxiety pattern)",
            "inconsistent — high gap variance (possible alexithymia)",
        ],
    }
