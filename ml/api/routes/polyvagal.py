"""Polyvagal state tracking API routes (#439).

Endpoints:
  POST /polyvagal/classify  -- classify current autonomic state
  POST /polyvagal/profile   -- compute full polyvagal profile from history
  GET  /polyvagal/status     -- availability check

References:
  - Issue #439: Polyvagal state tracking
  - Porges (2011) The Polyvagal Theory
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/polyvagal", tags=["polyvagal"])


# -- Request / Response schemas -----------------------------------------


class AutonomicSampleSchema(BaseModel):
    hrv_rmssd: float = Field(
        ..., ge=0, description="Heart rate variability RMSSD in ms"
    )
    heart_rate: float = Field(
        ..., gt=0, description="Heart rate in bpm"
    )
    alpha_power: float = Field(
        ..., ge=0, le=1.0, description="EEG alpha relative power (0-1)"
    )
    beta_alpha_ratio: float = Field(
        ..., ge=0, description="EEG beta / alpha power ratio"
    )
    resp_rate: float = Field(
        ..., ge=0, description="Respiratory rate in breaths per minute"
    )
    timestamp: float = Field(
        default=0.0, description="Unix epoch seconds (0 = auto)"
    )


class ClassifyRequest(BaseModel):
    sample: AutonomicSampleSchema


class ProfileRequest(BaseModel):
    samples: List[AutonomicSampleSchema] = Field(
        ...,
        min_length=3,
        description="At least 3 chronologically ordered autonomic samples",
    )


# -- Endpoints ----------------------------------------------------------


@router.post("/classify")
def classify_polyvagal(req: ClassifyRequest) -> Dict[str, Any]:
    """Classify current autonomic state from a single sample.

    Returns the polyvagal state (ventral_vagal, sympathetic, or
    dorsal_vagal) with confidence and probability distribution.
    """
    from models.polyvagal_model import AutonomicSample, classify_polyvagal_state

    import time

    s = req.sample
    sample = AutonomicSample(
        hrv_rmssd=s.hrv_rmssd,
        heart_rate=s.heart_rate,
        alpha_power=s.alpha_power,
        beta_alpha_ratio=s.beta_alpha_ratio,
        resp_rate=s.resp_rate,
        timestamp=s.timestamp if s.timestamp > 0 else time.time(),
    )

    result = classify_polyvagal_state(sample)

    return {
        "status": "ok",
        "state": result.state,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "timestamp": result.timestamp,
    }


@router.post("/profile")
def compute_profile(req: ProfileRequest) -> Dict[str, Any]:
    """Compute a full polyvagal profile from a history of samples.

    Requires at least 3 samples. Returns state distribution, transition
    matrix, dwell times, autonomic flexibility score, and full trajectory.
    """
    from models.polyvagal_model import (
        AutonomicSample,
        compute_polyvagal_profile,
        profile_to_dict,
    )

    import time

    samples = []
    for s in req.samples:
        samples.append(
            AutonomicSample(
                hrv_rmssd=s.hrv_rmssd,
                heart_rate=s.heart_rate,
                alpha_power=s.alpha_power,
                beta_alpha_ratio=s.beta_alpha_ratio,
                resp_rate=s.resp_rate,
                timestamp=s.timestamp if s.timestamp > 0 else time.time(),
            )
        )

    try:
        profile = compute_polyvagal_profile(samples)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        **profile_to_dict(profile),
    }


@router.get("/status")
def polyvagal_status() -> Dict[str, Any]:
    """Check whether the polyvagal state tracking module is available."""
    try:
        from models.polyvagal_model import STATES  # noqa: F401

        return {
            "available": True,
            "module": "polyvagal_model",
            "version": "1.0.0",
            "states": list(STATES),
            "features": [
                "state_classification",
                "trajectory_analysis",
                "transition_matrix",
                "dwell_times",
                "autonomic_flexibility",
            ],
        }
    except ImportError:
        return {
            "available": False,
            "module": "polyvagal_model",
            "error": "module_not_found",
        }
