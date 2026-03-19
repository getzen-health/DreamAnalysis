"""Neural age biomarker API routes (issue #447).

POST /neural-age/estimate  — estimate brain age from EEG features
POST /neural-age/profile   — full neural age profile with longitudinal tracking
GET  /neural-age/status    — model availability
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/neural-age", tags=["neural-age"])


# ------------------------------------------------------------------ #
# Pydantic models
# ------------------------------------------------------------------ #


class EEGAgeFeaturesInput(BaseModel):
    """Input EEG features for neural age estimation."""

    alpha_peak_freq: float = Field(
        ..., ge=4.0, le=16.0, description="Alpha peak frequency in Hz"
    )
    theta_beta_ratio: float = Field(
        ..., ge=0.0, description="Theta/beta power ratio"
    )
    alpha_power: float = Field(
        ..., ge=0.0, le=1.0, description="Relative alpha power (0-1)"
    )
    emotional_range: float = Field(
        ..., ge=0.0, le=1.0, description="Emotional breadth (0-1)"
    )
    reaction_time_ms: float = Field(
        ..., ge=0.0, description="Reaction time proxy in ms"
    )


class NeuralAgeEstimateRequest(BaseModel):
    """Request for neural age estimation."""

    user_id: str
    features: EEGAgeFeaturesInput
    chronological_age: Optional[float] = Field(
        None, ge=10.0, le=120.0, description="User's actual age in years"
    )


class NeuralAgeEstimateResponse(BaseModel):
    """Response from /estimate."""

    user_id: str
    neural_age: float
    brain_age_gap: Optional[float]
    gap_interpretation: Optional[str]
    feature_ages: Dict[str, float]
    confidence: float
    warnings: List[str]
    disclaimer: str
    processed_at: float


class HistoryEntry(BaseModel):
    """A single historical measurement for longitudinal tracking."""

    neural_age: float
    elapsed_days: float = Field(..., ge=0.0, description="Days since first measurement")


class LifestyleInput(BaseModel):
    """Optional lifestyle factors for personalized recommendations."""

    sleep_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    exercise_hours_weekly: Optional[float] = Field(None, ge=0.0)
    meditation_minutes_daily: Optional[float] = Field(None, ge=0.0)
    stress_level: Optional[float] = Field(None, ge=0.0, le=1.0)
    social_hours_weekly: Optional[float] = Field(None, ge=0.0)


class NeuralAgeProfileRequest(BaseModel):
    """Request for full neural age profile with longitudinal data."""

    user_id: str
    features: EEGAgeFeaturesInput
    chronological_age: float = Field(
        ..., ge=10.0, le=120.0, description="User's actual age in years"
    )
    history: Optional[List[HistoryEntry]] = Field(
        None, description="Past measurements for aging rate computation"
    )
    lifestyle: Optional[LifestyleInput] = None


class NeuralAgeProfileResponse(BaseModel):
    """Response from /profile."""

    user_id: str
    neural_age: float
    chronological_age: float
    brain_age_gap: float
    gap_interpretation: str
    aging_rate: Optional[float]
    aging_rate_interpretation: Optional[str]
    feature_contributions: Dict[str, float]
    modifiable_factors: List[Dict[str, Any]]
    percentile: int
    confidence: float
    warnings: List[str]
    disclaimer: str
    processed_at: float


# ------------------------------------------------------------------ #
# In-memory history
# ------------------------------------------------------------------ #

_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))


# ------------------------------------------------------------------ #
# POST /neural-age/estimate
# ------------------------------------------------------------------ #


@router.post("/estimate", response_model=NeuralAgeEstimateResponse)
async def estimate_neural_age_endpoint(req: NeuralAgeEstimateRequest):
    """Estimate brain age from EEG features.

    Takes alpha peak frequency, theta/beta ratio, alpha power,
    emotional range, and reaction time proxy to compute a neural
    age estimate using population norm matching.
    """
    try:
        from models.neural_age import EEGAgeFeatures, estimate_neural_age
    except ImportError as exc:
        log.error("neural_age model import failed: %s", exc)
        raise HTTPException(503, "Neural age model unavailable") from exc

    features = EEGAgeFeatures(
        alpha_peak_freq=req.features.alpha_peak_freq,
        theta_beta_ratio=req.features.theta_beta_ratio,
        alpha_power=req.features.alpha_power,
        emotional_range=req.features.emotional_range,
        reaction_time_ms=req.features.reaction_time_ms,
    )

    try:
        result = estimate_neural_age(features, req.chronological_age)
    except Exception as exc:
        log.exception("Neural age estimation failed: %s", exc)
        raise HTTPException(500, f"Estimation error: {exc}") from exc

    out = NeuralAgeEstimateResponse(
        user_id=req.user_id,
        neural_age=result["neural_age"],
        brain_age_gap=result.get("brain_age_gap"),
        gap_interpretation=result.get("gap_interpretation"),
        feature_ages=result["feature_ages"],
        confidence=result["confidence"],
        warnings=result["warnings"],
        disclaimer=result["disclaimer"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


# ------------------------------------------------------------------ #
# POST /neural-age/profile
# ------------------------------------------------------------------ #


@router.post("/profile", response_model=NeuralAgeProfileResponse)
async def neural_age_profile_endpoint(req: NeuralAgeProfileRequest):
    """Full neural age profile with longitudinal tracking.

    Combines age estimation, brain age gap analysis, aging rate
    computation (if history provided), factor identification, and
    personalized lifestyle recommendations.
    """
    try:
        from models.neural_age import (
            EEGAgeFeatures,
            compute_neural_age_profile,
            profile_to_dict,
        )
    except ImportError as exc:
        log.error("neural_age model import failed: %s", exc)
        raise HTTPException(503, "Neural age model unavailable") from exc

    features = EEGAgeFeatures(
        alpha_peak_freq=req.features.alpha_peak_freq,
        theta_beta_ratio=req.features.theta_beta_ratio,
        alpha_power=req.features.alpha_power,
        emotional_range=req.features.emotional_range,
        reaction_time_ms=req.features.reaction_time_ms,
    )

    history = None
    if req.history:
        history = [
            {"neural_age": h.neural_age, "elapsed_days": h.elapsed_days}
            for h in req.history
        ]

    lifestyle = None
    if req.lifestyle:
        lifestyle = {
            k: v
            for k, v in req.lifestyle.model_dump().items()
            if v is not None
        }

    try:
        profile = compute_neural_age_profile(
            features=features,
            chronological_age=req.chronological_age,
            history=history,
            lifestyle=lifestyle,
        )
        result = profile_to_dict(profile)
    except Exception as exc:
        log.exception("Neural age profile computation failed: %s", exc)
        raise HTTPException(500, f"Profile error: {exc}") from exc

    out = NeuralAgeProfileResponse(
        user_id=req.user_id,
        neural_age=result["neural_age"],
        chronological_age=result["chronological_age"],
        brain_age_gap=result["brain_age_gap"],
        gap_interpretation=result["gap_interpretation"],
        aging_rate=result.get("aging_rate"),
        aging_rate_interpretation=result.get("aging_rate_interpretation"),
        feature_contributions=result["feature_contributions"],
        modifiable_factors=result["modifiable_factors"],
        percentile=result["percentile"],
        confidence=result["confidence"],
        warnings=result["warnings"],
        disclaimer=result["disclaimer"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


# ------------------------------------------------------------------ #
# GET /neural-age/status
# ------------------------------------------------------------------ #


@router.get("/status")
async def neural_age_status() -> Dict[str, Any]:
    """Return availability status of the neural age biomarker model."""
    try:
        from models.neural_age import estimate_neural_age  # noqa: F401

        available = True
    except ImportError:
        available = False

    return {
        "available": available,
        "model_type": "population-norm EEG feature matching",
        "description": (
            "Estimates neural age from EEG features (alpha peak frequency, "
            "theta/beta ratio, alpha power, emotional range, reaction time) "
            "and computes Brain Age Gap relative to chronological age. "
            "Supports longitudinal aging rate tracking and lifestyle factor analysis."
        ),
        "features_required": [
            "alpha_peak_freq",
            "theta_beta_ratio",
            "alpha_power",
            "emotional_range",
            "reaction_time_ms",
        ],
        "endpoints": [
            "POST /neural-age/estimate",
            "POST /neural-age/profile",
            "GET /neural-age/status",
        ],
    }
