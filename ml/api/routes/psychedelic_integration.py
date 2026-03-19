"""Psychedelic integration companion API routes.

POST /psychedelic/phase        -- detect current session phase from EEG
POST /psychedelic/readiness    -- pre-session readiness assessment
POST /psychedelic/integration  -- compute integration score
GET  /psychedelic/status       -- model availability

Issue #446: Psychedelic integration companion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.psychedelic_integration import (
    EEGReading,
    EmotionCategory,
    ReadinessLevel,
    SafetyLevel,
    SessionPhase,
    _CLINICAL_DISCLAIMER,
    _SAFETY_RESOURCES,
    assess_readiness,
    check_session_safety,
    compute_integration_score,
    compute_session_profile,
    detect_session_phase,
    profile_to_dict,
    track_emotional_processing,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/psychedelic", tags=["psychedelic-integration"])


# ---------------------------------------------------------------------------
# Pydantic request schemas
# ---------------------------------------------------------------------------


class PhaseDetectRequest(BaseModel):
    theta_fraction: float = Field(..., ge=0.0, le=1.0, description="Theta band power fraction")
    alpha_fraction: float = Field(..., ge=0.0, le=1.0, description="Alpha band power fraction")
    beta_fraction: float = Field(default=0.0, ge=0.0, le=1.0, description="Beta band power fraction")
    gamma_fraction: float = Field(default=0.0, ge=0.0, le=1.0, description="Gamma band power fraction")
    spectral_entropy: float = Field(default=0.0, ge=0.0, le=1.0, description="Spectral entropy (0-1)")
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Emotional valence")
    arousal: float = Field(default=0.0, ge=0.0, le=1.0, description="Emotional arousal")


class ReadinessRequest(BaseModel):
    valence: float = Field(..., ge=-1.0, le=1.0, description="Current emotional valence")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Current arousal level")
    stress_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Stress index")
    anxiety_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Anxiety index")
    sleep_quality: float = Field(default=0.5, ge=0.0, le=1.0, description="Recent sleep quality")
    intention_clarity: float = Field(default=0.5, ge=0.0, le=1.0, description="Therapeutic intention clarity")


class IntegrationRequest(BaseModel):
    pre_session_valence: float = Field(..., ge=-1.0, le=1.0, description="Pre-session emotional valence")
    post_session_valence: float = Field(..., ge=-1.0, le=1.0, description="Current post-session valence")
    processing_depth: float = Field(default=0.0, ge=0.0, le=1.0, description="Average session processing depth")
    unresolved_count: int = Field(default=0, ge=0, description="Number of unresolved themes")
    days_since_session: int = Field(default=0, ge=0, description="Days since the session")
    emotional_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="Post-session emotional stability")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/phase")
async def detect_phase(req: PhaseDetectRequest) -> Dict[str, Any]:
    """Detect current session phase from EEG spectral features.

    Analyzes theta, alpha, beta, gamma fractions and spectral entropy
    to determine which phase of the psychedelic experience the user is in:
    preparation, onset, peak, plateau, comedown, or integration.
    """
    reading = EEGReading(
        theta_fraction=req.theta_fraction,
        alpha_fraction=req.alpha_fraction,
        beta_fraction=req.beta_fraction,
        gamma_fraction=req.gamma_fraction,
        spectral_entropy=req.spectral_entropy,
        valence=req.valence,
        arousal=req.arousal,
    )
    return detect_session_phase(reading)


@router.post("/readiness")
async def check_readiness(req: ReadinessRequest) -> Dict[str, Any]:
    """Pre-session readiness assessment.

    Evaluates emotional state, stress, anxiety, sleep quality, and
    intention clarity to determine if the user is ready for a session.
    """
    return assess_readiness(
        valence=req.valence,
        arousal=req.arousal,
        stress_index=req.stress_index,
        anxiety_index=req.anxiety_index,
        sleep_quality=req.sleep_quality,
        intention_clarity=req.intention_clarity,
    )


@router.post("/integration")
async def integration_score(req: IntegrationRequest) -> Dict[str, Any]:
    """Compute integration score: did the session produce lasting change?

    Measures how well insights from the session have been integrated
    based on valence shift, processing depth, resolution of themes,
    and emotional stability.
    """
    return compute_integration_score(
        pre_session_valence=req.pre_session_valence,
        post_session_valence=req.post_session_valence,
        processing_depth=req.processing_depth,
        unresolved_count=req.unresolved_count,
        days_since_session=req.days_since_session,
        emotional_stability=req.emotional_stability,
    )


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Model availability and psychedelic integration companion info."""
    return {
        "status": "ok",
        "model": "psychedelic_integration",
        "version": "1.0.0",
        "session_phases": [p.value for p in SessionPhase if p != SessionPhase.UNKNOWN],
        "safety_levels": [s.value for s in SafetyLevel],
        "readiness_levels": [r.value for r in ReadinessLevel],
        "emotion_categories": [e.value for e in EmotionCategory],
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        "safety_resources": _SAFETY_RESOURCES,
    }
