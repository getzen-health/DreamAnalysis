"""Emotion-aware accessibility layer API routes — issue #460.

Endpoints:
  POST /accessibility/adapt   -- get UI adaptations for current EEG state
  GET  /accessibility/status   -- health check

Issue #460.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.emotion_accessibility import (
    AccessibilityProfile,
    EEGState,
    assess_accessibility_needs,
    assessment_to_dict,
    compute_accessibility_profile,
    profile_to_dict,
    recommend_adaptations,
)

router = APIRouter(prefix="/accessibility", tags=["accessibility"])


# -- Request / response models -----------------------------------------------

class EEGStateInput(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Emotional valence")
    arousal: float = Field(0.5, ge=0.0, le=1.0, description="Arousal level")
    stress: float = Field(0.3, ge=0.0, le=1.0, description="Stress level")
    fatigue: float = Field(0.3, ge=0.0, le=1.0, description="Fatigue level")
    cognitive_load: float = Field(0.3, ge=0.0, le=1.0, description="Cognitive load")


class ProfileInput(BaseModel):
    visual_impairment: float = Field(0.0, ge=0.0, le=1.0, description="Visual impairment severity")
    motor_difficulty: float = Field(0.0, ge=0.0, le=1.0, description="Motor difficulty severity")
    cognitive_load_sensitivity: float = Field(0.0, ge=0.0, le=1.0, description="Cognitive load sensitivity")
    sensory_sensitivity: float = Field(0.0, ge=0.0, le=1.0, description="Sensory sensitivity")


class AdaptRequest(BaseModel):
    eeg_state: EEGStateInput = Field(..., description="Current EEG-derived state")
    profile: Optional[ProfileInput] = Field(None, description="User accessibility profile")


# -- Endpoints ----------------------------------------------------------------

@router.post("/adapt")
async def adapt(req: AdaptRequest) -> Dict[str, Any]:
    """Get UI adaptations for the current EEG emotional state.

    Accepts current EEG state and optional accessibility profile. Returns
    concrete UI adaptation recommendations (font size, contrast, animation
    speed, information density, audio cues) tailored to the user's needs.
    """
    eeg = EEGState(
        valence=req.eeg_state.valence,
        arousal=req.eeg_state.arousal,
        stress=req.eeg_state.stress,
        fatigue=req.eeg_state.fatigue,
        cognitive_load=req.eeg_state.cognitive_load,
        timestamp=time.time(),
    )

    profile = None
    if req.profile:
        profile = AccessibilityProfile(
            visual_impairment=req.profile.visual_impairment,
            motor_difficulty=req.profile.motor_difficulty,
            cognitive_load_sensitivity=req.profile.cognitive_load_sensitivity,
            sensory_sensitivity=req.profile.sensory_sensitivity,
        )

    assessment = assess_accessibility_needs(eeg, profile)
    recommend_adaptations(assessment)

    return assessment_to_dict(assessment)


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Health check — confirms the accessibility module is loaded."""
    return {
        "status": "ready",
        "model_type": "eeg-accessibility",
        "description": "Emotion-aware accessibility layer",
        "supported_profiles": list(
            ("visual_impairment", "motor_difficulty",
             "cognitive_load_sensitivity", "sensory_sensitivity")
        ),
        "timestamp": time.time(),
    }
