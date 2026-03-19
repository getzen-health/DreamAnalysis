"""Neurodivergent emotion model API routes (issue #413).

POST /neurodivergent/map-input   -- map alternative input to valence/arousal
POST /neurodivergent/profile     -- compute neurodivergent-adapted profile
GET  /neurodivergent/status      -- model availability check
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/neurodivergent", tags=["neurodivergent"])


# -- Pydantic schemas --------------------------------------------------------

class MapInputRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    modality: str = Field(
        ...,
        description="Input modality: 'color', 'energy', or 'sensory'",
    )
    value: str = Field(
        ...,
        description="Input value (e.g. color name, energy level, sensory state)",
    )
    # Optional modality-specific fields
    intensity: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Color intensity multiplier (color modality only)",
    )
    trend: str = Field(
        "stable",
        description="Energy trend: 'rising', 'falling', or 'stable' (energy modality only)",
    )
    sensory_domains: Optional[List[str]] = Field(
        None,
        description="Affected sensory domains (sensory modality only)",
    )


class MapInputResponse(BaseModel):
    user_id: str
    valence: float
    arousal: float
    confidence: float
    modality: str
    raw_input: str


class ProfileRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    input_modality: str = Field("direct", description="How valence/arousal was obtained")
    raw_input: str = Field("", description="Original user input")
    emotion_history: Optional[List[float]] = Field(
        None,
        description="Recent valence readings for ADHD volatility computation",
    )
    negative_spikes: int = Field(0, ge=0, description="Count of sharp negative valence drops")
    total_readings: int = Field(0, ge=0, description="Total emotion readings for RSD computation")
    spike_magnitude_avg: float = Field(0.0, ge=0.0, description="Average magnitude of negative spikes")


class ADHDProfileResponse(BaseModel):
    intensity: float
    volatility: float
    rejection_sensitivity: float
    hyperfocus_valence: float
    emotional_inertia: float
    current_state: str


class ProfileResponse(BaseModel):
    user_id: str
    valence: float
    arousal: float
    input_modality: str
    raw_input: str
    calibration_applied: bool
    confidence: float
    adhd_profile: Optional[ADHDProfileResponse] = None


# -- Endpoints ---------------------------------------------------------------

@router.post("/map-input", response_model=MapInputResponse)
def map_input(req: MapInputRequest) -> Dict[str, Any]:
    """Map an alternative input modality to standard valence/arousal space.

    Supports three modalities:
    - **color**: map color names (red, blue, etc.) to emotion space
    - **energy**: map battery-level metaphors (charged, drained, etc.)
    - **sensory**: map sensory processing states (overstimulated, regulated, etc.)
    """
    try:
        from models.neurodivergent_model import (
            map_color_to_emotion,
            map_energy_to_emotion,
            map_sensory_state_to_emotion,
        )
    except ImportError as exc:
        log.error("neurodivergent_model import failed: %s", exc)
        raise HTTPException(503, "Neurodivergent model unavailable") from exc

    modality = req.modality.lower().strip()

    try:
        if modality == "color":
            result = map_color_to_emotion(req.value, intensity=req.intensity)
        elif modality == "energy":
            result = map_energy_to_emotion(req.value, trend=req.trend)
        elif modality == "sensory":
            result = map_sensory_state_to_emotion(
                req.value,
                sensory_domains=req.sensory_domains,
            )
        else:
            raise HTTPException(
                422,
                f"Unknown modality '{req.modality}'. Must be 'color', 'energy', or 'sensory'.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Input mapping failed: %s", exc)
        raise HTTPException(500, f"Mapping error: {exc}") from exc

    return {
        "user_id": req.user_id,
        "valence": result["valence"],
        "arousal": result["arousal"],
        "confidence": result["confidence"],
        "modality": modality,
        "raw_input": req.value,
    }


@router.post("/profile", response_model=ProfileResponse)
def compute_profile(req: ProfileRequest) -> Dict[str, Any]:
    """Compute a neurodivergent-adapted emotional profile.

    Includes ADHD intensity/volatility dimension, rejection sensitivity
    tracking, and emotional state classification.
    """
    try:
        from models.neurodivergent_model import (
            compute_neurodivergent_profile,
            profile_to_dict,
        )
    except ImportError as exc:
        log.error("neurodivergent_model import failed: %s", exc)
        raise HTTPException(503, "Neurodivergent model unavailable") from exc

    try:
        profile = compute_neurodivergent_profile(
            valence=req.valence,
            arousal=req.arousal,
            input_modality=req.input_modality,
            raw_input=req.raw_input,
            emotion_history=req.emotion_history,
            negative_spikes=req.negative_spikes,
            total_readings=req.total_readings,
            spike_magnitude_avg=req.spike_magnitude_avg,
        )
    except Exception as exc:
        log.exception("Profile computation failed: %s", exc)
        raise HTTPException(500, f"Computation error: {exc}") from exc

    result = profile_to_dict(profile)
    result["user_id"] = req.user_id
    return result


@router.get("/status")
def neurodivergent_status() -> Dict[str, Any]:
    """Return availability status of the neurodivergent emotion model."""
    try:
        from models.neurodivergent_model import (
            map_color_to_emotion,
            compute_neurodivergent_profile,
        )
        available = True
    except ImportError:
        available = False

    return {
        "available": available,
        "model_type": "alternative-modality emotion mapping + ADHD adaptation",
        "description": (
            "Maps alternative emotion inputs (color, energy battery, sensory state) "
            "to standard valence/arousal space. Includes ADHD-adapted intensity/"
            "volatility dimension with rejection sensitivity tracking and per-user "
            "calibration."
        ),
        "input_modalities": ["color", "energy", "sensory"],
        "adhd_features": [
            "intensity (how strongly emotions are felt)",
            "volatility (how rapidly emotions shift)",
            "rejection_sensitivity (RSD score)",
            "emotional_inertia (difficulty shifting out of emotion)",
            "state classification (regulated/dysregulated/hyperfocused/rsd_triggered)",
        ],
    }
