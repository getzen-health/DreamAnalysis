"""Emotional genome API routes (issue #444).

POST /emotional-genome/profile  -- compute full emotional genome from longitudinal data
GET  /emotional-genome/status   -- model availability
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.emotional_genome import (
    EmotionSample,
    compute_genome_profile,
    profile_to_dict,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/emotional-genome", tags=["emotional-genome"])

# In-memory profile cache keyed by user_id.
_profiles: Dict[str, Any] = {}


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------

class SampleInput(BaseModel):
    timestamp: float = Field(..., description="Unix epoch seconds")
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)
    stress: float = Field(0.3, ge=0.0, le=1.0)
    energy: float = Field(0.5, ge=0.0, le=1.0)
    social_context: float = Field(0.0, ge=-1.0, le=1.0)
    novelty_context: float = Field(0.0, ge=0.0, le=1.0)
    threat_context: float = Field(0.0, ge=0.0, le=1.0)


class ProfileRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    samples: List[SampleInput] = Field(..., min_length=1)
    max_chapters: int = Field(5, ge=1, le=20)
    evolution_window: int = Field(20, ge=5, le=100)
    evolution_step: int = Field(10, ge=1, le=50)


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@router.post("/profile")
async def compute_emotional_genome_profile(req: ProfileRequest) -> Dict[str, Any]:
    """Compute the full emotional genome from longitudinal emotion data."""
    samples = [
        EmotionSample(
            timestamp=s.timestamp,
            valence=s.valence,
            arousal=s.arousal,
            stress=s.stress,
            energy=s.energy,
            social_context=s.social_context,
            novelty_context=s.novelty_context,
            threat_context=s.threat_context,
        )
        for s in req.samples
    ]

    try:
        profile = compute_genome_profile(
            user_id=req.user_id,
            samples=samples,
            max_chapters=req.max_chapters,
            evolution_window=req.evolution_window,
            evolution_step=req.evolution_step,
        )
    except Exception as exc:
        log.exception("compute_genome_profile failed for user %s", req.user_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result = profile_to_dict(profile)
    _profiles[req.user_id] = result
    return {
        "status": "ok",
        "profile": result,
    }


@router.get("/status")
async def emotional_genome_status() -> Dict[str, Any]:
    """Return model availability and cached profile count."""
    return {
        "status": "available",
        "cached_profiles": len(_profiles),
        "profile_ids": list(_profiles.keys()),
    }
