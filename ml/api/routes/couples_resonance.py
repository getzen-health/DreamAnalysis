"""Couples emotional resonance API routes — issue #440.

Consent-gated shared emotional dashboards for relationship wellness.

Endpoints:
  POST /couples/synchrony  -- compute synchrony between two users
  POST /couples/profile    -- full relationship emotional profile
  POST /couples/consent    -- manage partnership consent (opt-in / revoke / status)
  GET  /couples/status     -- health check

GitHub issue: #440
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.couples_resonance import (
    EmotionSample,
    compute_emotional_synchrony,
    detect_resonance_periods,
    detect_conflict,
    detect_repair,
    compute_relationship_profile,
    manage_partnership_consent,
    profile_to_dict,
    _check_partnership_active,
    _conflict_to_dict,
    _repair_to_dict,
)

router = APIRouter(prefix="/couples", tags=["couples-resonance"])


# -- Request / response models -----------------------------------------------

class EmotionSampleInput(BaseModel):
    timestamp: float = Field(..., description="Unix epoch seconds")
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional valence")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Arousal level")


class SynchronyRequest(BaseModel):
    user_a_id: str = Field(..., min_length=1, description="Partner A user ID")
    user_b_id: str = Field(..., min_length=1, description="Partner B user ID")
    samples_a: List[EmotionSampleInput] = Field(
        ..., min_length=2, description="Partner A emotion samples (chronological)"
    )
    samples_b: List[EmotionSampleInput] = Field(
        ..., min_length=2, description="Partner B emotion samples (chronological)"
    )


class ProfileRequest(BaseModel):
    user_a_id: str = Field(..., min_length=1, description="Partner A user ID")
    user_b_id: str = Field(..., min_length=1, description="Partner B user ID")
    samples_a: List[EmotionSampleInput] = Field(
        ..., min_length=2, description="Partner A emotion samples"
    )
    samples_b: List[EmotionSampleInput] = Field(
        ..., min_length=2, description="Partner B emotion samples"
    )


class ConsentRequest(BaseModel):
    user_a_id: str = Field(..., min_length=1, description="Partner A user ID")
    user_b_id: str = Field(..., min_length=1, description="Partner B user ID")
    action: str = Field(
        ..., description="One of: opt_in, revoke, status"
    )
    acting_user: str = Field(
        ..., min_length=1, description="The user performing the action"
    )


# -- Helpers ------------------------------------------------------------------

def _to_domain_samples(inputs: List[EmotionSampleInput]) -> List[EmotionSample]:
    """Convert Pydantic input models to domain dataclass instances."""
    return [
        EmotionSample(timestamp=s.timestamp, valence=s.valence, arousal=s.arousal)
        for s in inputs
    ]


# -- Endpoints ----------------------------------------------------------------

@router.post("/synchrony")
async def compute_synchrony(req: SynchronyRequest):
    """Compute emotional synchrony between two partners.

    Requires an active partnership (both users opted in).
    Returns valence synchrony, arousal synchrony, overall score,
    plus resonance periods and conflict/repair analysis.
    """
    error = _check_partnership_active(req.user_a_id, req.user_b_id)
    if error:
        raise HTTPException(status_code=403, detail=error)

    samples_a = _to_domain_samples(req.samples_a)
    samples_b = _to_domain_samples(req.samples_b)

    sync = compute_emotional_synchrony(samples_a, samples_b)
    resonance = detect_resonance_periods(samples_a, samples_b)
    conflicts = detect_conflict(samples_a, samples_b)
    repairs = detect_repair(conflicts, samples_a, samples_b)

    return {
        "synchrony": sync,
        "resonance": {
            "resonance_ratio": resonance["resonance_ratio"],
            "n_windows": resonance["n_windows"],
            "resonance_periods_count": len(resonance["resonance_periods"]),
            "divergence_periods_count": len(resonance["divergence_periods"]),
        },
        "conflicts": [_conflict_to_dict(c) for c in conflicts],
        "repairs": [_repair_to_dict(r) for r in repairs],
    }


@router.post("/profile")
async def compute_profile(req: ProfileRequest):
    """Compute a full relationship emotional profile.

    Requires an active partnership.  Returns synchrony, resonance,
    contagion, conflict, repair, and overall health classification.
    """
    error = _check_partnership_active(req.user_a_id, req.user_b_id)
    if error:
        raise HTTPException(status_code=403, detail=error)

    samples_a = _to_domain_samples(req.samples_a)
    samples_b = _to_domain_samples(req.samples_b)

    profile = compute_relationship_profile(samples_a, samples_b)
    return {"profile": profile_to_dict(profile)}


@router.post("/consent")
async def manage_consent(req: ConsentRequest):
    """Manage partnership consent: opt-in, revoke, or check status.

    Both partners must opt-in before any shared analysis is allowed.
    Either partner can revoke at any time, which deactivates the
    partnership and wipes all shared emotion data.
    """
    result = manage_partnership_consent(
        user_a_id=req.user_a_id,
        user_b_id=req.user_b_id,
        action=req.action,
        acting_user=req.acting_user,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/status")
async def status():
    """Health check -- confirms the couples resonance module is loaded."""
    return {
        "status": "ready",
        "model_type": "couples-emotional-resonance",
        "description": "Consent-gated shared emotional dashboards for relationship wellness",
    }
