"""Adaptive Music Wellness — ISO Principle Controller (#284).

Endpoints:
  POST /music-therapy/prescribe   — full ISO session recommendation from emotion state
  GET  /music-therapy/parameters  — music parameters for a given valence/arousal

References:
  - de Witte et al. 2020/2025: d=0.723 stress reduction (51 RCTs)
  - Neurophone 2024: EEG → real-time music parameter control
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/music-therapy", tags=["music-therapy"])

_WELLNESS_DISCLAIMER = (
    "This is not a medical device. Music wellness suggestions are based on "
    "published research (ISO Principle) and are for personal wellness only, "
    "not a substitute for licensed music wellness practice or professional "
    "mental health support."
)


class PrescribeRequest(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Current emotional valence -1 to 1")
    arousal: float = Field(0.5, ge=0.0, le=1.0, description="Current emotional arousal 0 to 1")
    target_valence: float = Field(0.4, ge=-1.0, le=1.0, description="Target valence -1 to 1")
    target_arousal: Optional[float] = Field(None, ge=0.0, le=1.0)
    session_min: int = Field(20, ge=5, le=60, description="Session length in minutes")


class ParametersRequest(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)


@router.post("/prescribe")
def prescribe_music_session(req: PrescribeRequest) -> Dict[str, Any]:
    """Recommend an ISO-principle music wellness session from current emotion state.

    Note: 'prescribe' is retained in the API path for backward compatibility.
    This is a wellness recommendation, not a medical prescription. This is not
    a medical device.
    """
    from models.music_mood_engine import get_iso_controller
    controller = get_iso_controller()
    result = controller.prescribe(
        current_valence=req.valence,
        current_arousal=req.arousal,
        target_valence=req.target_valence,
        target_arousal=req.target_arousal,
        session_min=req.session_min,
    )
    result["disclaimer"] = _WELLNESS_DISCLAIMER
    return result


@router.post("/parameters")
def get_parameters(req: ParametersRequest) -> Dict[str, Any]:
    """Return raw music parameters for a given valence/arousal state."""
    from models.music_mood_engine import get_music_parameters
    p = get_music_parameters(req.valence, req.arousal)
    return {
        "tempo_bpm": round(p.tempo_bpm),
        "key": p.key,
        "mode": p.mode,
        "energy_level": round(p.energy, 2),
        "valence_label": p.valence_label,
        "arousal_label": p.arousal_label,
        "search_query": p.search_query,
        "genre_suggestions": p.genre_suggestions,
        "avoid_features": p.avoid_features,
    }
