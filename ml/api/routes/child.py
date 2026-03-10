"""Child/Adolescent EI endpoints — #286.

Endpoints:
  POST /child/ei-score    — age-adapted EI composite score
  GET  /child/age-config  — get age-band config (session length, calibration, etc.)
  GET  /child/badges      — list all available achievement badges

Note: COPPA compliance (parental consent flow, data deletion) must be
enforced at the Express/auth layer before these endpoints are called.
Children's data should never be stored without explicit parental consent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/child", tags=["child-ei"])

ALL_BADGES = [
    {"id": "5-day streak", "description": "5 check-ins in a row", "icon": "🔥"},
    {"id": "EI Explorer", "description": "10 total sessions", "icon": "🧭"},
    {"id": "Emotion Master", "description": "30 total sessions", "icon": "⭐"},
    {"id": "High EI", "description": "EI composite ≥ 75", "icon": "🏆"},
    {"id": "Calm Champion", "description": "Stress management ≥ 80", "icon": "🧘"},
    {"id": "Empathy Star", "description": "Interpersonal score ≥ 80", "icon": "❤️"},
    {"id": "Feeling Smart!", "description": "Young learner with high EI", "icon": "🌟"},
]


class ChildEIRequest(BaseModel):
    age: int = Field(..., ge=6, le=17, description="Child's age in years")
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)
    stress_index: float = Field(0.3, ge=0.0, le=1.0)
    focus_index: float = Field(0.5, ge=0.0, le=1.0)
    emotion: str = Field("neutral")
    session_count: int = Field(1, ge=1)
    peer_interaction_rating: Optional[float] = Field(None, ge=0.0, le=1.0,
        description="Parent-reported peer interaction quality 0-1")
    emotion_labeling_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0,
        description="Emotion labeling game accuracy 0-1")


@router.post("/ei-score")
def compute_child_ei(req: ChildEIRequest) -> Dict[str, Any]:
    """Compute age-adapted EI composite score for a child/adolescent."""
    from models.child_ei import get_child_ei_scorer
    scorer = get_child_ei_scorer()
    return scorer.score(
        age=req.age,
        valence=req.valence,
        arousal=req.arousal,
        stress_index=req.stress_index,
        focus_index=req.focus_index,
        emotion=req.emotion,
        session_count=req.session_count,
        peer_interaction_rating=req.peer_interaction_rating,
        emotion_labeling_accuracy=req.emotion_labeling_accuracy,
    )


@router.get("/age-config/{age}")
def get_age_config(age: int) -> Dict[str, Any]:
    """Return session configuration for a child's age."""
    from models.child_ei import (
        get_age_band, EMOTION_VOCABULARY,
        MAX_SESSION_SECONDS, CALIBRATION_FRAMES, F0_RANGES
    )
    try:
        band = get_age_band(age)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(422, str(e))

    return {
        "age": age,
        "age_band": band,
        "emotions_available": EMOTION_VOCABULARY[band],
        "max_session_seconds": MAX_SESSION_SECONDS[band],
        "calibration_frames_needed": CALIBRATION_FRAMES[band],
        "f0_range_hz": {"min": F0_RANGES[band][0], "max": F0_RANGES[band][1]},
        "ui_style": {
            "young": "game-based with visual emotion cards",
            "middle": "reflective check-ins with short journal prompts",
            "teen": "full EI journal with data insights",
        }[band],
        "coppa_notes": (
            "All data collection requires verifiable parental consent (COPPA §312.5). "
            "Children under 13 must have parent account linked before any data is stored."
        ),
    }


@router.get("/badges")
def list_badges() -> List[Dict[str, str]]:
    """Return all available achievement badges."""
    return ALL_BADGES
