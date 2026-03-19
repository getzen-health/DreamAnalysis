"""Emotional granularity engine API routes (#423).

Endpoints:
  POST /granularity/score     -- compute granularity from emotion history
  POST /granularity/exercise  -- generate differentiation exercise
  GET  /granularity/status    -- health check
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.emotional_granularity_engine import EmotionalGranularityEngine

router = APIRouter(prefix="/granularity", tags=["granularity"])

_engine = EmotionalGranularityEngine()

# -- Request / response models ------------------------------------------------


class EmotionEntry(BaseModel):
    label: str = Field(..., description="Emotion label (any taxonomy level)")
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="Intensity 0-1")
    context: Optional[str] = Field(None, description="Optional context")


class GranularityScoreRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    emotion_history: List[EmotionEntry] = Field(
        ..., description="List of emotion reports"
    )


class ExerciseRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    emotion_history: List[EmotionEntry] = Field(
        default_factory=list,
        description="Emotion history for personalisation",
    )
    difficulty: str = Field(
        "auto",
        description="Exercise difficulty: easy, medium, hard, or auto",
    )


# -- Endpoints ----------------------------------------------------------------


@router.post("/score")
async def compute_granularity_score(req: GranularityScoreRequest) -> Dict[str, Any]:
    """Compute the Emotional Granularity Score from a user's emotion history.

    Returns granularity score (0-1), sub-scores (label diversity, taxonomy
    depth, distribution evenness, ICC), and a human-readable level.
    """
    history = [e.model_dump() for e in req.emotion_history]
    score = _engine.compute_granularity_score(history)
    vocab = _engine.compute_emotion_vocabulary(history)
    trend = _engine.track_granularity_trend(req.user_id, history)
    return {
        "user_id": req.user_id,
        "score": score,
        "vocabulary": vocab,
        "trend": {
            "trend_length": trend["trend_length"],
            "improvement": trend["improvement"],
            "improving": trend["improving"],
        },
    }


@router.post("/exercise")
async def generate_exercise(req: ExerciseRequest) -> Dict[str, Any]:
    """Generate a differentiation exercise for the user.

    The exercise asks the user to distinguish between similar emotions,
    helping them build finer-grained emotional vocabulary.
    """
    history = [e.model_dump() for e in req.emotion_history]
    exercise = _engine.generate_differentiation_exercise(
        history, difficulty=req.difficulty
    )
    return {"user_id": req.user_id, "exercise": exercise}


@router.get("/status")
async def granularity_status() -> Dict[str, Any]:
    """Health check for the emotional granularity engine."""
    taxonomy = _engine.get_taxonomy()
    return {
        "status": "ok",
        "model": "EmotionalGranularityEngine",
        "taxonomy_counts": taxonomy["counts"],
    }
