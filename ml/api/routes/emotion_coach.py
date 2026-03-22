"""CBT-based AI Emotion Coach API routes.

Endpoints:
  POST /emotion-coach/recommend/{user_id}  -- ranked interventions for current state
  POST /emotion-coach/log-session/{user_id} -- log completed session with before/after
  GET  /emotion-coach/history/{user_id}    -- user coaching history + top techniques
  GET  /emotion-coach/insight/{user_id}    -- personalised insight text
  GET  /emotion-coach/techniques           -- list all available techniques
  GET  /emotion-coach/status               -- health check

GitHub issue: #229
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ._shared import sanitize_id
from models.emotion_coach import EmotionCoach, _TECHNIQUES

router = APIRouter(tags=["Emotion Coach"])

_coach = EmotionCoach()

# ── Request / response models ─────────────────────────────────────────────────


class StateRequest(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Emotional valence (-1 negative to +1 positive)")
    arousal: float = Field(0.5, ge=0.0, le=1.0, description="Arousal level (0 calm to 1 energetic)")
    stress_index: float = Field(0.3, ge=0.0, le=1.0, description="Stress index (0-1)")
    focus_index: float = Field(0.5, ge=0.0, le=1.0, description="Focus index (0-1)")
    hrv_ms: Optional[float] = Field(None, description="Heart-rate variability in milliseconds (optional)")


class SessionLogRequest(BaseModel):
    intervention_name: str = Field(..., description="Name of the technique used")
    state_before: StateRequest = Field(..., description="Emotional state before the session")
    state_after: StateRequest = Field(..., description="Emotional state after the session")
    duration_seconds: int = Field(..., ge=0, description="Actual duration of the session in seconds")


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/emotion-coach/recommend/{user_id}")
async def recommend_interventions(user_id: str, state: StateRequest):
    """Return the top-3 ranked CBT interventions for the user's current emotional state.

    Ranking combines rule-based selection (valence / arousal / stress thresholds)
    with personalised historical effectiveness derived from past sessions.
    No LLM required — purely deterministic logic.
    """
    sanitize_id(user_id, "user_id")
    history_data = _coach.get_history(user_id)
    user_history = history_data.get("sessions", [])[-30:]  # last 30 sessions

    state_dict = {
        "valence": state.valence,
        "arousal": state.arousal,
        "stress_index": state.stress_index,
        "focus_index": state.focus_index,
    }
    if state.hrv_ms is not None:
        state_dict["hrv_ms"] = state.hrv_ms

    interventions = _coach.get_interventions(state_dict, user_history)
    return {
        "user_id": user_id,
        "state": state_dict,
        "interventions": interventions,
        "sessions_used_for_ranking": len(user_history),
    }


@router.post("/emotion-coach/log-session/{user_id}")
async def log_session(user_id: str, body: SessionLogRequest):
    """Log a completed coaching session with before/after emotional state.

    Persists to data/emotion_coach/<user_id>.json.
    Returns effectiveness_delta = stress_after - stress_before
    (negative = technique reduced stress).
    """
    sanitize_id(user_id, "user_id")
    state_before = {
        "valence": body.state_before.valence,
        "arousal": body.state_before.arousal,
        "stress_index": body.state_before.stress_index,
        "focus_index": body.state_before.focus_index,
    }
    state_after = {
        "valence": body.state_after.valence,
        "arousal": body.state_after.arousal,
        "stress_index": body.state_after.stress_index,
        "focus_index": body.state_after.focus_index,
    }
    result = _coach.log_session(
        user_id=user_id,
        intervention=body.intervention_name,
        state_before=state_before,
        state_after=state_after,
    )
    result["duration_seconds"] = body.duration_seconds
    return result


@router.get("/emotion-coach/history/{user_id}")
async def get_history(user_id: str):
    """Return the user's full coaching history and top-performing techniques.

    top_interventions is sorted by avg_stress_delta ascending
    (most negative = most effective).
    """
    sanitize_id(user_id, "user_id")
    return _coach.get_history(user_id)


@router.get("/emotion-coach/insight/{user_id}")
async def get_insight(user_id: str):
    """Return a personalised text insight based on the user's session patterns.

    Example: 'Box breathing reduces your stress by 28% on average across 7 sessions.'
    Requires at least 3 completed sessions for a meaningful insight.
    """
    sanitize_id(user_id, "user_id")
    insight = _coach.get_insight(user_id)
    return {"user_id": user_id, "insight": insight}


@router.get("/emotion-coach/techniques")
async def list_techniques():
    """List all available CBT techniques with full descriptions and base effectiveness scores."""
    return {"techniques": _TECHNIQUES, "count": len(_TECHNIQUES)}


@router.get("/emotion-coach/status")
async def status():
    """Health check — confirms the emotion coach module is loaded and ready."""
    return {
        "status": "ready",
        "techniques_loaded": len(_TECHNIQUES),
        "model_type": "rule-based",
        "llm_required": False,
    }
