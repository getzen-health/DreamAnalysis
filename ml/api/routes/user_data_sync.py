"""User data sync endpoints — persist mood and food logs as session JSONs.

These endpoints accept mood and food log entries from the frontend and write
them to the sessions directory so that ``SessionRecorder.list_sessions()``
returns a complete activity history and ML models can retrain on all user data.

POST /user-data/mood-log    — persist a mood/feeling log entry
POST /user-data/food-log    — persist a food nutrition log entry
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/user-data", tags=["user-data-sync"])

# Session storage directory (shared with SessionRecorder and voice_checkin)
_SESSIONS_DIR = Path(__file__).parent.parent.parent / "sessions"
_SESSIONS_DIR.mkdir(exist_ok=True)


# ── Schemas ───────────────────────────────────────────────────────────────────


class MoodLogRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    mood_score: float = Field(..., ge=1, le=10, description="Mood score 1-10")
    energy_level: Optional[float] = Field(None, ge=1, le=10, description="Energy level 1-10")
    notes: Optional[str] = Field(None, description="Free-text feeling note")
    emotion: Optional[str] = Field(None, description="Detected/selected emotion label")
    valence: Optional[float] = Field(None, ge=-1, le=1, description="Emotional valence")


class FoodLogRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    total_calories: float = Field(0, ge=0, description="Total calories")
    total_protein_g: float = Field(0, ge=0)
    total_carbs_g: float = Field(0, ge=0)
    total_fat_g: float = Field(0, ge=0)
    total_fiber_g: float = Field(0, ge=0)
    dominant_macro: Optional[str] = Field(None)
    glycemic_impact: Optional[str] = Field(None)
    meal_type: Optional[str] = Field(None, description="breakfast|lunch|dinner|snack")
    summary: Optional[str] = Field(None, description="Meal summary text")
    food_items: Optional[List[Dict[str, Any]]] = Field(None, description="Individual food items")


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/mood-log")
def sync_mood_log(req: MoodLogRequest) -> Dict[str, Any]:
    """Persist a mood/feeling log as a session JSON for history and ML retraining."""
    ts = time.time()
    session_id = str(uuid.uuid4())[:8]

    # Map mood_score (1-10) to valence (-1 to 1)
    valence = req.valence if req.valence is not None else (req.mood_score - 5.5) / 4.5
    # Map mood_score to stress proxy (inverse)
    stress = max(0.0, min(1.0, 1.0 - (req.mood_score - 1) / 9))
    # Energy as arousal proxy
    arousal = (req.energy_level - 1) / 9 if req.energy_level is not None else 0.5

    session_meta: Dict[str, Any] = {
        "session_id": session_id,
        "user_id": req.user_id,
        "session_type": "mood_log",
        "start_time": ts,
        "end_time": ts,
        "status": "completed",
        "metadata": {
            "mood_score": req.mood_score,
            "energy_level": req.energy_level,
            "notes": req.notes,
            "emotion": req.emotion,
        },
        "summary": {
            "duration_sec": 0,
            "n_frames": 0,
            "n_channels": 0,
            "n_samples": 0,
            "avg_stress": stress,
            "avg_focus": 0.5,
            "avg_relaxation": max(0.0, 1.0 - stress),
            "avg_valence": valence,
            "avg_arousal": arousal,
            "dominant_emotion": req.emotion or ("happy" if valence > 0.2 else "neutral" if valence > -0.2 else "sad"),
            "avg_flow": 0.0,
        },
        "analysis_timeline": [],
    }

    try:
        meta_path = _SESSIONS_DIR / f"{session_id}.json"
        with open(meta_path, "w") as f:
            json.dump(session_meta, f, indent=2, default=str)
        log.info("Mood log session saved: %s (user=%s, score=%.1f)", session_id, req.user_id, req.mood_score)
    except Exception as exc:
        log.warning("Failed to persist mood log session: %s", exc)

    return {"status": "ok", "session_id": session_id}


@router.post("/food-log")
def sync_food_log(req: FoodLogRequest) -> Dict[str, Any]:
    """Persist a food/nutrition log as a session JSON for history and ML correlation."""
    ts = time.time()
    session_id = str(uuid.uuid4())[:8]

    session_meta: Dict[str, Any] = {
        "session_id": session_id,
        "user_id": req.user_id,
        "session_type": "food_log",
        "start_time": ts,
        "end_time": ts,
        "status": "completed",
        "metadata": {
            "total_calories": req.total_calories,
            "total_protein_g": req.total_protein_g,
            "total_carbs_g": req.total_carbs_g,
            "total_fat_g": req.total_fat_g,
            "total_fiber_g": req.total_fiber_g,
            "dominant_macro": req.dominant_macro,
            "glycemic_impact": req.glycemic_impact,
            "meal_type": req.meal_type,
            "summary": req.summary,
            "food_items": req.food_items,
        },
        "summary": {
            "duration_sec": 0,
            "n_frames": 0,
            "n_channels": 0,
            "n_samples": 0,
            "avg_stress": 0.0,
            "avg_focus": 0.0,
            "avg_relaxation": 0.0,
            "avg_valence": 0.0,
            "avg_arousal": 0.0,
            "dominant_emotion": "neutral",
            "avg_flow": 0.0,
            "total_calories": req.total_calories,
            "meal_type": req.meal_type,
        },
        "analysis_timeline": [],
    }

    try:
        meta_path = _SESSIONS_DIR / f"{session_id}.json"
        with open(meta_path, "w") as f:
            json.dump(session_meta, f, indent=2, default=str)
        log.info(
            "Food log session saved: %s (user=%s, %d kcal)",
            session_id, req.user_id, int(req.total_calories),
        )
    except Exception as exc:
        log.warning("Failed to persist food log session: %s", exc)

    return {"status": "ok", "session_id": session_id}
