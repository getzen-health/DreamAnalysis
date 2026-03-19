"""Emotion-aware adaptive education API routes.

POST /education/learning-state  -- detect current learning state from EEG features
POST /education/adapt           -- get difficulty/pacing recommendations
POST /education/profile         -- compute full education profile from session data
GET  /education/status          -- model availability

GitHub issue: #450
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.adaptive_education import (
    LEARNING_STATES,
    DIFFICULTY_ACTIONS,
    PACING_ACTIONS,
    LearningEEGFeatures,
    LearningState,
    EducationProfile,
    detect_learning_state,
    recommend_difficulty_adjustment,
    recommend_pacing,
    track_attention_span,
    compute_learning_windows,
    compute_education_profile,
    profile_to_dict,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/education", tags=["adaptive-education"])


# ---------------------------------------------------------------------------
# Pydantic request schemas
# ---------------------------------------------------------------------------


class LearningStateRequest(BaseModel):
    theta: float = Field(..., ge=0.0, le=1.0, description="Theta band power (0-1)")
    alpha: float = Field(..., ge=0.0, le=1.0, description="Alpha band power (0-1)")
    beta: float = Field(..., ge=0.0, le=1.0, description="Beta band power (0-1)")
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Emotional valence (-1 to 1)")
    fatigue: float = Field(default=0.0, ge=0.0, le=1.0, description="Fatigue index (0-1)")


class AdaptRequest(BaseModel):
    theta: float = Field(..., ge=0.0, le=1.0, description="Theta band power (0-1)")
    alpha: float = Field(..., ge=0.0, le=1.0, description="Alpha band power (0-1)")
    beta: float = Field(..., ge=0.0, le=1.0, description="Beta band power (0-1)")
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Emotional valence (-1 to 1)")
    fatigue: float = Field(default=0.0, ge=0.0, le=1.0, description="Fatigue index (0-1)")
    current_difficulty: float = Field(default=0.5, ge=0.0, le=1.0, description="Current difficulty level (0-1)")
    session_minutes: float = Field(default=0.0, ge=0.0, description="Minutes elapsed in session")
    consecutive_same_state: int = Field(default=0, ge=0, description="Consecutive readings in same state")


class EEGFeaturesItem(BaseModel):
    theta: float = Field(..., ge=0.0, le=1.0)
    alpha: float = Field(..., ge=0.0, le=1.0)
    beta: float = Field(..., ge=0.0, le=1.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    fatigue: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: float = Field(default=0.0, ge=0.0)


class ProfileRequest(BaseModel):
    features_history: List[EEGFeaturesItem] = Field(
        ..., min_length=1, description="List of EEG feature snapshots from the session"
    )
    hourly_engagement: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Historical hourly engagement data: hour (0-23) -> list of engagement scores",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/learning-state")
async def detect_state(req: LearningStateRequest) -> Dict[str, Any]:
    """Detect current learning state from EEG band power features.

    Returns the detected state (engaged, confused, bored, frustrated, flow),
    confidence, per-state scores, and recommendations for difficulty
    and pacing adjustment.
    """
    features = LearningEEGFeatures(
        theta=req.theta,
        alpha=req.alpha,
        beta=req.beta,
        valence=req.valence,
        fatigue=req.fatigue,
    )
    state = detect_learning_state(features)
    return {
        "state": state.state,
        "confidence": state.confidence,
        "scores": state.scores,
        "difficulty_recommendation": state.difficulty_recommendation,
        "pacing_recommendation": state.pacing_recommendation,
        "engagement_level": state.engagement_level,
        "attention_declining": state.attention_declining,
    }


@router.post("/adapt")
async def adapt(req: AdaptRequest) -> Dict[str, Any]:
    """Get difficulty and pacing recommendations for current EEG state.

    Combines learning state detection with difficulty adjustment and
    pacing recommendation into a single response.
    """
    features = LearningEEGFeatures(
        theta=req.theta,
        alpha=req.alpha,
        beta=req.beta,
        valence=req.valence,
        fatigue=req.fatigue,
    )
    state = detect_learning_state(features)
    difficulty = recommend_difficulty_adjustment(state, req.current_difficulty)
    pacing = recommend_pacing(state, req.session_minutes, req.consecutive_same_state)

    return {
        "learning_state": state.state,
        "confidence": state.confidence,
        "engagement_level": state.engagement_level,
        "difficulty": difficulty,
        "pacing": pacing,
    }


@router.post("/profile")
async def compute_profile(req: ProfileRequest) -> Dict[str, Any]:
    """Compute a full education profile from session EEG data.

    Analyzes a sequence of EEG feature snapshots to produce a comprehensive
    profile including attention span, state distribution, effectiveness
    correlation, and actionable recommendations.
    """
    features_history = [
        LearningEEGFeatures(
            theta=item.theta,
            alpha=item.alpha,
            beta=item.beta,
            valence=item.valence,
            fatigue=item.fatigue,
            timestamp=item.timestamp,
        )
        for item in req.features_history
    ]

    # Convert hourly_engagement keys from string to int if provided
    hourly_eng: Optional[Dict[int, List[float]]] = None
    if req.hourly_engagement:
        hourly_eng = {int(k): v for k, v in req.hourly_engagement.items()}

    profile = compute_education_profile(features_history, hourly_eng)
    return profile_to_dict(profile)


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Model availability and adaptive education info."""
    return {
        "status": "ok",
        "model": "adaptive_education",
        "version": "1.0.0",
        "learning_states": LEARNING_STATES,
        "difficulty_actions": DIFFICULTY_ACTIONS,
        "pacing_actions": PACING_ACTIONS,
        "description": "Emotion-aware adaptive education: EEG-driven learning state detection, difficulty adaptation, and pacing recommendations.",
    }
