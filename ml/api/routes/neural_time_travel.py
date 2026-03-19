"""Neural time travel API routes -- replay past emotional states.

POST /time-travel/store     -- store an emotional snapshot
POST /time-travel/replay    -- plan a guided replay session
POST /time-travel/visualize -- generate visualization data
GET  /time-travel/status    -- module availability

Issue #459: Neural time travel.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.neural_time_travel import (
    compute_travel_profile,
    find_similar_states,
    generate_visualization_data,
    plan_replay_session,
    profile_to_dict,
    store_emotional_snapshot,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/time-travel", tags=["neural-time-travel"])


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class StoreSnapshotRequest(BaseModel):
    """Request body for storing an emotional snapshot."""

    user_id: str = Field(..., description="User identifier")
    emotion_label: str = Field(..., description="Emotion name (e.g. happy, calm, sad)")
    feature_vector: List[float] = Field(
        ...,
        description="EEG feature vector (at least 1 element)",
        min_length=1,
    )
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Valence [-1, 1]")
    arousal: float = Field(default=0.0, ge=-1.0, le=1.0, description="Arousal [-1, 1]")
    context: str = Field(default="", description="Free-text context description")
    tags: Optional[List[str]] = Field(default=None, description="Searchable tags")


class ReplayRequest(BaseModel):
    """Request body for planning a replay session."""

    user_id: str = Field(..., description="User identifier")
    current_features: List[float] = Field(
        ...,
        description="User's current EEG feature vector",
        min_length=1,
    )
    target_emotion: Optional[str] = Field(
        default=None,
        description="Target emotion label to search for in stored snapshots",
    )
    target_snapshot_id: Optional[str] = Field(
        default=None,
        description="Specific snapshot ID to replay toward (overrides target_emotion)",
    )
    num_steps: int = Field(default=8, ge=2, le=50, description="Number of replay steps")
    step_duration: float = Field(
        default=30.0, ge=5.0, le=300.0, description="Duration per step in seconds",
    )


class VisualizeRequest(BaseModel):
    """Request body for generating visualization data."""

    user_id: str = Field(..., description="User identifier")
    current_features: List[float] = Field(
        ...,
        description="User's current EEG feature vector",
        min_length=1,
    )
    target_snapshot_id: Optional[str] = Field(
        default=None,
        description="Specific snapshot ID to visualize toward",
    )
    target_emotion: Optional[str] = Field(
        default=None,
        description="Emotion label to find closest snapshot for",
    )
    num_frames: int = Field(default=20, ge=2, le=100, description="Number of animation frames")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_snapshot_by_id(user_id: str, snapshot_id: str):
    """Look up a snapshot by ID from the user's library."""
    from models.neural_time_travel import _get_user_library

    for snap in _get_user_library(user_id):
        if snap.snapshot_id == snapshot_id:
            return snap
    return None


def _resolve_target(user_id: str, current_features: List[float],
                    snapshot_id: Optional[str], emotion_label: Optional[str]):
    """Resolve a target snapshot from either ID or emotion search."""
    if snapshot_id:
        snap = _find_snapshot_by_id(user_id, snapshot_id)
        if snap is None:
            raise HTTPException(status_code=404, detail=f"Snapshot {snapshot_id} not found")
        return snap

    if emotion_label:
        matches = find_similar_states(
            user_id, current_features,
            top_k=1, emotion_filter=emotion_label, min_similarity=-1.0,
        )
        if not matches:
            raise HTTPException(
                status_code=404,
                detail=f"No stored snapshots matching emotion '{emotion_label}' for user '{user_id}'",
            )
        return matches[0][0]

    # No target specified -- pick most recent snapshot
    from models.neural_time_travel import _get_user_library
    lib = _get_user_library(user_id)
    if not lib:
        raise HTTPException(
            status_code=404,
            detail=f"No stored snapshots for user '{user_id}'",
        )
    return lib[-1]


def _snapshot_to_dict(snap) -> Dict[str, Any]:
    """Serialize a snapshot for JSON response."""
    return {
        "snapshot_id": snap.snapshot_id,
        "user_id": snap.user_id,
        "emotion_label": snap.emotion_label,
        "valence": snap.valence,
        "arousal": snap.arousal,
        "feature_vector": snap.feature_vector,
        "context": snap.context,
        "tags": snap.tags,
        "timestamp": snap.timestamp,
    }


def _replay_session_to_dict(session) -> Dict[str, Any]:
    """Serialize a replay session for JSON response."""
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "source_label": session.source_label,
        "target_snapshot_id": session.target_snapshot_id,
        "target_emotion": session.target_emotion,
        "total_duration_seconds": session.total_duration_seconds,
        "estimated_difficulty": session.estimated_difficulty,
        "num_steps": len(session.steps),
        "steps": [
            {
                "step_number": s.step_number,
                "total_steps": s.total_steps,
                "instruction": s.instruction,
                "duration_seconds": s.duration_seconds,
                "progress_fraction": s.progress_fraction,
                "target": {
                    "target_alpha_power": s.target.target_alpha_power,
                    "target_theta_power": s.target.target_theta_power,
                    "target_beta_power": s.target.target_beta_power,
                    "target_frontal_asymmetry": s.target.target_frontal_asymmetry,
                    "target_valence": s.target.target_valence,
                    "target_arousal": s.target.target_arousal,
                    "difficulty": s.target.difficulty,
                },
            }
            for s in session.steps
        ],
        "created_at": session.created_at,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
async def time_travel_status() -> Dict[str, Any]:
    """Return availability status for the neural time travel module."""
    return {
        "available": True,
        "model": "neural_time_travel",
        "version": "1.0.0",
        "capabilities": ["store", "replay", "visualize", "profile"],
        "timestamp": time.time(),
    }


@router.post("/store")
async def store_snapshot(req: StoreSnapshotRequest) -> Dict[str, Any]:
    """Store an emotional state snapshot for later replay.

    Returns the created snapshot with its generated ID.
    """
    snap = store_emotional_snapshot(
        user_id=req.user_id,
        emotion_label=req.emotion_label,
        feature_vector=req.feature_vector,
        valence=req.valence,
        arousal=req.arousal,
        context=req.context,
        tags=req.tags,
    )
    return {
        "snapshot": _snapshot_to_dict(snap),
        "library_size": len(
            __import__("models.neural_time_travel", fromlist=["_get_user_library"])
            ._get_user_library(req.user_id)
        ),
        "timestamp": time.time(),
    }


@router.post("/replay")
async def replay(req: ReplayRequest) -> Dict[str, Any]:
    """Plan a guided replay session toward a stored emotional state.

    Resolves the target via snapshot_id, target_emotion search, or most
    recent snapshot. Returns a step-by-step neurofeedback session plan.
    """
    target_snap = _resolve_target(
        req.user_id, req.current_features,
        req.target_snapshot_id, req.target_emotion,
    )

    session = plan_replay_session(
        user_id=req.user_id,
        current_features=req.current_features,
        target_snapshot=target_snap,
        num_steps=req.num_steps,
        step_duration=req.step_duration,
    )

    profile = compute_travel_profile(req.current_features, target_snap)

    return {
        "session": _replay_session_to_dict(session),
        "travel_profile": profile_to_dict(profile),
        "timestamp": time.time(),
    }


@router.post("/visualize")
async def visualize(req: VisualizeRequest) -> Dict[str, Any]:
    """Generate visualization data for an emotional time-travel journey.

    Returns frame-by-frame interpolation data suitable for rendering
    an animated emotional trajectory.
    """
    target_snap = _resolve_target(
        req.user_id, req.current_features,
        req.target_snapshot_id, req.target_emotion,
    )

    viz_data = generate_visualization_data(
        user_id=req.user_id,
        current_features=req.current_features,
        target_snapshot=target_snap,
        num_frames=req.num_frames,
    )

    return viz_data
