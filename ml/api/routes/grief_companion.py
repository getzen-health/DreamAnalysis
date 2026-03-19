"""Grief and loss processing companion API routes.

POST /grief/assess       -- assess current grief stage from emotional data
POST /grief/trajectory   -- track grief trajectory over time
GET  /grief/status       -- model availability

Issue #424: Grief and loss processing companion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.grief_companion import (
    GriefReading,
    GriefStage,
    SafetyLevel,
    TrajectoryTrend,
    _CLINICAL_DISCLAIMER,
    _CRISIS_RESOURCES,
    INTERVENTION_LIBRARY,
    check_safety,
    compute_grief_profile,
    detect_anniversary_effect,
    detect_grief_stage,
    profile_to_dict,
    select_support_intervention,
    track_grief_trajectory,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/grief", tags=["grief-companion"])


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class AssessRequest(BaseModel):
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional valence (-1 to 1)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Emotional arousal (0 to 1)")
    stress_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Stress index (0 to 1)")
    anger_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Anger index (0 to 1)")
    focus_index: float = Field(default=0.5, ge=0.0, le=1.0, description="Focus index (0 to 1)")
    isolation_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Isolation index (0 to 1)")
    hopelessness_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Hopelessness index (0 to 1)")
    recent_readings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Previous readings for safety/trajectory context",
    )
    stage_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Previous stage detection results for trajectory tracking",
    )
    significant_dates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Significant dates for anniversary detection",
    )


class TrajectoryRequest(BaseModel):
    readings: List[Dict[str, Any]] = Field(
        ...,
        description="List of previous stage detection results, each with 'stage' and 'timestamp'",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/assess")
async def assess_grief(req: AssessRequest) -> Dict[str, Any]:
    """Assess current grief stage from emotional state data.

    Performs comprehensive grief assessment including stage detection,
    safety check, anniversary effect detection, and intervention
    selection. Returns a full grief profile.
    """
    reading = GriefReading(
        valence=req.valence,
        arousal=req.arousal,
        stress_index=req.stress_index,
        anger_index=req.anger_index,
        focus_index=req.focus_index,
        isolation_index=req.isolation_index,
        hopelessness_index=req.hopelessness_index,
    )

    # Reconstruct recent readings if provided
    recent: List[GriefReading] = []
    if req.recent_readings:
        for r in req.recent_readings:
            recent.append(GriefReading(
                valence=r.get("valence", 0.0),
                arousal=r.get("arousal", 0.0),
                stress_index=r.get("stress_index", 0.0),
                anger_index=r.get("anger_index", 0.0),
                focus_index=r.get("focus_index", 0.5),
                isolation_index=r.get("isolation_index", 0.0),
                hopelessness_index=r.get("hopelessness_index", 0.0),
                timestamp=r.get("timestamp", 0.0),
            ))

    profile = compute_grief_profile(
        current_reading=reading,
        recent_readings=recent if recent else None,
        stage_history=req.stage_history,
        significant_dates=req.significant_dates,
    )

    return profile_to_dict(profile)


@router.post("/trajectory")
async def grief_trajectory(req: TrajectoryRequest) -> Dict[str, Any]:
    """Track grief trajectory over time.

    Analyzes a sequence of grief stage detections to determine whether
    the user is progressing, stuck, regressing, or oscillating between
    stages. Requires at least 3 readings.
    """
    return track_grief_trajectory(req.readings)


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Model availability and grief companion info."""
    stages = [s.value for s in GriefStage if s != GriefStage.UNKNOWN]
    return {
        "status": "ok",
        "model": "grief_companion",
        "version": "1.0.0",
        "grief_stages": stages,
        "total_interventions": len(INTERVENTION_LIBRARY),
        "safety_levels": [s.value for s in SafetyLevel],
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
        "crisis_resources": _CRISIS_RESOURCES,
    }
