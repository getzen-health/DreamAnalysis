"""Micro-intervention JIT API routes.

POST /interventions/jit/check-trigger    -- check if current state warrants intervention
POST /interventions/jit/select           -- select best intervention for current state
POST /interventions/jit/record-outcome   -- record intervention effectiveness
GET  /interventions/jit/stats/{user_id}  -- get intervention effectiveness stats
GET  /interventions/jit/status           -- model availability

Issue #435: Just-in-time micro-intervention engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.micro_intervention import (
    EmotionState,
    InterventionEngine,
    INTERVENTION_LIBRARY,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interventions/jit", tags=["micro-intervention"])

# Module-level engine instance (shared across requests)
_engine = InterventionEngine()


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class CheckTriggerRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional valence (-1 to 1)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Emotional arousal (0 to 1)")
    stress_index: float = Field(default=0.0, ge=0.0, le=1.0)
    focus_index: float = Field(default=0.5, ge=0.0, le=1.0)
    flow_index: float = Field(default=0.0, ge=0.0, le=1.0)
    context: str = Field(default="", description="Current activity context")
    upcoming_meeting_minutes: Optional[float] = Field(
        default=None, description="Minutes until next meeting (None if no meeting)"
    )


class SelectRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    stress_index: float = Field(default=0.0, ge=0.0, le=1.0)
    focus_index: float = Field(default=0.5, ge=0.0, le=1.0)
    flow_index: float = Field(default=0.0, ge=0.0, le=1.0)
    trigger_type: str = Field(..., description="Trigger type that fired")
    hour_of_day: Optional[float] = Field(
        default=None, ge=0.0, le=24.0,
        description="Current hour of day (0-24) for circadian-aware selection"
    )
    context: str = Field(default="")


class RecordOutcomeRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    intervention_id: str = Field(..., description="ID of the intervention that was delivered")
    trigger_type: str = Field(..., description="Trigger type that caused the intervention")
    valence_before: float = Field(..., ge=-1.0, le=1.0)
    arousal_before: float = Field(..., ge=0.0, le=1.0)
    valence_after: float = Field(..., ge=-1.0, le=1.0)
    arousal_after: float = Field(..., ge=0.0, le=1.0)
    stress_before: float = Field(default=0.0, ge=0.0, le=1.0)
    stress_after: float = Field(default=0.0, ge=0.0, le=1.0)
    felt_helpful: Optional[bool] = Field(default=None)
    hour_of_day: Optional[float] = Field(default=None, ge=0.0, le=24.0)
    context: str = Field(default="")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/check-trigger")
async def check_trigger(req: CheckTriggerRequest) -> Dict[str, Any]:
    """Check if the current emotion state warrants a micro-intervention.

    Call every 30 seconds from the frontend.  Returns whether a trigger
    fired and whether it was suppressed by cooldown / flow / daily cap.
    """
    state = EmotionState(
        valence=req.valence,
        arousal=req.arousal,
        stress_index=req.stress_index,
        focus_index=req.focus_index,
        flow_index=req.flow_index,
        context=req.context,
    )
    return _engine.check_trigger(
        user_id=req.user_id,
        state=state,
        upcoming_meeting_minutes=req.upcoming_meeting_minutes,
    )


@router.post("/select")
async def select_intervention(req: SelectRequest) -> Dict[str, Any]:
    """Select the best micro-intervention for the current state and trigger.

    Uses per-user effectiveness history and circadian phase to personalize
    selection.  Falls back to evidence-based defaults when no history exists.
    """
    state = EmotionState(
        valence=req.valence,
        arousal=req.arousal,
        stress_index=req.stress_index,
        focus_index=req.focus_index,
        flow_index=req.flow_index,
        context=req.context,
    )
    return _engine.select_intervention(
        user_id=req.user_id,
        state=state,
        trigger_type=req.trigger_type,
        hour_of_day=req.hour_of_day,
    )


@router.post("/record-outcome")
async def record_outcome(req: RecordOutcomeRequest) -> Dict[str, Any]:
    """Record intervention effectiveness.

    Call 5 minutes after intervention delivery with before/after emotion
    readings.  This data feeds the personalized learning loop.
    """
    return _engine.record_outcome(
        user_id=req.user_id,
        intervention_id=req.intervention_id,
        trigger_type=req.trigger_type,
        valence_before=req.valence_before,
        arousal_before=req.arousal_before,
        valence_after=req.valence_after,
        arousal_after=req.arousal_after,
        stress_before=req.stress_before,
        stress_after=req.stress_after,
        felt_helpful=req.felt_helpful,
        hour_of_day=req.hour_of_day,
        context=req.context,
    )


@router.get("/stats/{user_id}")
async def get_stats(user_id: str) -> Dict[str, Any]:
    """Get intervention stats and effectiveness breakdown for a user.

    Returns high-level counts plus per-intervention, per-phase, and
    per-trigger effectiveness breakdowns.
    """
    stats = _engine.get_intervention_stats(user_id)
    effectiveness = _engine.compute_effectiveness(user_id)
    return {**stats, "effectiveness": effectiveness}


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Model availability and library info."""
    return {
        "status": "ok",
        "model": "micro_intervention_engine",
        "version": "1.0.0",
        "library_size": len(INTERVENTION_LIBRARY),
        "categories": ["breathing", "grounding", "movement", "cognitive"],
        "engine_summary": _engine.engine_to_dict(),
    }
