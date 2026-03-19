"""Emotional first-aid protocol API routes.

POST /first-aid/detect          -- detect crisis type from current state
POST /first-aid/start           -- start a protocol for detected crisis
POST /first-aid/evaluate-step   -- evaluate current step with EEG data
GET  /first-aid/protocols       -- list all available protocols
GET  /first-aid/status          -- model availability

Issue #438: Emotional first-aid protocols.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.emotional_first_aid import (
    CrisisState,
    CrisisType,
    PROTOCOL_LIBRARY,
    _CLINICAL_DISCLAIMER,
    advance_or_repeat,
    detect_crisis_type,
    get_current_step,
    select_protocol,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/first-aid", tags=["emotional-first-aid"])


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class DetectRequest(BaseModel):
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional valence (-1 to 1)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Emotional arousal (0 to 1)")
    stress_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Stress index (0 to 1)")
    anger_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Anger index (0 to 1)")
    focus_index: float = Field(default=0.5, ge=0.0, le=1.0, description="Focus index (0 to 1)")


class StartProtocolRequest(BaseModel):
    crisis_type: str = Field(..., description="Crisis type from detect endpoint")
    severity: float = Field(default=0.5, ge=0.0, le=1.0, description="Crisis severity (0 to 1)")


class EvaluateStepRequest(BaseModel):
    protocol_id: str = Field(..., description="Active protocol ID")
    current_step: int = Field(..., ge=1, description="Current step number (1-indexed)")
    arousal_before: float = Field(..., ge=0.0, le=1.0, description="Arousal before the step")
    arousal_after: float = Field(..., ge=0.0, le=1.0, description="Arousal after the step")
    stress_before: float = Field(..., ge=0.0, le=1.0, description="Stress before the step")
    stress_after: float = Field(..., ge=0.0, le=1.0, description="Stress after the step")
    repeat_count: int = Field(default=0, ge=0, description="How many times this step has been repeated")
    max_repeats: int = Field(default=2, ge=0, description="Maximum repeats before forced advance")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/detect")
async def detect_crisis(req: DetectRequest) -> Dict[str, Any]:
    """Detect crisis type from current emotional state.

    Analyzes valence, arousal, stress, and anger indices to determine
    if the user is experiencing a panic attack, acute stress, dissociation,
    or rage episode. Returns crisis type, severity, and safety guidance.
    """
    state = CrisisState(
        valence=req.valence,
        arousal=req.arousal,
        stress_index=req.stress_index,
        anger_index=req.anger_index,
        focus_index=req.focus_index,
    )
    return detect_crisis_type(state)


@router.post("/start")
async def start_protocol(req: StartProtocolRequest) -> Dict[str, Any]:
    """Start a de-escalation protocol for a detected crisis.

    Selects the best-matching protocol for the crisis type and severity,
    then returns the full protocol with its first step ready to begin.
    """
    try:
        crisis = CrisisType(req.crisis_type)
    except ValueError:
        valid = [ct.value for ct in CrisisType]
        return {"error": f"Unknown crisis type: {req.crisis_type}. Valid: {valid}"}

    result = select_protocol(crisis, req.severity)

    # If a protocol was selected, also include the first step
    if result.get("selected"):
        protocol_id = result["protocol"]["id"]
        first_step = get_current_step(protocol_id, 1)
        result["first_step"] = first_step

    return result


@router.post("/evaluate-step")
async def evaluate_step(req: EvaluateStepRequest) -> Dict[str, Any]:
    """Evaluate current step effectiveness and decide next action.

    Compares arousal and stress before/after the step to determine if
    brain metrics improved. Advances to the next step if effective,
    repeats if not (up to max_repeats).
    """
    return advance_or_repeat(
        protocol_id=req.protocol_id,
        current_step=req.current_step,
        arousal_before=req.arousal_before,
        arousal_after=req.arousal_after,
        stress_before=req.stress_before,
        stress_after=req.stress_after,
        max_repeats=req.max_repeats,
        repeat_count=req.repeat_count,
    )


@router.get("/protocols")
async def list_protocols() -> Dict[str, Any]:
    """List all available emotional first-aid protocols.

    Returns the full protocol library organized by category, with
    step counts and severity ranges for each protocol.
    """
    by_category: Dict[str, List[Dict]] = {}
    for protocol in PROTOCOL_LIBRARY:
        cat = protocol.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(protocol.to_dict())

    return {
        "total_protocols": len(PROTOCOL_LIBRARY),
        "categories": list(by_category.keys()),
        "protocols_by_category": by_category,
        "all_protocols": [p.to_dict() for p in PROTOCOL_LIBRARY],
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Model availability and protocol library info."""
    return {
        "status": "ok",
        "model": "emotional_first_aid",
        "version": "1.0.0",
        "total_protocols": len(PROTOCOL_LIBRARY),
        "categories": [cat.value for cat in set(p.category for p in PROTOCOL_LIBRARY)],
        "crisis_types": [ct.value for ct in CrisisType if ct != CrisisType.NONE],
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }
