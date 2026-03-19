"""Embodied AI emotional companion API routes.

POST /companion/respond   -- generate EEG-adapted companion response
POST /companion/session   -- track session data (themes, shifts, interventions)
GET  /companion/status    -- model availability and companion info

Issue #457: Embodied AI emotional companion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.embodied_companion import (
    ConversationState,
    EmotionalTone,
    TherapeuticStance,
    _CLINICAL_DISCLAIMER,
    adapt_response_to_eeg,
    compute_companion_profile,
    detect_conversation_state,
    EEGState,
    generate_response_template,
    profile_to_dict,
    select_therapeutic_stance,
    track_session,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/companion", tags=["embodied-companion"])


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class RespondRequest(BaseModel):
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional valence (-1 to 1)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Emotional arousal (0 to 1)")
    stress_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Stress index (0 to 1)")
    focus_index: float = Field(default=0.5, ge=0.0, le=1.0, description="Focus index (0 to 1)")
    anger_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Anger index (0 to 1)")
    relaxation_index: float = Field(default=0.5, ge=0.0, le=1.0, description="Relaxation index (0 to 1)")
    conversation_state: str = Field(
        default="greeting",
        description="Current conversation state",
    )
    turn_count: int = Field(default=0, ge=0, description="Number of conversation turns")
    session_duration: float = Field(
        default=0.0, ge=0.0, description="Elapsed session time in seconds"
    )
    themes: Optional[List[str]] = Field(
        default=None, description="Identified conversation themes"
    )
    session_memory: Optional[Dict[str, Any]] = Field(
        default=None, description="Current session memory from previous calls"
    )


class SessionTrackRequest(BaseModel):
    valence: float = Field(..., ge=-1.0, le=1.0, description="Current valence")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Current arousal")
    stress_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Current stress")
    focus_index: float = Field(default=0.5, ge=0.0, le=1.0, description="Current focus")
    anger_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Current anger")
    relaxation_index: float = Field(default=0.5, ge=0.0, le=1.0, description="Current relaxation")
    session_memory: Optional[Dict[str, Any]] = Field(
        default=None, description="Existing session memory to update"
    )
    theme: Optional[str] = Field(default=None, description="Conversation theme to record")
    intervention: Optional[str] = Field(
        default=None, description="Intervention name being tried"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/respond")
async def respond(req: RespondRequest) -> Dict[str, Any]:
    """Generate an EEG-adapted companion response.

    Computes a complete companion response profile including conversation
    state management, therapeutic stance selection, EEG adaptation, and
    response template generation. Returns everything the companion needs
    to deliver an emotionally-attuned response.
    """
    eeg = EEGState(
        valence=req.valence,
        arousal=req.arousal,
        stress_index=req.stress_index,
        focus_index=req.focus_index,
        anger_index=req.anger_index,
        relaxation_index=req.relaxation_index,
    )

    profile = compute_companion_profile(
        eeg=eeg,
        conversation_state=req.conversation_state,
        turn_count=req.turn_count,
        session_duration=req.session_duration,
        themes=req.themes,
        session_memory=req.session_memory,
    )

    return profile_to_dict(profile)


@router.post("/session")
async def session_track(req: SessionTrackRequest) -> Dict[str, Any]:
    """Track session data: emotional shifts, themes, interventions.

    Updates the session memory with the current EEG reading, detects
    emotional shifts, and records any themes or interventions. Returns
    the updated session memory for use in subsequent calls.
    """
    eeg = EEGState(
        valence=req.valence,
        arousal=req.arousal,
        stress_index=req.stress_index,
        focus_index=req.focus_index,
        anger_index=req.anger_index,
        relaxation_index=req.relaxation_index,
    )

    memory = req.session_memory or {}
    updated = track_session(
        session_memory=memory,
        eeg=eeg,
        theme=req.theme,
        intervention=req.intervention,
    )

    return updated


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Model availability and embodied companion info."""
    return {
        "status": "ok",
        "model": "embodied_companion",
        "version": "1.0.0",
        "conversation_states": [s.value for s in ConversationState],
        "therapeutic_stances": [s.value for s in TherapeuticStance],
        "emotional_tones": [t.value for t in EmotionalTone],
        "clinical_disclaimer": _CLINICAL_DISCLAIMER,
    }
