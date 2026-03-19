"""Generative neural narrative routes (#415).

Endpoints:
  POST /narrative/generate   -- generate adaptive narrative segment
  POST /narrative/feedback    -- process EEG feedback for next segment
  GET  /narrative/frameworks  -- list available story frameworks
  GET  /narrative/status      -- engine status and profile info
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.narrative_engine import (
    BrainStateCategory,
    EEGFeedback,
    EEGTrend,
    EmotionalArcPhase,
    NarrativeEngine,
)

router = APIRouter(prefix="/narrative", tags=["narrative"])

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_engine = NarrativeEngine()

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EEGFeedbackInput(BaseModel):
    alpha_trend: str = Field(
        default="stable",
        description="Alpha power trend: increasing, stable, or decreasing",
    )
    beta_level: float = Field(
        default=0.3,
        description="Normalised high-beta level 0-1",
    )
    theta_trend: str = Field(
        default="stable",
        description="Theta power trend: increasing, stable, or decreasing",
    )


class GenerateRequest(BaseModel):
    user_id: str = Field(default="anonymous", description="User identifier")
    framework_id: Optional[str] = Field(
        default=None,
        description="Explicit story framework ID (omit for auto-selection)",
    )
    arc_phase: Optional[str] = Field(
        default=None,
        description="Emotional arc phase: opening, building, peak, resolution, integration",
    )
    eeg_feedback: Optional[EEGFeedbackInput] = Field(
        default=None,
        description="Current EEG state for adaptive selection",
    )
    preferred_imagery: Optional[List[str]] = Field(
        default=None,
        description="Imagery preferences: nature, ocean, space, interpersonal",
    )
    trigger_words: Optional[List[str]] = Field(
        default=None,
        description="Additional words to filter from narrative text",
    )


class FeedbackRequest(BaseModel):
    user_id: str = Field(default="anonymous", description="User identifier")
    alpha_trend: str = Field(
        default="stable",
        description="Alpha power trend: increasing, stable, or decreasing",
    )
    beta_level: float = Field(
        default=0.3,
        description="Normalised high-beta level 0-1",
    )
    theta_trend: str = Field(
        default="stable",
        description="Theta power trend: increasing, stable, or decreasing",
    )
    current_arc_phase: str = Field(
        default="opening",
        description="Current phase in emotional arc",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_eeg_trend(value: str) -> EEGTrend:
    """Convert string to EEGTrend enum, defaulting to STABLE."""
    try:
        return EEGTrend(value.lower())
    except (ValueError, AttributeError):
        return EEGTrend.STABLE


def _parse_arc_phase(value: str) -> EmotionalArcPhase:
    """Convert string to EmotionalArcPhase, defaulting to OPENING."""
    try:
        return EmotionalArcPhase(value.lower())
    except (ValueError, AttributeError):
        return EmotionalArcPhase.OPENING


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/generate")
async def generate_narrative(req: GenerateRequest) -> Dict:
    """Generate an adaptive narrative segment.

    Optionally pass eeg_feedback for brain-state-aware framework selection
    and imagery adaptation. On first call for a user_id, a narrative profile
    is created automatically.

    Returns the generated segment, safety status, and clinical disclaimer.
    """
    # Ensure profile exists
    _engine.build_narrative_profile(
        user_id=req.user_id,
        preferred_imagery=req.preferred_imagery,
        trigger_words=req.trigger_words,
    )

    # Build EEGFeedback if provided
    eeg_fb = None
    if req.eeg_feedback is not None:
        eeg_fb = EEGFeedback(
            alpha_trend=_parse_eeg_trend(req.eeg_feedback.alpha_trend),
            beta_level=max(0.0, min(1.0, req.eeg_feedback.beta_level)),
            theta_trend=_parse_eeg_trend(req.eeg_feedback.theta_trend),
        )

    segment = _engine.generate_narrative_segment(
        user_id=req.user_id,
        framework_id=req.framework_id,
        arc_phase=req.arc_phase,
        eeg_feedback=eeg_fb,
    )

    return NarrativeEngine.narrative_to_dict(segment)


@router.post("/feedback")
async def process_eeg_feedback(req: FeedbackRequest) -> Dict:
    """Process EEG feedback and determine the next narrative direction.

    Given the current brain state (alpha/beta/theta trends) and the
    current position in the emotional arc, returns guidance for the
    next segment: advance, hold, ground, or re-engage.
    """
    feedback = EEGFeedback(
        alpha_trend=_parse_eeg_trend(req.alpha_trend),
        beta_level=max(0.0, min(1.0, req.beta_level)),
        theta_trend=_parse_eeg_trend(req.theta_trend),
    )

    current_phase = _parse_arc_phase(req.current_arc_phase)

    result = _engine.adapt_to_eeg_feedback(
        feedback=feedback,
        current_arc_phase=current_phase,
        user_id=req.user_id,
    )

    return result


@router.get("/frameworks")
async def list_frameworks() -> Dict:
    """List all available therapeutic story frameworks.

    Returns framework IDs, names, types, descriptions, available arc
    phases, and any contraindications.
    """
    frameworks = NarrativeEngine.list_frameworks()
    return {
        "frameworks": frameworks,
        "count": len(frameworks),
    }


@router.get("/status")
async def narrative_status() -> Dict:
    """Return engine status and summary.

    Shows number of active profiles, available frameworks, and
    supported imagery types.
    """
    from models.narrative_engine import (
        ImageryPreference,
        FrameworkType,
        _FRAMEWORK_LIBRARY,
    )

    return {
        "status": "ok",
        "engine": "narrative",
        "profiles_loaded": len(_engine._profiles),
        "frameworks_available": len(_FRAMEWORK_LIBRARY),
        "framework_types": [ft.value for ft in FrameworkType],
        "imagery_types": [ip.value for ip in ImageryPreference],
        "arc_phases": [p.value for p in EmotionalArcPhase],
    }
