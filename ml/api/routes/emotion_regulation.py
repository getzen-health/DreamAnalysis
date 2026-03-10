"""Closed-loop emotion regulation biofeedback API endpoints.

Endpoints:
    POST /emotion-regulation/analyze  — analyze one EEG epoch
    POST /emotion-regulation/update   — update session with new epoch
    GET  /emotion-regulation/summary  — session summary
    POST /emotion-regulation/reset    — reset session state
"""

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Emotion Regulation"])


# ── Request models ─────────────────────────────────────────────────────────────

class EmotionRegulationAnalyzeRequest(BaseModel):
    eeg_data: List[List[float]] = Field(
        ...,
        description="EEG data: shape [n_channels, n_samples] or [[n_samples]]",
    )
    fs: float = Field(default=256.0, gt=0, description="Sampling rate in Hz")
    duration_sec: int = Field(
        default=0, ge=0, description="Epoch duration in seconds (informational)"
    )
    user_id: str = Field(..., description="User identifier")


class EmotionRegulationUpdateRequest(BaseModel):
    eeg_data: List[List[float]] = Field(
        ...,
        description="EEG data: shape [n_channels, n_samples] or [[n_samples]]",
    )
    fs: float = Field(default=256.0, gt=0, description="Sampling rate in Hz")
    duration_sec: int = Field(
        default=1, ge=0, description="Seconds to add to session clock"
    )
    user_id: str = Field(..., description="User identifier")


class EmotionRegulationResetRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/emotion-regulation/analyze")
async def emotion_regulation_analyze(request: EmotionRegulationAnalyzeRequest):
    """Analyze one EEG epoch and return emotion regulation biofeedback.

    Returns emotion_state, regulation_score (0–1), biofeedback_cue,
    alpha_asymmetry, anxiety_index, regulation_trend, session_duration.
    """
    try:
        from models.emotion_regulation import get_emotion_regulation_biofeedback
        model = get_emotion_regulation_biofeedback(request.user_id)
        eeg = np.array(request.eeg_data, dtype=np.float32)
        return model.predict(eeg, fs=request.fs)
    except Exception as exc:
        logger.exception("emotion_regulation_analyze failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/emotion-regulation/update")
async def emotion_regulation_update(request: EmotionRegulationUpdateRequest):
    """Update session with a new EEG epoch and advance the session clock.

    Returns the same keys as /analyze plus session_count.
    """
    try:
        from models.emotion_regulation import get_emotion_regulation_biofeedback
        model = get_emotion_regulation_biofeedback(request.user_id)
        eeg = np.array(request.eeg_data, dtype=np.float32)
        return model.update_session(
            eeg, fs=request.fs, duration_sec=request.duration_sec
        )
    except Exception as exc:
        logger.exception("emotion_regulation_update failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/emotion-regulation/summary")
async def emotion_regulation_summary(user_id: str):
    """Return session summary statistics.

    Returns mean_regulation_score, peak_score, session_count, dominant_state.
    """
    try:
        from models.emotion_regulation import get_emotion_regulation_biofeedback
        model = get_emotion_regulation_biofeedback(user_id)
        return model.get_session_summary()
    except Exception as exc:
        logger.exception("emotion_regulation_summary failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/emotion-regulation/reset")
async def emotion_regulation_reset(request: EmotionRegulationResetRequest):
    """Reset EMA state and session history for the given user."""
    try:
        from models.emotion_regulation import get_emotion_regulation_biofeedback
        model = get_emotion_regulation_biofeedback(request.user_id)
        model.reset()
        return {"success": True, "message": "Emotion regulation state reset."}
    except Exception as exc:
        logger.exception("emotion_regulation_reset failed")
        raise HTTPException(status_code=500, detail=str(exc))
