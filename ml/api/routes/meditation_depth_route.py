"""Objective meditation depth quantification API.

GitHub issue: #124
"""
import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe
from models.meditation_depth import MeditationDepthQuantifier

router = APIRouter(tags=["meditation-depth"])

_quantifier = MeditationDepthQuantifier()


@router.post("/meditation-depth/set-baseline")
async def set_meditation_baseline(data: EEGInput):
    """Record pre-meditation resting baseline.

    Call before the user begins meditating.
    Enables more accurate depth quantification via relative changes.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    result = _quantifier.set_baseline(eeg_signals=signals, fs=data.fs, user_id=data.user_id)
    result["user_id"] = data.user_id
    return _numpy_safe(result)


@router.post("/meditation-depth/assess")
async def assess_meditation_depth(data: EEGInput):
    """Assess current meditation depth from an EEG epoch.

    Returns depth_score (0-100), depth_level (surface/light/moderate/deep/transcendent),
    fmt_power, alpha_coherence, theta_alpha_ratio, gamma_bursts_detected,
    stability_index, and recommendations.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    result = _quantifier.assess(eeg_signals=signals, fs=data.fs, user_id=data.user_id)
    result["user_id"] = data.user_id
    return _numpy_safe(result)


@router.get("/meditation-depth/timeline")
async def get_meditation_timeline(user_id: str = "default"):
    """Get meditation depth timeline for the current session."""
    timeline = _quantifier.get_session_timeline(user_id=user_id)
    return _numpy_safe({"timeline": timeline, "user_id": user_id, "count": len(timeline)})


@router.get("/meditation-depth/stats")
async def get_meditation_stats(user_id: str = "default"):
    """Get session-level meditation statistics."""
    result = _quantifier.get_session_stats(user_id=user_id)
    result["user_id"] = user_id
    return _numpy_safe(result)


@router.get("/meditation-depth/history")
async def get_meditation_history(user_id: str = "default", last_n: int = 50):
    """Get meditation depth assessment history for a user."""
    history = _quantifier.get_history(user_id=user_id, last_n=last_n)
    return _numpy_safe({"history": history, "user_id": user_id, "count": len(history)})


@router.post("/meditation-depth/reset")
async def reset_meditation_depth(user_id: str = "default"):
    """Clear all meditation depth state for a user."""
    _quantifier.reset(user_id=user_id)
    return {"status": "ok", "message": "Meditation depth state cleared.", "user_id": user_id}
