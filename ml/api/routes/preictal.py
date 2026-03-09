"""Pre-ictal seizure prediction API.

GitHub issue: #117

IMPORTANT: NOT a medical device. Research/educational use only.
Pre-ictal prediction from consumer EEG is NOT clinically validated.
"""
import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe
from models.preictal_predictor import PreictalPredictor

router = APIRouter(tags=["preictal"])

_predictor = PreictalPredictor()


@router.post("/preictal/set-baseline")
async def set_preictal_baseline(data: EEGInput):
    """Record resting-state baseline for pre-ictal comparison.

    Call during a known normal (interictal) period. Minimum 0.5 seconds.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    result = _predictor.set_baseline(eeg=signals, fs=data.fs)
    return _numpy_safe(result)


@router.post("/preictal/assess")
async def assess_preictal(data: EEGInput):
    """Assess current pre-ictal risk from an EEG epoch.

    Returns preictal_risk (0-1), risk_level, entropy_trend,
    synchrony_index, feature_changes, alert, and medical disclaimer.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    result = _predictor.assess(eeg=signals, fs=data.fs)
    return _numpy_safe(result)


@router.get("/preictal/timeline")
async def get_preictal_timeline():
    """Get the full timeline of risk assessments for this session."""
    timeline = _predictor.get_risk_timeline()
    return _numpy_safe({"timeline": timeline, "count": len(timeline)})


@router.get("/preictal/stats")
async def get_preictal_stats():
    """Get session-level pre-ictal statistics."""
    result = _predictor.get_session_stats()
    return _numpy_safe(result)


@router.get("/preictal/history")
async def get_preictal_history(last_n: int = 50):
    """Get recent pre-ictal assessment history."""
    history = _predictor.get_history(last_n=last_n)
    return _numpy_safe({"history": history, "count": len(history)})


@router.post("/preictal/reset")
async def reset_preictal():
    """Reset all pre-ictal state (baseline and history)."""
    _predictor.reset()
    return {"status": "ok", "message": "Pre-ictal state cleared."}
