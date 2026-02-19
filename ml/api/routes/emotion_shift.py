"""Emotional shift detection endpoints — pre-conscious awareness."""

import numpy as np
from fastapi import APIRouter

from ._shared import (
    _numpy_safe,
    _emotion_shift_detectors,
    EmotionShiftDetector,
    emotion_model,
    EmotionShiftRequest,
)
import api.routes._shared as _state

router = APIRouter()


@router.post("/emotion-shift/detect")
async def detect_emotion_shift(req: EmotionShiftRequest):
    """Feed EEG data and detect pre-conscious emotional shifts."""
    if req.user_id not in _state._emotion_shift_detectors:
        _state._emotion_shift_detectors[req.user_id] = EmotionShiftDetector(fs=req.fs)

    detector = _state._emotion_shift_detectors[req.user_id]
    signals = np.array(req.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    emotion_pred = emotion_model.predict(eeg, req.fs)
    return _numpy_safe(detector.update(eeg, emotion_pred))


@router.get("/emotion-shift/summary/{user_id}")
async def emotion_shift_summary(user_id: str):
    """Get session summary of all emotional shifts detected."""
    detector = _state._emotion_shift_detectors.get(user_id)
    if detector is None:
        return {"total_shifts": 0, "message": "No active session for this user"}
    return _numpy_safe(detector.get_session_summary())


@router.get("/emotion-shift/awareness-score/{user_id}")
async def emotion_awareness_score(user_id: str):
    """Get emotional awareness score for the session."""
    detector = _state._emotion_shift_detectors.get(user_id)
    if detector is None:
        return {"awareness_score": 0, "level": "Beginning", "message": "Start a session first."}
    return _numpy_safe(detector.get_emotional_awareness_score())


@router.post("/emotion-shift/reset/{user_id}")
async def reset_emotion_shift(user_id: str):
    """Reset the emotion shift detector for a new session."""
    if user_id in _state._emotion_shift_detectors:
        del _state._emotion_shift_detectors[user_id]
    return {"status": "reset", "user_id": user_id}
