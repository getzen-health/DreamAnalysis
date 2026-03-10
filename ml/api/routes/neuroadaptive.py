"""Neuroadaptive learning system API.

EEG-driven adaptive tutoring — assesses the learner's cognitive state and
recommends difficulty/intervention adjustments in real time.

Endpoints:
  POST /neuroadaptive/assess       -- assess learning zone from live EEG
  POST /neuroadaptive/ack-break    -- acknowledge a recommended break was taken
  GET  /neuroadaptive/summary      -- session summary statistics
  GET  /neuroadaptive/history      -- full assessment history
  POST /neuroadaptive/reset        -- clear state for a user

GitHub issue: #116
"""

import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe, extract_band_powers, preprocess
from models.neuroadaptive_tutor import NeuroadaptiveTutor

router = APIRouter(tags=["neuroadaptive"])

_tutor = NeuroadaptiveTutor()


def _get_band_powers(signals: np.ndarray, fs: float):
    """Extract normalised theta/alpha/beta from multi-channel EEG."""
    if signals.ndim == 2:
        signal = signals[1] if signals.shape[0] > 1 else signals[0]  # prefer AF7
    else:
        signal = signals
    processed = preprocess(signal, fs)
    bp = extract_band_powers(processed, fs)
    total = bp["theta"] + bp["alpha"] + bp["beta"] + 1e-10
    return (
        float(np.clip(bp["theta"] / total, 0, 1)),
        float(np.clip(bp["alpha"] / total, 0, 1)),
        float(np.clip(bp["beta"] / total, 0, 1)),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/neuroadaptive/assess")
async def assess_neuroadaptive(data: EEGInput, session_minutes: float = 0.0,
                               fatigue_index: float = 0.0):
    """Assess learning zone from live EEG and get adaptation recommendations.

    Returns learning_zone (flow/overload/boredom/fatigue/recovery),
    intervention, difficulty_adjustment, difficulty_level, break_recommended,
    engagement_score, zone_confidence, and n_samples.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    theta, alpha, beta = _get_band_powers(signals, data.fs)
    result = _tutor.assess(
        theta_power=theta,
        alpha_power=alpha,
        beta_power=beta,
        fatigue_index=float(fatigue_index),
        session_minutes=float(session_minutes),
        user_id=data.user_id,
    )
    result["user_id"] = data.user_id
    return _numpy_safe(result)


@router.post("/neuroadaptive/ack-break")
async def acknowledge_break(user_id: str):
    """Acknowledge that the user has taken a recommended break.

    Resets the break cooldown counter so break recommendations resume
    appropriately after sufficient activity.
    """
    _tutor.acknowledge_break(user_id=user_id)
    return {"status": "ok", "message": "Break acknowledged.", "user_id": user_id}


@router.get("/neuroadaptive/summary")
async def get_neuroadaptive_summary(user_id: str):
    """Get session summary statistics for a user.

    Returns zone distribution, mean engagement, break count, difficulty
    progression, and dominant learning zone.
    """
    result = _tutor.get_session_summary(user_id=user_id)
    result["user_id"] = user_id
    return _numpy_safe(result)


@router.get("/neuroadaptive/history")
async def get_neuroadaptive_history(user_id: str, last_n: int = 50):
    """Get assessment history for a user."""
    history = _tutor.get_history(user_id=user_id, last_n=last_n)
    return _numpy_safe({"history": history, "user_id": user_id, "count": len(history)})


@router.post("/neuroadaptive/reset")
async def reset_neuroadaptive(user_id: str):
    """Clear assessment history and difficulty state for a user."""
    _tutor.reset(user_id=user_id)
    return {"status": "ok", "message": "Neuroadaptive state cleared.", "user_id": user_id}
