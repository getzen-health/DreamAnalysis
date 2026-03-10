"""Student engagement detection API.

Endpoints:
  POST /engagement/baseline  -- set per-user focused-state baseline
  POST /engagement/assess    -- assess engagement from live EEG
  GET  /engagement/curve     -- engagement time-series for a user
  GET  /engagement/summary   -- session summary statistics
  POST /engagement/reset     -- clear state for a user

GitHub issue: #114
"""

import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe, extract_band_powers, preprocess
from models.engagement_detector import EngagementDetector

router = APIRouter(tags=["engagement"])

_detector = EngagementDetector()


def _get_band_powers(signals: np.ndarray, fs: float):
    """Extract normalised theta/alpha/beta/gamma from multi-channel EEG."""
    if signals.ndim == 2:
        signal = signals[1] if signals.shape[0] > 1 else signals[0]  # prefer AF7
    else:
        signal = signals
    processed = preprocess(signal, fs)
    bp = extract_band_powers(processed, fs)
    total = bp["theta"] + bp["alpha"] + bp["beta"] + bp.get("gamma", 0.0) + 1e-10
    return (
        float(np.clip(bp["theta"] / total, 0, 1)),
        float(np.clip(bp["alpha"] / total, 0, 1)),
        float(np.clip(bp["beta"] / total, 0, 1)),
        float(np.clip(bp.get("gamma", 0.0) / total, 0, 1)),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/engagement/baseline")
async def set_engagement_baseline(data: EEGInput):
    """Set per-user focused-state baseline from EEG.

    Record during a focused task (e.g., counting backwards).
    Baseline improves engagement index accuracy by providing
    a personal reference point.

    Returns the stored baseline band-power ratios and user_id.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    theta, alpha, beta, _ = _get_band_powers(signals, data.fs)
    _detector.set_baseline(
        theta_power=theta,
        alpha_power=alpha,
        beta_power=beta,
        user_id=data.user_id,
    )
    return _numpy_safe({
        "status": "ok",
        "baseline_set": True,
        "user_id": data.user_id,
        "theta": round(theta, 6),
        "alpha": round(alpha, 6),
        "beta": round(beta, 6),
    })


@router.post("/engagement/assess")
async def assess_engagement(data: EEGInput):
    """Assess student engagement state from live EEG.

    Classifies engagement into attentive / passive / disengaged
    and detects educational emotions: boredom, confusion, curiosity,
    frustration, concentration.

    Returns engagement_index (0-1), engagement_state,
    educational_emotion, emotion_scores, attention_index,
    mind_wandering_risk, and n_samples.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    theta, alpha, beta, gamma = _get_band_powers(signals, data.fs)
    result = _detector.assess(
        theta_power=theta,
        alpha_power=alpha,
        beta_power=beta,
        gamma_power=gamma,
        user_id=data.user_id,
    )
    result["user_id"] = data.user_id
    return _numpy_safe(result)


@router.get("/engagement/curve")
async def get_engagement_curve(user_id: str):
    """Get engagement time-series for a user.

    Returns ordered list of past assessments with engagement_index,
    engagement_state, and educational_emotion for trend visualisation.
    """
    curve = _detector.get_engagement_curve(user_id=user_id)
    return _numpy_safe({"curve": curve, "user_id": user_id, "count": len(curve)})


@router.get("/engagement/summary")
async def get_engagement_summary(user_id: str):
    """Get session summary statistics for a user.

    Returns mean_engagement, state distribution, dominant_emotion,
    attention statistics, and mind_wandering statistics.
    """
    result = _detector.get_session_summary(user_id=user_id)
    result["user_id"] = user_id
    return _numpy_safe(result)


@router.post("/engagement/reset")
async def reset_engagement(user_id: str):
    """Clear engagement history and baseline for a user."""
    _detector.reset(user_id=user_id)
    return {"status": "ok", "message": "Engagement state cleared.", "user_id": user_id}
