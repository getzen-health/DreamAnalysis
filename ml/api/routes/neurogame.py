"""EEG-driven neurogaming engine API.

Endpoints:
  POST /neurogame/calibrate  -- set per-user calibration from focus/relax EEG
  POST /neurogame/command    -- convert live EEG to game command + difficulty
  GET  /neurogame/stats      -- session stats for a user
  GET  /neurogame/history    -- command history for a user
  POST /neurogame/reset      -- clear state for a user

GitHub issue: #110
"""

import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe, extract_band_powers, preprocess
from models.neurogame_engine import NeurogameEngine

router = APIRouter(tags=["neurogame"])

_engine = NeurogameEngine()


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
        bp,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/neurogame/calibrate")
async def calibrate_neurogame(data: EEGInput):
    """Set per-user calibration thresholds from EEG baseline.

    Send EEG recorded during a focused task; the route extracts
    beta/theta (focus) and alpha/beta (relax) ratios and stores
    them as calibration thresholds for this user.

    Returns calibration thresholds and status.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    _, _, _, bp = _get_band_powers(signals, data.fs)
    focus_beta_theta = float(bp["beta"] / max(bp["theta"], 1e-10))
    relax_alpha_beta = float(bp["alpha"] / max(bp["beta"], 1e-10))

    result = _engine.calibrate(
        focus_beta_theta=focus_beta_theta,
        relax_alpha_beta=relax_alpha_beta,
        user_id=data.user_id,
    )
    result["user_id"] = data.user_id
    return _numpy_safe(result)


@router.post("/neurogame/command")
async def get_game_command(data: EEGInput):
    """Convert live EEG into a game command with adaptive difficulty.

    Returns command (focus_boost/relax_action/idle), intensity,
    engagement_level, difficulty_level, and calibration status.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    theta, alpha, beta, _ = _get_band_powers(signals, data.fs)
    result = _engine.get_command(
        theta_power=theta,
        alpha_power=alpha,
        beta_power=beta,
        user_id=data.user_id,
    )
    result["user_id"] = data.user_id
    return _numpy_safe(result)


@router.get("/neurogame/stats")
async def get_neurogame_stats(user_id: str = "default"):
    """Get session gameplay statistics for a user.

    Returns command distribution, mean engagement, mean difficulty,
    session duration, and n_commands.
    """
    result = _engine.get_session_stats(user_id=user_id)
    result["user_id"] = user_id
    return _numpy_safe(result)


@router.get("/neurogame/history")
async def get_neurogame_history(user_id: str = "default", last_n: int = 50):
    """Get command history for a user (most recent first).

    Query params:
      last_n: number of most recent entries to return (default 50)
    """
    history = _engine.get_history(user_id=user_id, last_n=last_n)
    return _numpy_safe({"history": history, "user_id": user_id, "count": len(history)})


@router.post("/neurogame/reset")
async def reset_neurogame(user_id: str = "default"):
    """Clear game state and history for a user."""
    _engine.reset(user_id=user_id)
    return {"status": "ok", "message": "Neurogame state cleared.", "user_id": user_id}
