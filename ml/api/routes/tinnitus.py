"""Alpha neurofeedback API for tinnitus relief training.

Endpoints:
  POST /tinnitus/baseline  -- set resting alpha baseline at TP9/TP10
  POST /tinnitus/evaluate  -- evaluate a training epoch, return reward + feedback
  GET  /tinnitus/stats     -- session statistics (reward_rate, trend, ...)
  POST /tinnitus/reset     -- clear session and baseline for a user

GitHub issue: #105
"""

import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe
from models.tinnitus_nf_protocol import TinnitusNFProtocol

router = APIRouter(tags=["tinnitus"])

# Single shared instance — per-user state is held inside the protocol object
_protocol = TinnitusNFProtocol()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/tinnitus/baseline")
async def set_tinnitus_baseline(data: EEGInput):
    """Set resting alpha baseline at temporal channels (TP9/TP10).

    Record 2 minutes of eyes-closed resting EEG and call this endpoint
    once per epoch.  Alpha power at TP9 (ch0) and TP10 (ch3) is averaged
    and stored as the per-user baseline for subsequent training evaluations.

    Returns baseline_alpha, channel_powers, and baseline_set flag.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    result = _protocol.set_baseline(signals, fs=data.fs, user_id=data.user_id)
    return _numpy_safe(result)


@router.post("/tinnitus/evaluate")
async def evaluate_tinnitus_epoch(data: EEGInput):
    """Evaluate a training epoch and return neurofeedback reward signal.

    Compares current temporal alpha against the recorded baseline.
    Returns reward (bool), alpha_ratio, feedback_intensity (0-1),
    and feedback_tone_hz for auditory feedback (None when no reward).

    Protocol (Crocetti et al. 2011, Dohrmann et al. 2007):
    - Reward when temporal alpha >= baseline * 1.1
    - Higher alpha → higher pitch tone (440–640 Hz)
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    result = _protocol.evaluate(signals, fs=data.fs, user_id=data.user_id)
    return _numpy_safe(result)


@router.get("/tinnitus/stats")
async def get_tinnitus_stats(user_id: str):
    """Get training session statistics for a user.

    Returns n_epochs, reward_rate, mean_alpha_ratio, max_alpha_ratio,
    has_baseline, and trend (improving / stable / declining /
    insufficient_data).
    """
    result = _protocol.get_session_stats(user_id=user_id)
    return _numpy_safe(result)


@router.post("/tinnitus/reset")
async def reset_tinnitus(user_id: str):
    """Clear session history and baseline for a user.

    Call before starting a fresh training session or when switching users.
    """
    _protocol.reset(user_id=user_id)
    return {
        "status": "ok",
        "message": "Tinnitus NF session reset. Record a new baseline before training.",
    }
