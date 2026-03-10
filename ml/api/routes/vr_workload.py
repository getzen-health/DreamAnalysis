"""Adaptive VR/AR cognitive workload control via frontal EEG (#132).

Estimates frontal-theta cognitive load from AF7/AF8, computes an adaptation
signal for VR/AR difficulty engines, and tracks how the brain responds to
workload changes. Based on Chaouachi et al. (2010) and Zander & Kothe (2011).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/vr-workload", tags=["vr-workload"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class WorkloadInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    current_difficulty: float = 0.5   # 0-1 scale from the VR engine
    user_id: str = Field(..., min_length=1)


class WorkloadResult(BaseModel):
    user_id: str
    cognitive_load: float             # 0-1
    frontal_theta: float              # µV²/Hz
    alpha_engagement: float           # engagement index
    adaptation_signal: float          # -1 (reduce difficulty) … +1 (increase)
    recommended_difficulty: float     # 0-1
    load_level: str                   # low / optimal / high / overload
    processed_at: float


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

class _UserState:
    def __init__(self):
        self.baseline_theta: Optional[float] = None
        self.history: deque = deque(maxlen=500)
        self.frame_count: int = 0

_user_states: Dict[str, _UserState] = defaultdict(_UserState)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band_power(signal: np.ndarray, fs: float, flo: float, fhi: float) -> float:
    from scipy.signal import welch
    nperseg = min(len(signal), int(fs * 2))
    f, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = np.logical_and(f >= flo, f <= fhi)
    return float(np.mean(psd[idx])) if idx.any() else 0.0


def _compute_workload(signals: np.ndarray, fs: float,
                       state: _UserState, difficulty: float) -> dict:
    n_ch = signals.shape[0]
    theta_vals, alpha_vals, beta_vals = [], [], []
    for ch in range(min(n_ch, 4)):
        theta_vals.append(_band_power(signals[ch], fs, 4, 8))
        alpha_vals.append(_band_power(signals[ch], fs, 8, 12))
        beta_vals.append(_band_power(signals[ch], fs, 12, 30))

    theta = float(np.mean(theta_vals)) + 1e-9
    alpha = float(np.mean(alpha_vals)) + 1e-9
    beta  = float(np.mean(beta_vals))  + 1e-9

    # Baseline establishment (first 30 frames at rest)
    state.frame_count += 1
    if state.baseline_theta is None and state.frame_count >= 30:
        recent = [r["frontal_theta"] for r in list(state.history)[-30:]]
        if recent:
            state.baseline_theta = float(np.mean(recent))

    baseline = state.baseline_theta or theta

    # Engagement index: β / (α × θ) — Sterman & Mann (1995)
    engagement = float(np.clip(beta / (alpha * theta), 0, 10))

    # Normalised cognitive load
    theta_norm = float(np.clip(theta / (baseline + 1e-9) - 1.0, 0, 3)) / 3.0
    engagement_norm = float(np.clip(engagement / 5.0, 0, 1))
    load = float(np.clip(0.6 * theta_norm + 0.4 * engagement_norm, 0, 1))

    if load < 0.25:
        level = "low"
    elif load < 0.55:
        level = "optimal"
    elif load < 0.75:
        level = "high"
    else:
        level = "overload"

    # Adaptation signal: negative → reduce difficulty, positive → increase
    target_load = 0.45
    adaptation = float(np.clip((target_load - load) * 2, -1, 1))
    recommended = float(np.clip(difficulty + adaptation * 0.1, 0, 1))

    return {
        "cognitive_load": load,
        "frontal_theta": theta,
        "alpha_engagement": engagement,
        "adaptation_signal": adaptation,
        "recommended_difficulty": recommended,
        "load_level": level,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/set-baseline/{user_id}")
async def set_vr_baseline(user_id: str):
    """Reset user workload baseline (use during 2-min resting state before VR session)."""
    state = _user_states[user_id]
    state.baseline_theta = None
    state.frame_count = 0
    state.history.clear()
    return {"user_id": user_id, "status": "baseline_cleared", "note": "Send 30+ frames to establish new baseline"}


@router.post("/assess", response_model=WorkloadResult)
async def assess_vr_workload(req: WorkloadInput):
    """Assess cognitive workload and return VR difficulty adaptation signal."""
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    state = _user_states[req.user_id]
    feats = _compute_workload(signals, req.fs, state, req.current_difficulty)

    result = WorkloadResult(
        user_id=req.user_id,
        processed_at=time.time(),
        **feats,
    )
    state.history.append(result.dict())
    return result


@router.get("/stats/{user_id}")
async def get_vr_workload_stats(user_id: str):
    """Return aggregate workload statistics."""
    state = _user_states[user_id]
    h = list(state.history)
    if not h:
        return {"user_id": user_id, "n_frames": 0}
    loads = [r["cognitive_load"] for r in h]
    levels = [r["load_level"] for r in h]
    return {
        "user_id": user_id,
        "n_frames": len(h),
        "mean_load": float(np.mean(loads)),
        "max_load": float(np.max(loads)),
        "baseline_established": state.baseline_theta is not None,
        "dominant_level": max(set(levels), key=levels.count),
    }


@router.post("/reset/{user_id}")
async def reset_vr_workload(user_id: str):
    """Clear workload state and history for a user."""
    _user_states[user_id] = _UserState()
    return {"user_id": user_id, "status": "reset"}
