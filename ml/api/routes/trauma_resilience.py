"""Trauma hyperarousal detection and entropy-based resilience tracking.

Endpoints:
  POST /hyperarousal          -- compute hyperarousal index from EEG
  POST /resilience/baseline   -- set resting baseline for resilience tracker
  POST /resilience/modulation -- measure entropy modulation vs baseline
  GET  /resilience/trend      -- session-over-session resilience trajectory
  POST /resilience/reset      -- clear baseline and history

GitHub issues: #52 (hyperarousal), #57 (resilience biomarker).
"""

import threading
from typing import Dict

import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe
from models.hyperarousal_detector import HyperarousalDetector
from models.resilience_tracker import ResilienceTracker

router = APIRouter(tags=["trauma-resilience"])

# ── Singletons ────────────────────────────────────────────────────────────────
_hyperarousal = HyperarousalDetector()

# Per-user ResilienceTracker instances (stateful: baseline + session scores)
_resilience_trackers: Dict[str, ResilienceTracker] = {}
_resilience_lock = threading.Lock()


def _get_tracker(user_id: str) -> ResilienceTracker:
    """Return the per-user ResilienceTracker, creating it on first use."""
    with _resilience_lock:
        if user_id not in _resilience_trackers:
            _resilience_trackers[user_id] = ResilienceTracker()
        return _resilience_trackers[user_id]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/hyperarousal")
async def predict_hyperarousal(data: EEGInput):
    """Compute PTSD/trauma hyperarousal index from EEG signals.

    Lower frontal Shannon entropy + lower AF8 alpha = higher hyperarousal.
    Based on Scientific Reports (2024): Shannon entropy at AF3/AF4
    negatively correlates with PTSD severity (r=-0.43 to -0.47).

    Returns hyperarousal_index (0-1), risk_level (low/moderate/high),
    component scores, and per-channel Shannon entropies.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    result = _hyperarousal.predict(signals, data.fs)
    return _numpy_safe(result)


@router.post("/resilience/baseline")
async def set_resilience_baseline(data: EEGInput):
    """Set resting baseline for the entropy resilience tracker.

    Call during eyes-closed resting state (2 min recommended).
    Computes average spectral entropy across all channels as the
    baseline reference for subsequent modulation measurements.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    tracker = _get_tracker(data.user_id)
    tracker.set_baseline(signals, data.fs)

    return {
        "status": "ok",
        "baseline_set": True,
        "message": "Resting baseline recorded. Call POST /resilience/modulation during task epochs.",
    }


@router.post("/resilience/modulation")
async def compute_resilience_modulation(data: EEGInput):
    """Measure entropy modulation from baseline (resilience score).

    Higher modulation = brain responds more dynamically to challenge =
    higher neural flexibility = higher resilience.

    Based on IBRO Neuroreports (2025): entropy modulation during
    emotional challenge predicts psychological resilience.

    Returns resilience_score (0-1), entropy_modulation (ratio),
    baseline/task entropy, and direction (increase/decrease).
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    tracker = _get_tracker(data.user_id)
    result = tracker.compute_modulation(signals, data.fs)
    return _numpy_safe(result)


@router.get("/resilience/trend")
async def get_resilience_trend(user_id: str = "default"):
    """Get resilience trend across measurements within this session.

    Returns mean_score, latest_score, measurement count, and
    directional trend (improving / declining / stable / insufficient_data).
    """
    tracker = _get_tracker(user_id)
    return _numpy_safe(tracker.get_trend())


@router.post("/resilience/reset")
async def reset_resilience(user_id: str = "default"):
    """Clear baseline and session history for the resilience tracker."""
    tracker = _get_tracker(user_id)
    tracker.reset()
    return {
        "status": "ok",
        "message": "Resilience tracker reset. Record a new baseline before measuring modulation.",
    }
