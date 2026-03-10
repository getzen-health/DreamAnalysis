"""Developmental EEG brain maturation tracking API.

Endpoints:
  POST /brain-maturation/assess   -- assess brain maturation stage from EEG
  GET  /brain-maturation/trajectory -- longitudinal trajectory for a user
  GET  /brain-maturation/summary    -- maturation summary statistics for a user
  POST /brain-maturation/reset      -- clear history for a user

GitHub issue: #111
"""

import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe
from models.brain_maturation import BrainMaturationTracker

router = APIRouter(tags=["brain-maturation"])

_tracker = BrainMaturationTracker()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/brain-maturation/assess")
async def assess_brain_maturation(data: EEGInput):
    """Assess brain maturation stage from EEG aperiodic and periodic features.

    Key markers:
    - Aperiodic exponent (1/f slope): decreases with maturation
    - Alpha peak frequency: increases from ~6 Hz (child) to ~10 Hz (adult)
    - Delta/theta dominance: decreases with development
    - Beta emergence: increases in adolescence/adulthood
    - Spectral entropy: increases toward adulthood

    Returns estimated_brain_age (years), maturation_stage, maturation_index (0-1),
    aperiodic_exponent, alpha_peak_hz, normative_comparison, and recommendations.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    result = _tracker.assess(signals, fs=data.fs, user_id=data.user_id)
    return _numpy_safe(result)


@router.get("/brain-maturation/trajectory")
async def get_maturation_trajectory(user_id: str):
    """Get longitudinal maturation trajectory for a user.

    Returns ordered list of past assessments with timestamps, estimated_brain_age,
    maturation_index, and maturation_stage for tracking developmental progress.
    """
    result = _tracker.get_trajectory(user_id=user_id)
    return _numpy_safe({"trajectory": result, "user_id": user_id})


@router.get("/brain-maturation/summary")
async def get_maturation_summary(user_id: str):
    """Get maturation summary statistics for a user.

    Returns n_assessments, mean_brain_age, mean_maturation_index,
    current_stage, trend, and longitudinal statistics.
    """
    result = _tracker.get_summary(user_id=user_id)
    return _numpy_safe(result)


@router.post("/brain-maturation/reset")
async def reset_maturation(user_id: str):
    """Clear maturation history for a user."""
    _tracker.reset(user_id=user_id)
    return {
        "status": "ok",
        "message": "Brain maturation history cleared.",
        "user_id": user_id,
    }
