"""Brain health score API — composite EEG-based cognitive health indicator.

Endpoints:
  POST /brain-health/baseline  -- record resting-state baseline
  POST /brain-health/assess    -- assess current brain health from EEG
  GET  /brain-health/stats     -- session statistics per user
  POST /brain-health/reset     -- clear baseline and history per user

GitHub issue: #131
"""

import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe
from models.brain_health_score import BrainHealthScore

router = APIRouter(tags=["brain-health"])

_estimator = BrainHealthScore()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/brain-health/baseline")
async def set_brain_health_baseline(data: EEGInput):
    """Record resting-state baseline for brain health estimation.

    Call during 2-3 minutes of eyes-closed rest.
    Baseline normalises subsequent assessments across five domains:
    spectral, connectivity, complexity, stability, and asymmetry.

    Returns baseline_set flag and per-domain baseline scores.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    result = _estimator.set_baseline(signals, fs=data.fs, user_id=data.user_id)
    return _numpy_safe(result)


@router.post("/brain-health/assess")
async def assess_brain_health(data: EEGInput):
    """Assess brain health from an EEG epoch.

    Composite score (0-100) across five domains:
    - Spectral: alpha peak frequency, delta/theta ratio
    - Connectivity: inter-channel coherence
    - Complexity: Hjorth mobility, spectral entropy
    - Stability: epoch-to-epoch variance
    - Asymmetry: frontal alpha asymmetry (FAA)

    Returns overall_score, grade (A/B/C/D/F), domain_scores,
    recommendations, and has_baseline flag.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    result = _estimator.assess(signals, fs=data.fs, user_id=data.user_id)
    return _numpy_safe(result)


@router.get("/brain-health/stats")
async def get_brain_health_stats(user_id: str = "default"):
    """Get session statistics for brain health assessments.

    Returns n_assessments, mean_score, score_trend, mean_domain_scores,
    best_domain, worst_domain, and has_baseline.
    """
    result = _estimator.get_session_stats(user_id=user_id)
    return _numpy_safe(result)


@router.post("/brain-health/reset")
async def reset_brain_health(user_id: str = "default"):
    """Clear baseline and session history for a user."""
    _estimator.reset(user_id=user_id)
    return {
        "status": "ok",
        "message": "Brain health session reset. Record a new baseline before assessing.",
    }
