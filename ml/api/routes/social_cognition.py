"""Social cognition detection API — empathy, mentalizing, and mu suppression.

Endpoints:
  POST /social-cognition/baseline  -- record non-social resting baseline
  POST /social-cognition/assess    -- assess social cognitive state from EEG
  GET  /social-cognition/stats     -- session statistics
  POST /social-cognition/reset     -- clear baseline and history per user

GitHub issue: #126
"""

import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe
from models.social_cognition import SocialCognitionDetector

router = APIRouter(tags=["social-cognition"])

_detector = SocialCognitionDetector()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/social-cognition/baseline")
async def set_social_baseline(data: EEGInput):
    """Record non-social resting baseline for mu and theta powers.

    Call during 2 minutes of eyes-closed rest without social stimuli.
    Baseline normalizes subsequent assessments for inter-individual
    differences in resting mu/theta amplitude.

    Returns baseline_mu_power, baseline_theta_power, and baseline_set flag.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    result = _detector.set_baseline(signals, fs=data.fs, user_id=data.user_id)
    return _numpy_safe(result)


@router.post("/social-cognition/assess")
async def assess_social_cognition(data: EEGInput):
    """Assess social cognitive state from EEG during social observation.

    Key EEG markers (Perry et al. 2010; Mu et al. 2008):
    - Mu suppression (8-13 Hz at temporal TP9/TP10): mirror neuron activation
    - Frontal theta increase: cognitive empathy / mentalizing
    - Alpha asymmetry: approach motivation during social engagement

    Returns empathy_score (0-1), social_engagement_score (0-1),
    mu_suppression_ratio, frontal_theta_ratio, social_state label,
    and behavioural recommendations.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    result = _detector.assess(signals, fs=data.fs, user_id=data.user_id)
    return _numpy_safe(result)


@router.get("/social-cognition/stats")
async def get_social_cognition_stats(user_id: str):
    """Get session statistics for social cognition assessments.

    Returns n_assessments, mean_empathy_score, mean_engagement_score,
    has_baseline, and state distribution across the session.
    """
    result = _detector.get_session_stats(user_id=user_id)
    return _numpy_safe(result)


@router.post("/social-cognition/reset")
async def reset_social_cognition(user_id: str):
    """Clear baseline and session history for a user."""
    _detector.reset(user_id=user_id)
    return {
        "status": "ok",
        "message": "Social cognition session reset. Record a new baseline before assessing.",
    }
