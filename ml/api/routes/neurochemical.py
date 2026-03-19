"""Neurochemical state inference API endpoints.

Endpoints
---------
POST /neurochemical/estimate
    Estimate current neurochemical state from EEG spectral features.

POST /neurochemical/profile
    Compute full neurochemical profile with trends for a user.

GET  /neurochemical/status
    Get service status and available neurochemicals.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.neurochemical_model import NeurochemicalEstimator

import numpy as np

router = APIRouter(prefix="/neurochemical", tags=["neurochemical"])

# Module-level singleton
_estimator = NeurochemicalEstimator()


def get_estimator() -> NeurochemicalEstimator:
    """Return the module-level NeurochemicalEstimator singleton."""
    return _estimator


# ── Request models ──────────────────────────────────────────────────


class EstimateRequest(BaseModel):
    """Request body for neurochemical estimation from EEG."""

    eeg_data: List[List[float]] = Field(
        ...,
        description=(
            "EEG data as list of channels, each a list of samples. "
            "Single channel: [[s1, s2, ...]]. "
            "Multi-channel: [[ch0_s1, ...], [ch1_s1, ...], ...]."
        ),
    )
    fs: float = Field(
        default=256.0,
        description="Sampling frequency in Hz.",
    )


class ProfileRequest(BaseModel):
    """Request body for full neurochemical profile."""

    user_id: str = Field(..., description="User identifier")
    eeg_data: List[List[float]] = Field(
        ...,
        description="EEG data as list of channels.",
    )
    fs: float = Field(
        default=256.0,
        description="Sampling frequency in Hz.",
    )


# ── Endpoints ───────────────────────────────────────────────────────


@router.post("/estimate")
async def estimate_neurochemical_state(req: EstimateRequest):
    """Estimate current neurochemical state from EEG data.

    Returns proxy estimates for dopamine, serotonin, cortisol,
    norepinephrine, GABA, and endorphins with confidence scores.
    """
    eeg = np.array(req.eeg_data, dtype=np.float64)
    if eeg.ndim == 1:
        eeg = eeg.reshape(1, -1)

    result = _estimator.estimate_neurochemical_state(eeg, req.fs)
    return result


@router.post("/profile")
async def compute_neurochemical_profile(req: ProfileRequest):
    """Compute full neurochemical profile with balance assessment.

    Stores result in trend history for the user and returns the
    complete profile including imbalance detection and mood inference.
    """
    eeg = np.array(req.eeg_data, dtype=np.float64)
    if eeg.ndim == 1:
        eeg = eeg.reshape(1, -1)

    profile = _estimator.compute_balance_profile(req.user_id, eeg, req.fs)
    return profile.to_dict()


@router.get("/status")
async def get_neurochemical_status():
    """Get neurochemical estimation service status.

    Returns the list of tracked neurochemicals, their descriptions,
    and the estimation methodology.
    """
    return {
        "status": "operational",
        "neurochemicals": [
            {
                "name": "dopamine",
                "description": "Reward, motivation, and pleasure signaling",
                "proxy_signals": ["frontal_beta", "alpha_suppression", "FAA"],
            },
            {
                "name": "serotonin",
                "description": "Mood stability, calm contentment",
                "proxy_signals": ["frontal_alpha", "low_irritability", "emotional_stability"],
            },
            {
                "name": "cortisol",
                "description": "Stress response, fight-or-flight",
                "proxy_signals": ["beta_alpha_ratio", "high_beta", "right_frontal"],
            },
            {
                "name": "norepinephrine",
                "description": "Arousal, alertness, and vigilance",
                "proxy_signals": ["overall_beta", "theta_suppression", "wakefulness"],
            },
            {
                "name": "gaba",
                "description": "Neural inhibition, relaxation, calming",
                "proxy_signals": ["alpha_magnitude", "alpha_beta_ratio", "sleep_markers"],
            },
            {
                "name": "endorphin",
                "description": "Pain relief, euphoria, well-being",
                "proxy_signals": ["alpha_rebound", "positive_affect", "low_stress"],
            },
        ],
        "methodology": "EEG spectral proxy estimation (not direct measurement)",
        "version": "1.0.0",
    }
