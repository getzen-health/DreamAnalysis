"""EEG-based deception detection (Concealed Information Test) API.

Endpoints:
  POST /deception/detect          -- compare single probe vs irrelevant epoch
  POST /deception/detect-average  -- average across multiple probe/irrelevant epochs
  GET  /deception/history         -- detection history for a user
  GET  /deception/summary         -- summary statistics for a user
  POST /deception/reset           -- clear history for a user

DISCLAIMER: Research tool only. Not validated for forensic or legal use.

GitHub issue: #104
"""

from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ._shared import _numpy_safe
from models.deception_detector import DeceptionDetector

router = APIRouter(tags=["deception"])

_detector = DeceptionDetector()


class DeceptionInput(BaseModel):
    """Two-epoch input for CIT deception detection."""

    probe_signals: List[List[float]] = Field(
        ...,
        description="EEG epoch time-locked to probe stimulus. Shape: (n_channels, n_samples) or (1, n_samples).",
    )
    irrelevant_signals: List[List[float]] = Field(
        ...,
        description="EEG epoch time-locked to irrelevant stimulus. Same shape as probe_signals.",
    )
    fs: float = Field(256.0, description="Sampling rate in Hz.")
    user_id: str = Field(..., description="User identifier.")
    channel_idx: int = Field(2, description="EEG channel index for P300 extraction (default 2 = AF8).")


class DeceptionAverageInput(BaseModel):
    """Multiple epochs for averaged CIT detection."""

    probe_epochs: List[List[List[float]]] = Field(
        ...,
        description="List of probe epochs. Each: (n_channels, n_samples).",
    )
    irrelevant_epochs: List[List[List[float]]] = Field(
        ...,
        description="List of irrelevant epochs. Same format.",
    )
    fs: float = Field(256.0, description="Sampling rate in Hz.")
    user_id: str = Field(..., description="User identifier.")
    channel_idx: int = Field(2, description="EEG channel index for P300 (default 2 = AF8).")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/deception/detect")
async def detect_deception(data: DeceptionInput):
    """Compare single probe vs irrelevant EEG epoch for concealed knowledge.

    The P300 ERP component is larger for stimuli the subject recognises
    even when trying to conceal knowledge (Concealed Information Test).

    Returns deception_score (0-1), deception_detected, p300_difference,
    probe/irrelevant P300 amplitudes, confidence, and disclaimer.

    DISCLAIMER: Research tool only. Not validated for forensic or legal use.
    """
    probe = np.array(data.probe_signals)
    irrelevant = np.array(data.irrelevant_signals)

    result = _detector.detect(
        probe_epoch=probe,
        irrelevant_epoch=irrelevant,
        fs=data.fs,
        channel_idx=data.channel_idx,
        user_id=data.user_id,
    )
    result["user_id"] = data.user_id
    return _numpy_safe(result)


@router.post("/deception/detect-average")
async def detect_deception_average(data: DeceptionAverageInput):
    """Average multiple probe and irrelevant epochs for more reliable CIT detection.

    Averaging across epochs reduces noise and improves P300 SNR.
    Recommended: ≥10 epochs per condition for reliable results.

    Returns same keys as /deception/detect plus n_probe_epochs and n_irrelevant_epochs.

    DISCLAIMER: Research tool only. Not validated for forensic or legal use.
    """
    probe_epochs = [np.array(ep) for ep in data.probe_epochs]
    irrelevant_epochs = [np.array(ep) for ep in data.irrelevant_epochs]

    result = _detector.detect_average(
        probe_epochs=probe_epochs,
        irrelevant_epochs=irrelevant_epochs,
        fs=data.fs,
        channel_idx=data.channel_idx,
        user_id=data.user_id,
    )
    result["user_id"] = data.user_id
    return _numpy_safe(result)


@router.get("/deception/history")
async def get_deception_history(user_id: str, last_n: int = 50):
    """Get detection history for a user.

    Query params:
      last_n: number of most recent entries to return (default 50)
    """
    history = _detector.get_history(user_id=user_id, last_n=last_n)
    return _numpy_safe({"history": history, "user_id": user_id, "count": len(history)})


@router.get("/deception/summary")
async def get_deception_summary(user_id: str):
    """Get summary statistics for a user.

    Returns n_detections, mean_deception_score, detected_count,
    detection_rate, mean_p300_difference, mean_confidence.
    """
    result = _detector.get_summary(user_id=user_id)
    result["user_id"] = user_id
    return _numpy_safe(result)


@router.post("/deception/reset")
async def reset_deception(user_id: str):
    """Clear detection history for a user."""
    _detector.reset(user_id=user_id)
    return {"status": "ok", "message": "Deception detection history cleared.", "user_id": user_id}
