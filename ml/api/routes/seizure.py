"""Seizure detection API endpoints.

Three endpoints:
    POST /seizure/analyze    — analyze EEG window for seizure activity
    GET  /seizure/status     — current alarm state, consecutive count, threshold
    POST /seizure/reset-alarm — reset consecutive counter and alarm state
"""
from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.seizure_detector import get_seizure_detector

router = APIRouter()


class SeizureAnalyzeRequest(BaseModel):
    signals: List[List[float]] = Field(
        ...,
        description="EEG data: list of channels, each a list of samples. "
                    "Single-channel: [[s0, s1, ...]]. Multi-channel: [[ch0...],[ch1...]].",
    )
    fs: float = Field(256.0, description="Sampling rate in Hz.")
    window_seconds: float = Field(4.0, description="Window length in seconds (informational).")


def _numpy_safe(obj):
    """Recursively convert numpy scalars to Python natives."""
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@router.post("/seizure/analyze")
async def analyze_seizure(req: SeizureAnalyzeRequest):
    """Analyze EEG window for seizure activity.

    Returns ictal/interictal classification, probability, alert level, and
    extracted features. The rolling alarm buffer tracks consecutive ictal
    windows; ``alarm_active`` is True when the trigger count is met.
    """
    detector = get_seizure_detector()
    signals = np.array(req.signals, dtype=float)
    # Accept (n_channels, n_samples) — already 2-D from JSON
    result = detector.predict(signals, fs=req.fs, window_seconds=req.window_seconds)
    return _numpy_safe(result)


@router.get("/seizure/status")
async def seizure_status():
    """Return current alarm state, consecutive ictal count, and threshold."""
    detector = get_seizure_detector()
    return _numpy_safe(detector.get_status())


@router.post("/seizure/reset-alarm")
async def reset_seizure_alarm():
    """Reset the consecutive ictal counter and clear the alarm state."""
    detector = get_seizure_detector()
    return _numpy_safe(detector.reset_alarm())
