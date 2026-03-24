"""YASA sleep staging + spindle + slow wave detection API endpoints.

Provides three endpoints:
- POST /sleep/stage-yasa — Stage sleep from raw EEG
- POST /sleep/spindles-yasa — Detect sleep spindles
- POST /sleep/slow-waves-yasa — Detect slow oscillations

GitHub issue: #527
"""
import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from ._shared import _numpy_safe

log = logging.getLogger(__name__)
router = APIRouter(tags=["yasa-sleep"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SleepEEGRequest(BaseModel):
    """Request body for YASA sleep analysis endpoints."""

    eeg_data: List[float] = Field(
        ..., description="1D array of EEG samples in microvolts"
    )
    sample_rate: int = Field(
        256, ge=64, le=1024, description="Sample rate in Hz (256 for Muse 2)"
    )
    channel: str = Field(
        "AF7", description="Channel name (AF7 or AF8 for Muse 2)"
    )


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

def _get_stager():
    """Lazy-load the YASASleepStager to avoid import-time overhead."""
    from models.yasa_sleep import YASASleepStager
    return YASASleepStager()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/sleep/stage-yasa")
async def stage_sleep_yasa(data: SleepEEGRequest):
    """Stage sleep from raw EEG using YASA's pre-trained LightGBM classifier.

    Requires at least 5 minutes of continuous EEG data. Returns per-30s-epoch
    sleep stage labels (WAKE, N1, N2, N3, REM) with probabilities and
    AASM-standard sleep statistics.

    Validated on Muse-S: Cohen's Kappa 0.76, accuracy 88-96% across stages.
    """
    try:
        from models.yasa_sleep import stage_with_yasa

        eeg = np.array(data.eeg_data, dtype=np.float64)
        result = stage_with_yasa(eeg, fs=data.sample_rate, channel_name=data.channel)
        return _numpy_safe(result)
    except Exception as exc:
        log.exception("YASA sleep staging endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/sleep/spindles-yasa")
async def detect_spindles_yasa_endpoint(data: SleepEEGRequest):
    """Detect sleep spindles (12-15 Hz bursts, 0.5-2s) using YASA.

    Returns spindle count, density (per minute), average duration and frequency,
    and details for up to 20 individual spindles.
    Spindle density during N2/N3 predicts memory consolidation.
    """
    try:
        from models.yasa_sleep import detect_spindles_yasa

        eeg = np.array(data.eeg_data, dtype=np.float64)
        result = detect_spindles_yasa(eeg, fs=data.sample_rate)
        return _numpy_safe(result)
    except Exception as exc:
        log.exception("YASA spindle detection endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/sleep/slow-waves-yasa")
async def detect_slow_waves_yasa_endpoint(data: SleepEEGRequest):
    """Detect slow oscillations (0.3-1.5 Hz) using YASA.

    Returns slow wave count, density (per minute), and average peak-to-trough
    amplitude. Slow waves orchestrate memory consolidation during deep sleep.
    """
    try:
        from models.yasa_sleep import detect_slow_waves_yasa

        eeg = np.array(data.eeg_data, dtype=np.float64)
        result = detect_slow_waves_yasa(eeg, fs=data.sample_rate)
        return _numpy_safe(result)
    except Exception as exc:
        log.exception("YASA slow wave detection endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class SleepOnsetRequest(BaseModel):
    """Request body for sleep onset detection from pre-staged epochs."""

    stages: List[str] = Field(
        ..., description="List of sleep stage labels per epoch (Wake, N1, N2, N3, REM)"
    )
    confidences: Optional[List[float]] = Field(
        None, description="Per-epoch confidence scores (0-1). Defaults to 0.5 if absent."
    )
    epoch_duration_s: float = Field(
        30.0, ge=1.0, le=300.0, description="Duration of each epoch in seconds"
    )
    recording_start_iso: Optional[str] = Field(
        None, description="ISO-format datetime of recording start (e.g. 2026-03-23T23:00:00+00:00)"
    )


@router.post("/sleep/onset")
async def detect_sleep_onset_endpoint(data: SleepOnsetRequest):
    """Detect the exact moment of sleep onset from staged epoch labels.

    Returns the first sustained Wake -> sleep transition (3+ consecutive
    non-Wake epochs = 90 seconds at 30s/epoch). Useful for "You fell asleep
    at 11:23 PM" features.

    If no sleep onset is found (all Wake, or non-Wake never sustained),
    returns {sleep_onset: null}.
    """
    try:
        from models.sleep_staging import detect_sleep_onset
        from datetime import datetime

        stage_idx_map = {"Wake": 0, "W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "R": 4}
        confs = data.confidences or [0.5] * len(data.stages)

        epochs = []
        for stage, conf in zip(data.stages, confs):
            epochs.append({
                "stage": stage if stage not in ("W", "R") else {"W": "Wake", "R": "REM"}.get(stage, stage),
                "stage_index": stage_idx_map.get(stage, 0),
                "confidence": conf,
            })

        recording_start = None
        if data.recording_start_iso:
            recording_start = datetime.fromisoformat(data.recording_start_iso)

        result = detect_sleep_onset(
            epochs,
            epoch_duration_s=data.epoch_duration_s,
            recording_start=recording_start,
        )

        return _numpy_safe({"sleep_onset": result})
    except Exception as exc:
        log.exception("Sleep onset detection endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/sleep/yasa-status")
async def yasa_status():
    """Check if YASA is available and return version info."""
    try:
        import yasa
        return {
            "available": True,
            "version": yasa.__version__,
            "features": ["sleep_staging", "spindle_detection", "slow_wave_detection", "sleep_onset_detection"],
        }
    except ImportError:
        return {
            "available": False,
            "version": None,
            "features": [],
            "install_hint": "pip install yasa",
        }
