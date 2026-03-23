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


@router.get("/sleep/yasa-status")
async def yasa_status():
    """Check if YASA is available and return version info."""
    try:
        import yasa
        return {
            "available": True,
            "version": yasa.__version__,
            "features": ["sleep_staging", "spindle_detection", "slow_wave_detection"],
        }
    except ImportError:
        return {
            "available": False,
            "version": None,
            "features": [],
            "install_hint": "pip install yasa",
        }
