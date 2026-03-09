"""Mental fatigue monitoring API endpoints.

Endpoints:
    POST /fatigue/analyze              — Analyze EEG epoch for fatigue index
    POST /fatigue/calibrate-baseline   — Accumulate epoch for baseline calibration
    GET  /fatigue/session-curve        — Return rolling fatigue curve
    GET  /fatigue/status               — Current monitor state
    POST /fatigue/reset                — Reset session (keep baseline)
    POST /fatigue/reset-baseline       — Full reset including baseline
"""

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Mental Fatigue"])


# ── Request models ────────────────────────────────────────────────────────────

class FatigueAnalyzeRequest(BaseModel):
    eeg: List[List[float]] = Field(
        ..., description="EEG data: shape [n_channels, n_samples] or [n_samples]"
    )
    fs: float = Field(default=256.0, gt=0, description="Sampling rate in Hz")
    session_minutes: float = Field(
        default=0.0, ge=0, description="Minutes elapsed since session start"
    )


class FatigueCalibrateRequest(BaseModel):
    eeg: List[List[float]] = Field(
        ..., description="EEG epoch for baseline calibration"
    )
    fs: float = Field(default=256.0, gt=0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/fatigue/analyze")
async def fatigue_analyze(request: FatigueAnalyzeRequest):
    """Compute mental fatigue index for one EEG epoch.

    Returns fatigue_index (0–1), fatigue_stage, theta/beta ratio,
    trend slope, and break recommendation.
    """
    try:
        from models.fatigue_monitor import get_fatigue_monitor
        monitor = get_fatigue_monitor()
        eeg = np.array(request.eeg, dtype=np.float32)
        result = monitor.predict(eeg, fs=request.fs, session_minutes=request.session_minutes)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/fatigue/calibrate-baseline")
async def fatigue_calibrate(request: FatigueCalibrateRequest):
    """Accumulate one EEG epoch for fatigue baseline calibration.

    Call once per epoch during the first 2 minutes of the session.
    Returns whether baseline is now ready (requires ≥10 frames).
    """
    try:
        from models.fatigue_monitor import get_fatigue_monitor
        monitor = get_fatigue_monitor()
        eeg = np.array(request.eeg, dtype=np.float32)
        result = monitor.calibrate_baseline(eeg, fs=request.fs)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fatigue/session-curve")
async def fatigue_session_curve():
    """Return the rolling fatigue curve for the current session."""
    try:
        from models.fatigue_monitor import get_fatigue_monitor
        monitor = get_fatigue_monitor()
        status = monitor.get_status()
        return {
            "current_fatigue_index": status["current_fatigue_index"],
            "history_length": status["history_length"],
            "baseline_calibrated": status["baseline_calibrated"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fatigue/status")
async def fatigue_status():
    """Current fatigue monitor status."""
    try:
        from models.fatigue_monitor import get_fatigue_monitor
        return get_fatigue_monitor().get_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/fatigue/reset")
async def fatigue_reset():
    """Reset session history (keeps baseline for next session)."""
    try:
        from models.fatigue_monitor import get_fatigue_monitor
        get_fatigue_monitor().reset_session()
        return {"success": True, "message": "Session reset. Baseline preserved."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/fatigue/reset-baseline")
async def fatigue_reset_baseline():
    """Full reset including baseline — use at start of a new calibration."""
    try:
        from models.fatigue_monitor import get_fatigue_monitor
        get_fatigue_monitor().reset_baseline()
        return {"success": True, "message": "Full reset. Recalibrate baseline before analyzing."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
