"""Personal model calibration and feedback endpoints (Phase 9)."""

import threading
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List

from ._shared import (
    _get_personal_model,
    CalibrationSubmitRequest, PersonalFeedbackRequest,
    EEGInput,
)
from processing.eeg_processor import BaselineCalibrator

router = APIRouter()

# Per-user BaselineCalibrator instances, keyed by user_id
_baseline_cals: Dict[str, BaselineCalibrator] = {}
_baseline_cals_lock = threading.Lock()

def _get_baseline_cal(user_id: str) -> BaselineCalibrator:
    """Return the per-user BaselineCalibrator, creating it on first use."""
    with _baseline_cals_lock:
        if user_id not in _baseline_cals:
            _baseline_cals[user_id] = BaselineCalibrator()
        return _baseline_cals[user_id]


class BaselineFrameRequest(BaseModel):
    signals: List[List[float]]   # shape: (n_channels, n_samples)
    fs: float = 256.0
    user_id: str = "default"


@router.post("/calibration/start")
async def start_calibration():
    """Start a calibration session (returns prompts for the user)."""
    return {
        "status": "started",
        "steps": [
            {"step": 1, "instruction": "Close your eyes and relax for 30 seconds", "label": "relaxed", "duration_sec": 30},
            {"step": 2, "instruction": "Focus on counting backwards from 100", "label": "focused", "duration_sec": 30},
            {"step": 3, "instruction": "Think about something stressful", "label": "stressed", "duration_sec": 30},
        ],
    }


@router.post("/calibration/submit")
async def submit_calibration(request: CalibrationSubmitRequest):
    """Submit labeled calibration data to train personal model."""
    pm = _get_personal_model(request.user_id)
    if pm is None:
        raise HTTPException(status_code=500, detail="Failed to initialize personal model")

    signals_list = [np.array(s) for s in request.signals_list]
    return pm.calibrate(signals_list, request.labels, request.fs)


@router.post("/feedback")
async def submit_personal_feedback(request: PersonalFeedbackRequest):
    """User corrects a prediction — triggers incremental personal model update."""
    pm = _get_personal_model(request.user_id)
    if pm is None:
        raise HTTPException(status_code=500, detail="Failed to initialize personal model")

    signal = np.array(request.signals)
    return pm.adapt(signal, request.predicted_label, request.correct_label, request.fs)


@router.get("/calibration/status")
async def calibration_status(user_id: str = "default"):
    """Check personal model calibration status."""
    pm = _get_personal_model(user_id)
    if pm is None:
        return {"calibrated": False, "n_samples": 0, "personal_accuracy": 0.0, "classes": []}
    return pm.get_calibration_status()


# ─── Baseline calibration endpoints ─────────────────────────────────────────
# Used during the 2-3 minute resting-state recording at session start.
# Call add-frame repeatedly (e.g. every second) while user sits still.
# Once 30+ frames are collected the baseline is ready; call normalize on
# each feature dict before passing to the emotion classifier.

@router.post("/calibration/baseline/add-frame")
async def baseline_add_frame(request: BaselineFrameRequest):
    """Add a resting-state EEG frame to the baseline calibrator.

    Send ~1 second of EEG at a time (256 samples at 256 Hz).
    Collect 2 minutes of eyes-closed data for best results.

    Returns the current baseline status.
    """
    try:
        signals = np.array(request.signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        cal = _get_baseline_cal(request.user_id)
        cal.add_baseline_frame(signals, request.fs)
        status = cal.to_dict()
        ready = status["is_ready"]
        return {
            "status": "ok",
            "n_frames": status["n_frames"],
            "ready": ready,
            "message": (
                "Baseline ready — normalize features before classification."
                if ready
                else f"Collecting baseline… {status['n_frames']}/{BaselineCalibrator._MIN_BASELINE_FRAMES} frames."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calibration/baseline/status")
async def baseline_status(user_id: str = "default"):
    """Return current baseline calibrator status."""
    status = _get_baseline_cal(user_id).to_dict()
    return {
        "n_frames": status["n_frames"],
        "ready": status["is_ready"],
        "n_features": len(status.get("mean", {})),
    }


@router.post("/calibration/baseline/reset")
async def baseline_reset(user_id: str = "default"):
    """Clear all collected baseline frames and start over."""
    _get_baseline_cal(user_id).reset()
    return {"status": "reset", "message": "Baseline calibrator cleared. Collect a fresh 2-min resting baseline."}
