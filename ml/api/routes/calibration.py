"""Personal model calibration and feedback endpoints (Phase 9)."""

import numpy as np
from fastapi import APIRouter, HTTPException

from ._shared import (
    _get_personal_model,
    CalibrationSubmitRequest, PersonalFeedbackRequest,
)

router = APIRouter()


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
