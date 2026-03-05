"""Personal model calibration and feedback endpoints (Phase 9)."""

import threading
import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List

log = logging.getLogger(__name__)

# Trigger fine-tune at 30 samples, then every 20 after that
_FINE_TUNE_FIRST = 30
_FINE_TUNE_INTERVAL = 20


def _should_fine_tune(n: int) -> bool:
    """True at the first milestone (30) and every 20 samples beyond."""
    if n < _FINE_TUNE_FIRST:
        return False
    if n == _FINE_TUNE_FIRST:
        return True
    return (n - _FINE_TUNE_FIRST) % _FINE_TUNE_INTERVAL == 0


def _fine_tune_bg(pm) -> None:
    """Run fine_tune() + save() in a daemon thread — non-blocking."""
    try:
        acc = pm.fine_tune()
        pm.save()
        log.info("Background fine-tune complete for %s — val_acc=%.1f%%", pm.user_id, acc * 100)
    except Exception as exc:
        log.warning("Background fine-tune failed for %s: %s", pm.user_id, exc)

from ._shared import (
    _get_personal_model,
    get_last_features,
    CalibrationSubmitRequest, PersonalFeedbackRequest,
)
from processing.eeg_processor import BaselineCalibrator


def _get_personal_model_for_feedback(user_id: str, n_channels: int = 4):
    """Return the PersonalModel (EEGNet-backed, fine-tunable) for a user.

    Distinct from _get_personal_model() which returns the SGD-based
    PersonalModelAdapter used for the online-learner blend in predict_emotion().
    The PersonalModel has add_labeled_epoch(), fine_tune(), and save() which
    are required by the /feedback and /calibration/status endpoints.
    """
    try:
        from models.personal_model import get_personal_model as _gpm
        return _gpm(user_id, n_channels=n_channels)
    except Exception:
        return None

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
    """User corrects a prediction — records correction and optionally adds labeled epoch.

    Signals are optional: label-only corrections (from the web UI) are recorded
    in the FeedbackCollector JSONL store and count toward the correction log.
    When signals ARE provided (e.g. from a live session), the labeled epoch is
    also added to the PersonalModel buffer so fine-tuning can use it.
    """
    from processing.user_feedback import FeedbackCollector

    # Always record the label correction (no EEG needed).
    # Attach the last cached feature vector so the k-NN PersonalizedPipeline
    # has training data even for label-only corrections (no raw EEG submitted).
    collector = FeedbackCollector(request.user_id)
    cached_features = get_last_features(request.user_id)
    collector.record_state_correction(
        model_name="emotion",
        predicted_state=request.predicted_label,
        corrected_state=request.correct_label,
        features=cached_features,
    )

    labeled = False
    fine_tune_triggered = False

    if request.signals is not None:
        # EEG provided — add to personal model buffer and maybe auto-fine-tune.
        # Uses PersonalModel (EEGNet-backed) which has add_labeled_epoch() + fine_tune().
        n_channels = len(request.signals) if request.signals else 4
        pm = _get_personal_model_for_feedback(request.user_id, n_channels=n_channels)
        if pm is not None:
            label_map = {"happy": 0, "sad": 1, "angry": 2, "fearful": 3, "relaxed": 4, "focused": 5}
            label_idx = label_map.get(request.correct_label, -1)
            if label_idx >= 0:
                signal = np.array(request.signals)
                pm.add_labeled_epoch(signal, label_idx)
                labeled = True
                n = len(pm._buffer_y)
                if _should_fine_tune(n):
                    fine_tune_triggered = True
                    t = threading.Thread(target=_fine_tune_bg, args=(pm,), daemon=True)
                    t.start()

    return {
        "recorded": True,
        "labeled_epoch_added": labeled,
        "fine_tune_triggered": fine_tune_triggered,
        "user_id": request.user_id,
        "predicted": request.predicted_label,
        "corrected": request.correct_label,
    }


@router.get("/calibration/status")
async def calibration_status(user_id: str = "default"):
    """Check personal model calibration status.

    Returns PersonalModel.status() — includes buffer size, head accuracy,
    session count, and whether the personal head is active.
    """
    pm = _get_personal_model_for_feedback(user_id)
    if pm is None:
        return {
            "calibrated": False,
            "personal_model_active": False,
            "n_samples": 0,
            "personal_accuracy": 0.0,
            "classes": [],
        }
    return pm.status()


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
