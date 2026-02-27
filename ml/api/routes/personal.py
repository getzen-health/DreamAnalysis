"""Personal model API — per-user EEGNet fine-tuning endpoints.

These endpoints let the dashboard:
  1. Submit a labeled EEG epoch (user corrected an emotion label)
  2. Check personalisation progress (how many sessions, accuracy so far)
  3. Trigger fine-tuning on demand
  4. Submit a resting-state frame to build the personal baseline
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.personal_model import get_personal_model, EMOTIONS

log = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────

class LabeledEpochRequest(BaseModel):
    """Submit one labeled EEG epoch to the personal model buffer."""
    user_id: str = "default"
    signals: List[List[float]]   # (n_channels, n_samples) nested list
    label: int                   # 0–5 index into EMOTIONS list
    fs: float = 256.0


class BaselineFrameRequest(BaseModel):
    """Submit one feature vector frame during resting-state baseline recording."""
    user_id: str = "default"
    features: List[float]        # feature vector from extract_features()


class FineTuneRequest(BaseModel):
    user_id: str = "default"
    n_channels: int = 4


class PredictPersonalRequest(BaseModel):
    user_id: str = "default"
    signals: List[List[float]]   # (n_channels, n_samples)
    fs: float = 256.0


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/personal/label-epoch")
async def label_epoch(req: LabeledEpochRequest):
    """Add a user-labeled EEG epoch to the personal training buffer.

    Call this when:
    - User taps a different emotion label on the dashboard (correction)
    - User rates the session with 😊 / 😐 / 😣 (maps to happy/neutral/stressed)
    - System infers a label (e.g., user opened biofeedback → stress was high)
    """
    if req.label < 0 or req.label >= len(EMOTIONS):
        raise HTTPException(
            status_code=400,
            detail=f"label must be 0–{len(EMOTIONS)-1}. Emotions: {EMOTIONS}"
        )

    eeg = np.array(req.signals, dtype=np.float32)
    if eeg.ndim != 2 or eeg.shape[0] < 1 or eeg.shape[1] < 64:
        raise HTTPException(status_code=422, detail="signals must be (n_channels, n_samples≥64)")

    pm = get_personal_model(req.user_id, n_channels=eeg.shape[0])
    pm.add_labeled_epoch(eeg, req.label)

    return {
        "status": "added",
        "user_id": req.user_id,
        "label": req.label,
        "emotion": EMOTIONS[req.label],
        "buffer_size": len(pm._buffer_y),
        "message": pm.status()["message"],
    }


@router.post("/personal/baseline-frame")
async def add_baseline_frame(req: BaselineFrameRequest):
    """Add one resting-state feature frame to build the personal EEG baseline.

    Call this once per second during the 2-minute eyes-closed resting period
    at the start of a session. The baseline calibrates z-score normalisation
    to this user's individual EEG amplitude, correcting for skull thickness,
    hair, and electrode contact differences.
    """
    features = np.array(req.features, dtype=np.float32)
    pm = get_personal_model(req.user_id)
    pm.baseline.add_resting_frame(features)

    if pm.baseline.is_ready:
        pm.baseline.save(req.user_id)

    return {
        "frames_collected": pm.baseline.n_frames_collected,
        "baseline_ready": pm.baseline.is_ready,
        "frames_needed": max(0, 30 - pm.baseline.n_frames_collected),
    }


@router.get("/personal/status")
async def personal_status(user_id: str = "default", n_channels: int = 4):
    """Return personalisation progress for the dashboard.

    Shown as a progress card: "Personal model: 45/100 epochs collected"
    """
    pm = get_personal_model(user_id, n_channels=n_channels)
    return pm.status()


@router.post("/personal/fine-tune")
async def fine_tune(req: FineTuneRequest):
    """Trigger fine-tuning of the personal head on buffered data.

    This is called automatically by mark_session_complete(), but can also
    be triggered manually from the settings page.
    """
    pm = get_personal_model(req.user_id, n_channels=req.n_channels)
    accuracy = pm.fine_tune()
    pm.save()

    return {
        "status": "fine_tuned" if accuracy > 0 else "insufficient_data",
        "val_accuracy_pct": round(accuracy * 100, 1),
        "buffer_size": len(pm._buffer_y),
        "personal_model_active": pm._personal_head_ready(),
        "message": pm.status()["message"],
    }


@router.post("/personal/session-complete")
async def session_complete(user_id: str = "default", n_channels: int = 4):
    """Mark a session as complete — triggers fine-tuning if enough new data."""
    pm = get_personal_model(user_id, n_channels=n_channels)
    pm.mark_session_complete()
    return pm.status()


@router.post("/personal/predict")
async def predict_personal(req: PredictPersonalRequest):
    """Predict emotion using personal model (if active) or central model.

    Same response format as /analyze-eeg, with an added 'model_type' field
    showing whether 'personal_4ch' or 'central_4ch' was used.
    """
    eeg = np.array(req.signals, dtype=np.float32)
    if eeg.ndim != 2 or eeg.shape[1] < 64:
        raise HTTPException(status_code=422, detail="signals must be (n_channels, n_samples≥64)")

    pm = get_personal_model(req.user_id, n_channels=eeg.shape[0])
    result = pm.predict(eeg, fs=req.fs)
    result["user_id"] = req.user_id
    result["personal_model_active"] = pm._personal_head_ready()
    return result


@router.delete("/personal/reset")
async def reset_personal_model(user_id: str = "default"):
    """Clear all personal model data for a user (buffer, head, baseline).

    Use this if a user wants to start fresh or if data quality was poor.
    """
    import shutil
    from pathlib import Path
    path = Path(__file__).parent.parent.parent / "models" / "saved" / "personal" / user_id
    if path.exists():
        shutil.rmtree(path)

    # Remove from registry
    from models.personal_model import _registry
    keys_to_remove = [k for k in _registry if k.startswith(f"{user_id}_")]
    for k in keys_to_remove:
        del _registry[k]

    return {"status": "reset", "user_id": user_id}
