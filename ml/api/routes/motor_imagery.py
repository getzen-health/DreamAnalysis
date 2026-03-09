"""Motor imagery BCI API.

GitHub issue: #123
"""
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ._shared import EEGInput, _numpy_safe
from models.motor_imagery import MotorImageryClassifier

router = APIRouter(tags=["motor-imagery"])

_classifier = MotorImageryClassifier()


class LabelInput(BaseModel):
    true_label: str = Field(..., description="True imagined movement label (left_hand/right_hand/feet/rest)")
    user_id: str = Field("default", description="User identifier")


@router.post("/motor-imagery/set-baseline")
async def set_motor_baseline(data: EEGInput):
    """Record resting-state baseline for ERD computation.

    Call while the user is at rest (no imagined movement).
    More accurate ERD detection with a good baseline.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    result = _classifier.set_baseline(eeg_signals=signals, fs=data.fs, user_id=data.user_id)
    return _numpy_safe(result)


@router.post("/motor-imagery/classify")
async def classify_motor_imagery(data: EEGInput):
    """Classify imagined movement from EEG mu/beta desynchronization.

    Returns predicted_class (left_hand/right_hand/feet/rest),
    probabilities, confidence, laterality_index, mu_suppression, and erd_map.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    result = _classifier.classify(eeg_signals=signals, fs=data.fs, user_id=data.user_id)
    result["user_id"] = data.user_id
    return _numpy_safe(result)


@router.post("/motor-imagery/label")
async def submit_motor_label(data: LabelInput):
    """Submit ground truth label for the last classification (for accuracy tracking)."""
    valid_labels = {"left_hand", "right_hand", "feet", "rest"}
    if data.true_label not in valid_labels:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail=f"true_label must be one of {sorted(valid_labels)}",
        )
    _classifier.submit_label(true_label=data.true_label, user_id=data.user_id)
    return {"status": "ok", "label": data.true_label, "user_id": data.user_id}


@router.get("/motor-imagery/accuracy")
async def get_motor_accuracy(user_id: str = "default"):
    """Get classification accuracy for a user (requires labeled history)."""
    accuracy = _classifier.get_accuracy(user_id=user_id)
    return {"accuracy": accuracy, "user_id": user_id}


@router.get("/motor-imagery/stats")
async def get_motor_stats(user_id: str = "default"):
    """Get per-class statistics for motor imagery classification."""
    result = _classifier.get_session_stats(user_id=user_id)
    result["user_id"] = user_id
    return _numpy_safe(result)


@router.get("/motor-imagery/history")
async def get_motor_history(user_id: str = "default", last_n: int = 50):
    """Get classification history for a user."""
    history = _classifier.get_history(user_id=user_id, last_n=last_n)
    return _numpy_safe({"history": history, "user_id": user_id, "count": len(history)})


@router.post("/motor-imagery/reset")
async def reset_motor_imagery(user_id: str = "default"):
    """Clear all motor imagery state for a user."""
    _classifier.reset(user_id=user_id)
    return {"status": "ok", "message": "Motor imagery state cleared.", "user_id": user_id}
