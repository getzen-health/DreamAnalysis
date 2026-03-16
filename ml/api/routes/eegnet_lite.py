"""EEGNet-Lite EEG emotion classification endpoint.

Exposes the EEGNetLiteClassifier (depthwise-separable 2-D CNN, ~1800 params,
ONNX export < 50 KB) as a FastAPI sub-router mounted at /eegnet-lite.

Endpoints:
    POST /eegnet-lite/classify        — classify EEG emotion (3-class valence)
    POST /eegnet-lite/fine-tune       — online last-layer SGD update (personalisation)
    GET  /eegnet-lite/status          — model status and architecture info
    GET  /eegnet-lite/history/{user}  — recent predictions for a user
    POST /eegnet-lite/reset/{user}    — clear per-user prediction history
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/eegnet-lite", tags=["eegnet-lite"])

# ── Lazy model loading ────────────────────────────────────────────────────────

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        try:
            from models.eegnet_lite import get_eegnet_lite_classifier
            _classifier = get_eegnet_lite_classifier()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"EEGNet-Lite model unavailable: {exc}",
            )
    return _classifier


# ── Per-user prediction history (last 500 per user, in-memory) ───────────────

_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

# ── Schemas ───────────────────────────────────────────────────────────────────


class EEGNetLiteRequest(BaseModel):
    """EEG data for EEGNet-Lite emotion classification.

    Attributes:
        signals:  2-D list of shape (n_channels, n_samples).
                  For Muse 2: 4 channels, 1024 samples (4 s @ 256 Hz).
                  Fewer samples are accepted (padded to 1024 internally).
                  Single-channel (n_samples,) is also accepted.
        fs:       EEG sampling rate in Hz. Default 256.0.
        user_id:  Identifier for per-user history.
    """

    signals: List[List[float]] = Field(
        ...,
        description=(
            "2-D EEG array [[ch0...], [ch1...], ...]. "
            "Shape (n_channels, n_samples). For Muse 2: 4 x 1024."
        ),
    )
    fs: float = Field(256.0, description="Sampling rate in Hz.")
    user_id: str = Field("default", description="User identifier for history.")


class EEGNetLiteResponse(BaseModel):
    """Classification result from EEGNet-Lite.

    Attributes:
        emotion:       Predicted class: "positive" | "neutral" | "negative".
        probabilities: Per-class probability (sums to 1.0).
        valence:       Continuous valence estimate in [-1, 1].
        model_type:    Inference path used.
        n_params:      Trainable parameter count of the model.
        user_id:       Echo of request user_id.
        n_channels:    Input channels received.
        n_samples:     Input samples per channel received.
        processed_at:  Unix timestamp.
        note:          Optional diagnostic string.
    """

    emotion: str
    probabilities: Dict[str, float]
    valence: float
    model_type: str
    n_params: int
    user_id: str
    n_channels: int
    n_samples: int
    processed_at: float
    note: Optional[str] = None


class FineTuneRequest(BaseModel):
    """Online last-layer personalisation request.

    Attributes:
        signals:  EEG epoch as 2-D list (n_channels, n_samples).
        label:    True class index: 0=positive, 1=neutral, 2=negative.
        lr:       SGD learning rate. Default 0.01.
        user_id:  User identifier.
        fs:       Sampling rate in Hz.
    """

    signals: List[List[float]] = Field(
        ..., description="EEG array (n_channels, n_samples)."
    )
    label: int = Field(..., ge=0, le=2, description="True class (0=pos, 1=neu, 2=neg).")
    lr: float = Field(0.01, gt=0.0, description="SGD learning rate.")
    user_id: str = Field("default")
    fs: float = Field(256.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_signals(raw: List[List[float]], endpoint: str) -> np.ndarray:
    """Parse and validate the signals field from a request."""
    try:
        signals = np.array(raw, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot parse signals: {exc}")

    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    if signals.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"{endpoint}: signals must be 1-D or 2-D, got shape {signals.shape}",
        )

    n_channels, n_samples = signals.shape
    if n_samples < 4:
        raise HTTPException(
            status_code=422,
            detail=f"{endpoint}: signals too short ({n_samples} samples, need >= 4)",
        )

    return signals


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/classify", response_model=EEGNetLiteResponse)
async def eegnet_lite_classify(req: EEGNetLiteRequest):
    """Classify EEG emotion using EEGNet-Lite (3-class valence).

    EEGNet-Lite is a compact depthwise-separable 2-D CNN with ~1,800 trainable
    parameters.  It can run entirely in the browser via onnxruntime-web once the
    model is exported to ONNX (< 50 KB).

    **Input format**: `signals` is a 2-D list `(n_channels, n_samples)`.
    For Muse 2 supply 4 channels x 1024 samples (4 s at 256 Hz).
    Shorter signals are padded automatically; longer signals are trimmed.

    **Output classes**:
    - `positive` — relaxed, happy, excited, calm
    - `neutral`  — alert, focused, baseline
    - `negative` — stressed, sad, fearful, anxious
    """
    signals = _parse_signals(req.signals, "/classify")
    n_channels, n_samples = signals.shape

    clf = _get_classifier()
    result = clf.predict(signals, fs=req.fs)

    response = EEGNetLiteResponse(
        emotion=result["emotion"],
        probabilities=result["probabilities"],
        valence=result.get("valence", 0.0),
        model_type=result.get("model_type", "feature-based"),
        n_params=result.get("n_params", 0),
        user_id=req.user_id,
        n_channels=n_channels,
        n_samples=n_samples,
        processed_at=time.time(),
        note=result.get("note"),
    )

    _history[req.user_id].append(response.dict())
    return response


@router.post("/fine-tune")
async def eegnet_lite_fine_tune(req: FineTuneRequest):
    """Online personalisation: one SGD step on the final layer.

    Freezes all layers except the classifier Linear, performs one gradient update
    using the provided EEG epoch and ground-truth label.  Call this after
    explicit user feedback ("I was actually feeling positive right now").

    Expected improvement: ~+7% accuracy after 20-50 labelled epochs per user
    (per ACM ISWC 2024 ablation).

    Returns the cross-entropy loss for the update step.
    """
    signals = _parse_signals(req.signals, "/fine-tune")

    clf = _get_classifier()
    result = clf.fine_tune_last_layer(
        signals, label=req.label, fs=req.fs, lr=req.lr
    )

    return {
        "user_id":  req.user_id,
        "label":    req.label,
        "lr":       req.lr,
        "updated":  result.get("updated", False),
        "loss":     result.get("loss"),
        "reason":   result.get("reason"),
    }


@router.get("/status")
async def eegnet_lite_status():
    """Return EEGNet-Lite model status and architecture information.

    Reports parameter count, inference path (ONNX vs PyTorch vs feature-based),
    and whether a trained checkpoint is loaded.
    """
    clf = _get_classifier()
    info = clf.get_model_info()
    return {
        "status": "ok",
        **info,
    }


@router.get("/history/{user_id}")
async def eegnet_lite_history(user_id: str, limit: int = 50):
    """Return recent EEGNet-Lite predictions for a user.

    Args:
        user_id: User identifier.
        limit:   Maximum predictions to return (default 50, max 500).
    """
    limit = min(limit, 500)
    entries = list(_history.get(user_id, []))
    return {
        "user_id": user_id,
        "count": len(entries),
        "predictions": entries[-limit:],
    }


@router.post("/reset/{user_id}")
async def eegnet_lite_reset(user_id: str):
    """Clear EEGNet-Lite prediction history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "cleared"}
