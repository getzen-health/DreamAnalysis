"""CNN-KAN EEG emotion classification endpoint.

Exposes the CNNKANClassifier (Conv1D backbone + KAN head with B-spline
learnable activations) as a FastAPI sub-router mounted at /cnn-kan.

Endpoints:
    POST /cnn-kan/classify      — classify EEG emotion (3-class valence)
    GET  /cnn-kan/status        — model status and architecture info
    POST /cnn-kan/reset/{user}  — clear per-user prediction history
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/cnn-kan", tags=["cnn-kan"])

# ── Lazy model loading (avoids import-time delay if torch absent) ─────────────

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        try:
            from models.cnn_kan import get_cnn_kan_classifier
            _classifier = get_cnn_kan_classifier()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"CNN-KAN model unavailable: {exc}",
            )
    return _classifier


# ── In-memory prediction history (last 500 predictions per user) ─────────────

_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

# ── Schemas ───────────────────────────────────────────────────────────────────


class CNNKANRequest(BaseModel):
    """EEG data for CNN-KAN emotion classification.

    Attributes:
        signals:   2D list of shape (n_channels, n_samples). For Muse 2:
                   4 channels, 1024 samples (4 s @ 256 Hz). Fewer samples are
                   accepted but accuracy degrades below 256 samples.
                   Single-channel (n_samples,) is also accepted.
        fs:        EEG sampling rate in Hz. Default 256.0.
        user_id:   Identifier for per-user history and caching.
    """

    signals: List[List[float]] = Field(
        ...,
        description="2D EEG array [[ch0...], [ch1...], ...]. Shape: (n_channels, n_samples).",
    )
    fs: float = Field(256.0, description="Sampling rate in Hz.")
    user_id: str = Field("default", description="User identifier for caching.")


class CNNKANResponse(BaseModel):
    """Classification result from CNN-KAN.

    Attributes:
        emotion:       Predicted class: "positive" | "neutral" | "negative".
        probabilities: Probability over each class (sums to 1.0).
        valence:       Continuous valence estimate in [-1, 1].
        model_type:    "cnn-kan" (weights loaded) or "feature-based" (fallback).
        user_id:       Echo of the request user_id.
        n_channels:    Number of input channels received.
        n_samples:     Number of samples per channel received.
        processed_at:  Unix timestamp of classification.
        note:          Optional diagnostic note (e.g. signal too short).
    """

    emotion: str
    probabilities: Dict[str, float]
    valence: float
    model_type: str
    user_id: str
    n_channels: int
    n_samples: int
    processed_at: float
    note: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/classify", response_model=CNNKANResponse)
async def cnn_kan_classify(req: CNNKANRequest):
    """Classify EEG emotion using the CNN-KAN hybrid model.

    Accepts raw EEG signal arrays and returns 3-class valence prediction
    (positive / neutral / negative) with per-class probabilities and a
    continuous valence estimate.

    The model uses a Conv1D backbone for temporal feature extraction followed
    by a KAN (Kolmogorov-Arnold Network) classification head with learnable
    B-spline activation functions. Falls back to feature-based heuristics
    (band-power + FAA) when PyTorch is unavailable.

    **Input format**: `signals` should be a 2D list of shape
    `(n_channels, n_samples)` — e.g. `[[...ch0 samples...], [...ch1...], ...]`.
    For Muse 2: 4 channels × 1024 samples (4 seconds at 256 Hz).
    """
    # ── Validate and parse signals ────────────────────────────────────────
    try:
        signals = np.array(req.signals, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot parse signals: {exc}")

    if signals.ndim == 1:
        signals = signals[np.newaxis, :]  # treat as single-channel

    if signals.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"signals must be 1D or 2D, got shape {signals.shape}",
        )

    n_channels, n_samples = signals.shape

    if n_samples < 4:
        raise HTTPException(
            status_code=422,
            detail=f"signals too short ({n_samples} samples). Need at least 4.",
        )

    # ── Run classifier ────────────────────────────────────────────────────
    clf = _get_classifier()
    result = clf.predict(signals, fs=req.fs)

    # ── Build response ────────────────────────────────────────────────────
    response = CNNKANResponse(
        emotion=result["emotion"],
        probabilities=result["probabilities"],
        valence=result.get("valence", 0.0),
        model_type=result.get("model_type", "feature-based"),
        user_id=req.user_id,
        n_channels=n_channels,
        n_samples=n_samples,
        processed_at=time.time(),
        note=result.get("note"),
    )

    # Store in history
    _history[req.user_id].append(response.dict())
    return response


@router.get("/status")
async def cnn_kan_status():
    """Return CNN-KAN model status and architecture information.

    Includes parameter counts, KAN B-spline configuration, and whether
    a trained checkpoint is loaded (vs. untrained / feature-based fallback).
    """
    clf = _get_classifier()
    info = clf.get_model_info()
    return {
        "status": "ok",
        "model_loaded": clf._model is not None,
        **info,
    }


@router.get("/history/{user_id}")
async def cnn_kan_history(user_id: str, limit: int = 50):
    """Return recent CNN-KAN predictions for a user.

    Args:
        user_id: User identifier.
        limit:   Maximum number of predictions to return (default 50, max 500).
    """
    limit = min(limit, 500)
    entries = list(_history.get(user_id, []))
    return {
        "user_id": user_id,
        "count": len(entries),
        "predictions": entries[-limit:],
    }


@router.post("/reset/{user_id}")
async def cnn_kan_reset(user_id: str):
    """Clear CNN-KAN prediction history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "cleared"}
