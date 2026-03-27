"""Temporal EEG emotion analysis endpoints.

Accumulates epoch feature vectors in a sliding buffer per user and returns
temporal-LSTM or heuristic emotion predictions once enough epochs are
buffered.

Endpoints
---------
POST /temporal-emotion/add-epoch
    Push one epoch's feature vector into the per-user buffer.
    Returns prediction when the buffer is full.

GET  /temporal-emotion/status/{user_id}
    Return buffer status (epochs buffered, is_ready, temporal_features).

POST /temporal-emotion/reset/{user_id}
    Clear the per-user buffer.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.temporal_emotion_model import TemporalEmotionClassifier

router = APIRouter(prefix="/temporal-emotion", tags=["Temporal Emotion"])

# ---------------------------------------------------------------------------
# Per-user classifier instances (thread-safe)
# ---------------------------------------------------------------------------

_classifiers: Dict[str, TemporalEmotionClassifier] = {}
_classifiers_lock = threading.Lock()

_DEFAULT_SEQ_LENGTH = 10
_DEFAULT_INPUT_DIM = 41


def _get_classifier(user_id: str) -> TemporalEmotionClassifier:
    """Return (creating if needed) the per-user temporal classifier."""
    with _classifiers_lock:
        if user_id not in _classifiers:
            _classifiers[user_id] = TemporalEmotionClassifier(
                model_path=None,
                seq_length=_DEFAULT_SEQ_LENGTH,
                input_dim=_DEFAULT_INPUT_DIM,
            )
        return _classifiers[user_id]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TemporalEpochInput(BaseModel):
    """One epoch's feature vector to push into the temporal buffer."""

    user_id: str = Field(default="default", description="User identifier")
    features: List[float] = Field(
        ..., description="Feature vector from one 4-second epoch"
    )


class TemporalPredictionResponse(BaseModel):
    """Response from the temporal emotion endpoint."""

    user_id: str
    epochs_buffered: int
    is_ready: bool
    prediction: Optional[Dict] = None


class TemporalStatusResponse(BaseModel):
    """Buffer status for a user."""

    user_id: str
    epochs_buffered: int
    seq_length: int
    is_ready: bool
    temporal_features: Dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/add-epoch", response_model=TemporalPredictionResponse)
async def add_temporal_epoch(request: TemporalEpochInput):
    """Add one epoch's features to the temporal buffer.

    Returns a prediction when the buffer has accumulated enough epochs
    (default: 10).  Before that, ``prediction`` is null.
    """
    clf = _get_classifier(request.user_id)
    features = np.array(request.features, dtype=np.float32)
    clf.add_epoch(features)

    prediction = clf.predict()  # None if not ready

    return TemporalPredictionResponse(
        user_id=request.user_id,
        epochs_buffered=len(clf.buffer),
        is_ready=clf.is_ready(),
        prediction=prediction,
    )


@router.get("/status/{user_id}", response_model=TemporalStatusResponse)
async def temporal_status(user_id: str):
    """Return current buffer status and temporal features for a user."""
    clf = _get_classifier(user_id)
    return TemporalStatusResponse(
        user_id=user_id,
        epochs_buffered=len(clf.buffer),
        seq_length=clf.seq_length,
        is_ready=clf.is_ready(),
        temporal_features=clf._compute_temporal_features(),
    )


@router.post("/reset/{user_id}")
async def temporal_reset(user_id: str):
    """Clear the temporal buffer for a user."""
    clf = _get_classifier(user_id)
    clf.reset()
    return {"user_id": user_id, "status": "reset", "epochs_buffered": 0}
