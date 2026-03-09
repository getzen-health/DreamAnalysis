"""EmoAdapt self-supervised emotion personalization endpoints."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.emo_adapt import get_emo_adapt_learner

router = APIRouter()


# ── Pydantic request/response models ─────────────────────────────────────────

class EmoAdaptPredictRequest(BaseModel):
    signals: List[List[float]] | List[float] = Field(
        ...,
        description="EEG signal: 1D list (n_samples,) or 2D list (n_channels, n_samples).",
    )
    fs: float = Field(256.0, description="Sampling frequency in Hz.")


class EmoAdaptUpdateRequest(BaseModel):
    signals: List[List[float]] | List[float] = Field(
        ...,
        description="EEG signal: 1D list (n_samples,) or 2D list (n_channels, n_samples).",
    )
    emotion_label: str = Field(
        ...,
        description="Labeled emotion: one of happy, sad, angry, fear, surprise, neutral.",
    )
    fs: float = Field(256.0, description="Sampling frequency in Hz.")


def _to_numpy(signals) -> np.ndarray:
    """Convert nested list input to numpy float32 array."""
    arr = np.array(signals, dtype=np.float32)
    return arr


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/emo-adapt/predict")
async def emo_adapt_predict(req: EmoAdaptPredictRequest):
    """Adapted emotion prediction using prototype bank + heuristic blend.

    Returns emotion label, probability distribution, valence, arousal,
    adaptation_gain (0=pure heuristic, 1=full adaptation), and n_updates.
    """
    learner = get_emo_adapt_learner()
    eeg = _to_numpy(req.signals)
    try:
        result = learner.predict(eeg, fs=req.fs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@router.post("/emo-adapt/update")
async def emo_adapt_update(req: EmoAdaptUpdateRequest):
    """Add a labeled EEG example to the prototype bank.

    Updates the running EMA prototype for the given emotion class.
    Returns updated_prototypes count and the label that was updated.
    """
    learner = get_emo_adapt_learner()
    eeg = _to_numpy(req.signals)
    try:
        result = learner.update_prototype(eeg, req.emotion_label, fs=req.fs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@router.get("/emo-adapt/status")
async def emo_adapt_status():
    """Return adaptation state.

    Keys: n_updates_per_class, n_updates_total, n_prototypes,
          adaptation_ready, adaptation_gain.
    """
    learner = get_emo_adapt_learner()
    return learner.get_status()


@router.post("/emo-adapt/reset")
async def emo_adapt_reset():
    """Reset prototype bank and all update counters.

    Returns status='reset' and n_updates=0.
    """
    learner = get_emo_adapt_learner()
    return learner.reset()
