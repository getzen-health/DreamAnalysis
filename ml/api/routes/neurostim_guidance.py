"""Closed-loop neurostimulation guidance API (#163)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/neurostim", tags=["neurostim-guidance"])


class NeurostimInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    target_protocol: str = "alpha_entrainment"
    user_id: str = Field(..., min_length=1)


class NeurostimResult(BaseModel):
    user_id: str
    protocol: str
    stim_frequency_hz: float
    suggested_intensity_normalized: float
    iaf_hz: float
    readiness_score: float
    should_stimulate: bool
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=NeurostimResult)
async def analyze_neurostim(req: NeurostimInput):
    """Compute closed-loop neurostimulation parameters from EEG state."""
    from models.neurostim_guidance import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs, req.target_protocol)

    out = NeurostimResult(
        user_id=req.user_id,
        protocol=result["protocol"],
        stim_frequency_hz=result["stim_frequency_hz"],
        suggested_intensity_normalized=result["suggested_intensity_normalized"],
        iaf_hz=result["iaf_hz"],
        readiness_score=result["readiness_score"],
        should_stimulate=result["should_stimulate"],
        model_used=result["model_used"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent neurostimulation guidance history for a user."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear neurostimulation guidance history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
