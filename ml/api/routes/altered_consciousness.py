"""Altered consciousness state detection API (#161)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/altered-consciousness", tags=["altered-consciousness"])


class AlteredConsciousnessInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = Field(..., min_length=1)


class AlteredConsciousnessResult(BaseModel):
    user_id: str
    state: str
    altered_consciousness_index: float
    theta_fraction: float
    alpha_fraction: float
    beta_suppression: float
    spectral_entropy: float
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=AlteredConsciousnessResult)
async def analyze_altered_consciousness(req: AlteredConsciousnessInput):
    """Detect altered consciousness state from EEG spectral features."""
    from models.altered_consciousness import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = AlteredConsciousnessResult(
        user_id=req.user_id,
        state=result["state"],
        altered_consciousness_index=result["altered_consciousness_index"],
        theta_fraction=result["theta_fraction"],
        alpha_fraction=result["alpha_fraction"],
        beta_suppression=result["beta_suppression"],
        spectral_entropy=result["spectral_entropy"],
        model_used=result["model_used"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent altered consciousness analysis history for a user."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear altered consciousness history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
