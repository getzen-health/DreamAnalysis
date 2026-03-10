"""Neuroaesthetic response detector API (#177)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/neuroaesthetic", tags=["neuroaesthetic"])


class NeuroaestheticInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = Field(..., min_length=1)


class NeuroaestheticResult(BaseModel):
    user_id: str
    aesthetic_response_level: str
    aesthetic_score: float
    alpha_synchronization: float
    theta_engagement: float
    gamma_burst_index: float
    approach_motivation_faa: float
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=NeuroaestheticResult)
async def analyze_neuroaesthetic(req: NeuroaestheticInput):
    """Detect neuroaesthetic response to art and beauty stimuli."""
    from models.neuroaesthetic_detector import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = NeuroaestheticResult(
        user_id=req.user_id,
        aesthetic_response_level=result["aesthetic_response_level"],
        aesthetic_score=result["aesthetic_score"],
        alpha_synchronization=result["alpha_synchronization"],
        theta_engagement=result["theta_engagement"],
        gamma_burst_index=result["gamma_burst_index"],
        approach_motivation_faa=result["approach_motivation_faa"],
        model_used=result["model_used"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
