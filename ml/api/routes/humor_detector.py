"""Humor and laughter response detector API (#181)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/humor-detector", tags=["humor-detector"])


class HumorInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = Field(..., min_length=1)


class HumorResult(BaseModel):
    user_id: str
    humor_level: str
    humor_score: float
    alpha_synchronization: float
    gamma_insight_index: float
    approach_motivation_faa: float
    theta_incongruity: float
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=HumorResult)
async def analyze_humor_response(req: HumorInput):
    """Detect humor and laughter response from frontal alpha-gamma EEG."""
    from models.humor_detector import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = HumorResult(
        user_id=req.user_id,
        humor_level=result["humor_level"],
        humor_score=result["humor_score"],
        alpha_synchronization=result["alpha_synchronization"],
        gamma_insight_index=result["gamma_insight_index"],
        approach_motivation_faa=result["approach_motivation_faa"],
        theta_incongruity=result["theta_incongruity"],
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
