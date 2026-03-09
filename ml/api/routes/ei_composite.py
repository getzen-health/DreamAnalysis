"""Emotional Intelligence composite score API (#191)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/ei-composite", tags=["ei-composite"])


class EICompositeInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class EICompositeResult(BaseModel):
    user_id: str
    ei_level: str
    ei_composite_score: float
    emotional_awareness: float
    emotional_regulation: float
    empathy_index: float
    emotional_memory: float
    self_regulation: float
    frontal_asymmetry_faa: float
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=EICompositeResult)
async def analyze_ei_composite(req: EICompositeInput):
    """Compute EI composite score from multi-metric EEG emotional intelligence index."""
    from models.ei_composite import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = EICompositeResult(
        user_id=req.user_id,
        ei_level=result["ei_level"],
        ei_composite_score=result["ei_composite_score"],
        emotional_awareness=result["emotional_awareness"],
        emotional_regulation=result["emotional_regulation"],
        empathy_index=result["empathy_index"],
        emotional_memory=result["emotional_memory"],
        self_regulation=result["self_regulation"],
        frontal_asymmetry_faa=result["frontal_asymmetry_faa"],
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
