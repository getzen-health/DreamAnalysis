"""Emotional memory enhancement predictor API (#189)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/emotional-memory", tags=["emotional-memory"])


class EmotionalMemoryInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class EmotionalMemoryResult(BaseModel):
    user_id: str
    encoding_quality: str
    encoding_score: float
    theta_gamma_coupling: float
    frontal_theta_index: float
    alpha_suppression: float
    arousal_optimality: float
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=EmotionalMemoryResult)
async def analyze_emotional_memory(req: EmotionalMemoryInput):
    """Predict emotional memory encoding quality via theta-gamma coupling."""
    from models.emotional_memory_enhancer import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = EmotionalMemoryResult(
        user_id=req.user_id,
        encoding_quality=result["encoding_quality"],
        encoding_score=result["encoding_score"],
        theta_gamma_coupling=result["theta_gamma_coupling"],
        frontal_theta_index=result["frontal_theta_index"],
        alpha_suppression=result["alpha_suppression"],
        arousal_optimality=result["arousal_optimality"],
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
