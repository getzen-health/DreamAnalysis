"""Big Five personality trait estimator API (#176)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/big-five", tags=["big-five-personality"])


class BigFiveInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class BigFiveResult(BaseModel):
    user_id: str
    neuroticism: float
    extraversion: float
    openness: float
    conscientiousness: float
    agreeableness: float
    frontal_alpha_asymmetry: float
    confidence: str
    note: str
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/estimate", response_model=BigFiveResult)
async def estimate_big_five(req: BigFiveInput):
    """Estimate Big Five personality traits from resting EEG."""
    from models.big_five_estimator import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = BigFiveResult(
        user_id=req.user_id,
        neuroticism=result["neuroticism"],
        extraversion=result["extraversion"],
        openness=result["openness"],
        conscientiousness=result["conscientiousness"],
        agreeableness=result["agreeableness"],
        frontal_alpha_asymmetry=result["frontal_alpha_asymmetry"],
        confidence=result["confidence"],
        note=result["note"],
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
