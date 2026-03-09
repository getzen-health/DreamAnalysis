"""Brain age estimation API (#174)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/brain-age", tags=["brain-age"])


class BrainAgeInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    chronological_age: Optional[float] = None
    user_id: str = "default"


class BrainAgeResult(BaseModel):
    user_id: str
    predicted_brain_age: Optional[float]
    brain_age_gap: Optional[float]
    aperiodic_exponent: float
    alpha_power: float
    beta_power: float
    delta_power: float
    disclaimer: str
    model_type: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/estimate", response_model=BrainAgeResult)
async def estimate_brain_age(req: BrainAgeInput):
    """Estimate biological brain age from EEG aperiodic features."""
    from models.brain_age_estimator import get_brain_age_estimator
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    estimator = get_brain_age_estimator()
    result = estimator.predict(
        signals[0], req.fs,
        chronological_age=req.chronological_age
    )

    out = BrainAgeResult(
        user_id=req.user_id,
        predicted_brain_age=result.get("predicted_age"),
        brain_age_gap=result.get("brain_age_gap"),
        aperiodic_exponent=result.get("aperiodic_exponent", 0.0),
        alpha_power=result.get("alpha_power", 0.0),
        beta_power=result.get("beta_power", 0.0),
        delta_power=result.get("delta_power", 0.0),
        disclaimer=result.get("disclaimer", "Wellness indicator only"),
        model_type=result.get("model_type", "aperiodic_heuristic"),
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
