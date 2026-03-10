"""Motor intention BCI decoder API (#167)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/motor-intention", tags=["motor-intention"])


class MotorIntentionInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = Field(..., min_length=1)


class MotorIntentionResult(BaseModel):
    user_id: str
    intention: str
    control_signal: float
    lateral_mu_asymmetry: float
    lateral_beta_erd: float
    erd_magnitude: float
    probabilities: Dict[str, float]
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=MotorIntentionResult)
async def analyze_motor_intention(req: MotorIntentionInput):
    """Decode motor intention from mu/beta ERD lateralization."""
    from models.motor_intention import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = MotorIntentionResult(
        user_id=req.user_id,
        intention=result["intention"],
        control_signal=result["control_signal"],
        lateral_mu_asymmetry=result["lateral_mu_asymmetry"],
        lateral_beta_erd=result["lateral_beta_erd"],
        erd_magnitude=result["erd_magnitude"],
        probabilities=result["probabilities"],
        model_used=result["model_used"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent motor intention history for a user."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear motor intention history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
