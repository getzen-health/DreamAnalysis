"""Interoceptive awareness training via EEG-heartbeat coupling API (#184)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/interoceptive-awareness", tags=["interoceptive-awareness"])


class InteroceptiveInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    set_as_baseline: bool = False
    user_id: str = "default"


class InteroceptiveResult(BaseModel):
    user_id: str
    body_awareness_level: str
    interoceptive_score: float
    frontal_theta_power: float
    alpha_suppression: float
    right_frontal_activation: float
    has_baseline: bool
    processed_at: float


_trainers: dict = {}
_history: dict = defaultdict(lambda: deque(maxlen=200))


def _get_trainer(user_id: str):
    if user_id not in _trainers:
        from models.interoceptive_awareness import InteroceptiveAwarenessTrainer
        _trainers[user_id] = InteroceptiveAwarenessTrainer()
    return _trainers[user_id]


@router.post("/analyze", response_model=InteroceptiveResult)
async def analyze_interoception(req: InteroceptiveInput):
    """Assess interoceptive awareness via EEG body-mind coupling."""
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    trainer = _get_trainer(req.user_id)

    if req.set_as_baseline:
        trainer.set_baseline(signals, req.fs, req.user_id)
        return {"status": "baseline_set", "user_id": req.user_id}

    result = trainer.assess(signals, req.fs, req.user_id)

    out = InteroceptiveResult(
        user_id=req.user_id,
        body_awareness_level=result.get("body_awareness_level", "low"),
        interoceptive_score=result.get("interoceptive_score", 0.0),
        frontal_theta_power=result.get("frontal_theta_power", 0.0),
        alpha_suppression=result.get("alpha_suppression", 0.0),
        right_frontal_activation=result.get("right_frontal_activation", 0.0),
        has_baseline=result.get("has_baseline", False),
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
    if user_id in _trainers:
        del _trainers[user_id]
    return {"user_id": user_id, "status": "reset"}
