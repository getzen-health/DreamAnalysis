"""Emotional granularity estimator API (#182)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/emotional-granularity", tags=["emotional-granularity"])


class GranularityEpisodeInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    emotion_label: Optional[str] = None
    user_id: str


class GranularityQueryInput(BaseModel):
    user_id: str


_history: dict = defaultdict(lambda: deque(maxlen=200))
_estimators: dict = {}


def _get_estimator(user_id: str):
    if user_id not in _estimators:
        from models.emotional_granularity import EmotionalGranularityEstimator
        _estimators[user_id] = EmotionalGranularityEstimator()
    return _estimators[user_id]


@router.post("/add-episode")
async def add_episode(req: GranularityEpisodeInput):
    """Add an emotional episode (EEG + optional label) for granularity tracking."""
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    estimator = _get_estimator(req.user_id)
    estimator.add_episode(signals[0], req.fs, label=req.emotion_label, user_id=req.user_id)
    return {"user_id": req.user_id, "status": "episode_added",
            "episode_count": len(estimator._episodes.get(req.user_id, []))}


@router.get("/compute/{user_id}")
async def compute_granularity(user_id: str):
    """Compute emotional granularity score from accumulated episodes."""
    estimator = _get_estimator(user_id)
    result = estimator.compute_granularity(user_id)
    result["processed_at"] = time.time()
    _history[user_id].append(result)
    return result


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_granularity(user_id: str):
    if user_id in _estimators:
        _estimators[user_id].reset(user_id)
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
