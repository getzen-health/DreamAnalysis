"""Affect labeling efficacy tracker via LPP amplitude reduction API (#185)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/affect-labeling", tags=["affect-labeling"])

_trackers: dict = {}
_history: dict = defaultdict(lambda: deque(maxlen=200))


def _get_tracker(user_id: str):
    if user_id not in _trackers:
        from models.affect_labeling import AffectLabelingTracker
        _trackers[user_id] = AffectLabelingTracker()
    return _trackers[user_id]


class AffectSignalInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class AffectPostLabelInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    label: Optional[str] = None
    user_id: str = "default"


@router.post("/baseline")
async def set_baseline(req: AffectSignalInput):
    """Record resting-state baseline for affect labeling tracker."""
    tracker = _get_tracker(req.user_id)
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    result = tracker.set_baseline(signals, fs=req.fs, user_id=req.user_id)
    return {"user_id": req.user_id, **result}


@router.post("/pre-label")
async def record_pre_label(req: AffectSignalInput):
    """Record EEG epoch immediately before the user labels their emotion."""
    tracker = _get_tracker(req.user_id)
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    result = tracker.record_pre_label(signals, fs=req.fs, user_id=req.user_id)
    return {"user_id": req.user_id, **result}


@router.post("/post-label")
async def record_post_label(req: AffectPostLabelInput):
    """Record EEG epoch after labeling and return labeling efficacy."""
    tracker = _get_tracker(req.user_id)
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    result = tracker.record_post_label(
        signals, label=req.label, fs=req.fs, user_id=req.user_id
    )
    out = {"user_id": req.user_id, "processed_at": time.time(), **result}
    _history[req.user_id].append(out)
    return out


@router.get("/session-stats/{user_id}")
async def get_session_stats(user_id: str):
    stats = _get_tracker(user_id).get_session_stats(user_id)
    return {"user_id": user_id, **stats}


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    _history[user_id].clear()
    if user_id in _trackers:
        _trackers[user_id].reset(user_id)
    return {"user_id": user_id, "status": "reset"}
