"""Emotional synchrony detector API (#188)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/emotional-synchrony", tags=["emotional-synchrony"])


class SynchronyInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    set_as_baseline: bool = False
    user_id: str = Field(..., min_length=1)


class SynchronyResult(BaseModel):
    user_id: str
    synchrony_score: float
    fronto_temporal_plv_alpha: float
    fronto_temporal_plv_beta: float
    frontal_interhemispheric_plv: float
    alpha_coherence: float
    beta_coherence: float
    engagement_level: str
    has_baseline: bool
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))
_detectors: dict = {}


def _get_detector(user_id: str):
    if user_id not in _detectors:
        from models.emotional_synchrony import EmotionalSynchronyDetector
        _detectors[user_id] = EmotionalSynchronyDetector()
    return _detectors[user_id]


@router.post("/analyze", response_model=SynchronyResult)
async def analyze_synchrony(req: SynchronyInput):
    """Detect emotional synchrony via EEG phase-locking and coherence."""
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    detector = _get_detector(req.user_id)

    if req.set_as_baseline:
        detector.set_baseline(signals, req.fs, req.user_id)
        return {"user_id": req.user_id, "status": "baseline_set"}

    result = detector.analyze(signals, req.fs, req.user_id)

    out = SynchronyResult(
        user_id=req.user_id,
        synchrony_score=result.get("synchrony_score", 0.0),
        fronto_temporal_plv_alpha=result.get("fronto_temporal_plv_alpha", 0.0),
        fronto_temporal_plv_beta=result.get("fronto_temporal_plv_beta", 0.0),
        frontal_interhemispheric_plv=result.get("frontal_interhemispheric_plv", 0.0),
        alpha_coherence=result.get("alpha_coherence", 0.0),
        beta_coherence=result.get("beta_coherence", 0.0),
        engagement_level=result.get("engagement_level", "disengaged"),
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
    if user_id in _detectors:
        del _detectors[user_id]
    return {"user_id": user_id, "status": "reset"}
