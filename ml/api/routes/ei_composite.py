"""Emotional Intelligence composite score API (#191)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/ei-composite", tags=["ei-composite"])

_ei_model = None
_history: dict = defaultdict(lambda: deque(maxlen=200))


def _get_model():
    global _ei_model
    if _ei_model is None:
        from models.ei_composite import get_model
        _ei_model = get_model()
    return _ei_model


class EICompositeInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    component_scores: Optional[Dict[str, float]] = None
    user_id: str = "default"


class EIBaselineInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class EIComponentInput(BaseModel):
    component_name: str
    score: float
    user_id: str = "default"


@router.post("/analyze")
async def analyze_ei_composite(req: EICompositeInput):
    """Compute Emotional Intelligence Quotient from EEG signals."""
    model = _get_model()
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = model.compute_eiq(
        signals=signals,
        fs=req.fs,
        component_scores=req.component_scores,
        user_id=req.user_id,
    )
    if result is None:
        return {"user_id": req.user_id, "status": "no_data", "eiq_score": None}

    out = {
        "user_id": req.user_id,
        "eiq_score": result["eiq_score"],
        "eiq_grade": result["eiq_grade"],
        "dimensions": result["dimensions"],
        "strengths": result["strengths"],
        "growth_areas": result["growth_areas"],
        "has_baseline": result["has_baseline"],
        "processed_at": time.time(),
    }
    _history[req.user_id].append(out)
    return out


@router.post("/baseline")
async def set_baseline(req: EIBaselineInput):
    """Set resting-state EEG baseline for EI normalisation."""
    model = _get_model()
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    result = model.set_baseline(signals, req.fs, req.user_id)
    return {"user_id": req.user_id, **result}


@router.post("/component")
async def update_component(req: EIComponentInput):
    """Inject an external EI component score (e.g. from granularity module)."""
    model = _get_model()
    model.update_component(req.component_name, req.score, req.user_id)
    return {
        "user_id": req.user_id,
        "component": req.component_name,
        "status": "updated",
    }


@router.get("/session-stats/{user_id}")
async def get_session_stats(user_id: str):
    stats = _get_model().get_session_stats(user_id)
    return {"user_id": user_id, **stats}


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    _history[user_id].clear()
    _get_model().reset(user_id)
    return {"user_id": user_id, "status": "reset"}
