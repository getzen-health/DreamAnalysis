"""Spatial navigation cognitive map detector API (#179)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/spatial-navigation", tags=["spatial-navigation"])


class SpatialNavInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = Field(..., min_length=1)


class SpatialNavResult(BaseModel):
    user_id: str
    navigation_state: str
    navigation_index: float
    fmt_power_fraction: float
    alpha_suppression: float
    theta_alpha_ratio: float
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=SpatialNavResult)
async def analyze_spatial_navigation(req: SpatialNavInput):
    """Detect spatial navigation and cognitive map building from frontal theta."""
    from models.spatial_navigation_detector import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = SpatialNavResult(
        user_id=req.user_id,
        navigation_state=result["navigation_state"],
        navigation_index=result["navigation_index"],
        fmt_power_fraction=result["fmt_power_fraction"],
        alpha_suppression=result["alpha_suppression"],
        theta_alpha_ratio=result["theta_alpha_ratio"],
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
