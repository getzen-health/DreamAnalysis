"""EEG visual attention and gaze zone estimation API (#160)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/visual-attention", tags=["visual-attention"])


class VisualAttentionInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class VisualAttentionResult(BaseModel):
    user_id: str
    attention_zone: str
    horizontal_bias: float
    vertical_bias: float
    alpha_suppression: float
    visual_engagement: float
    sustained_attention_index: float
    grid_col: int
    grid_row: int
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=VisualAttentionResult)
async def analyze_visual_attention(req: VisualAttentionInput):
    """Estimate visual attention zone from EEG alpha lateralization."""
    from models.visual_attention import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    r = get_model().predict(signals, req.fs)
    out = VisualAttentionResult(
        user_id=req.user_id,
        attention_zone=r["attention_zone"],
        horizontal_bias=r["horizontal_bias"],
        vertical_bias=r["vertical_bias"],
        alpha_suppression=r["alpha_suppression"],
        visual_engagement=r["visual_engagement"],
        sustained_attention_index=r["sustained_attention_index"],
        grid_col=r["grid_col"],
        grid_row=r["grid_row"],
        model_used=r["model_used"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent visual attention history for a user."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear visual attention history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
