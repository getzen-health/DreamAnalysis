"""Autism spectrum EEG screening API (#164)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/autism-screen", tags=["autism-screener"])


class AutismScreenInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = Field(..., min_length=1)


class AutismScreenResult(BaseModel):
    user_id: str
    risk_category: str
    asd_atypicality_score: float
    mu_suppression_index: float
    inter_hemispheric_coherence: float
    eeg_complexity: float
    note: str
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=AutismScreenResult)
async def analyze_autism(req: AutismScreenInput):
    """Screen for ASD atypicality using EEG connectivity and mu rhythm."""
    from models.autism_screener import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = AutismScreenResult(
        user_id=req.user_id,
        risk_category=result["risk_category"],
        asd_atypicality_score=result["asd_atypicality_score"],
        mu_suppression_index=result["mu_suppression_index"],
        inter_hemispheric_coherence=result["inter_hemispheric_coherence"],
        eeg_complexity=result["eeg_complexity"],
        note=result["note"],
        model_used=result["model_used"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent autism screening history for a user."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear autism screening history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
