"""Parkinson's disease EEG screening API (#168)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/parkinsons-screen", tags=["parkinsons-screener"])


class ParkinsonsScreenInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class ParkinsonsScreenResult(BaseModel):
    user_id: str
    risk_category: str
    pd_risk_score: float
    beta_burden: float
    tremor_oscillation_index: float
    peak_alpha_freq_hz: float
    note: str
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=ParkinsonsScreenResult)
async def analyze_parkinsons(req: ParkinsonsScreenInput):
    """Screen for Parkinson's risk using beta burden and tremor oscillations."""
    from models.parkinsons_screener import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = ParkinsonsScreenResult(
        user_id=req.user_id,
        risk_category=result["risk_category"],
        pd_risk_score=result["pd_risk_score"],
        beta_burden=result["beta_burden"],
        tremor_oscillation_index=result["tremor_oscillation_index"],
        peak_alpha_freq_hz=result["peak_alpha_freq_hz"],
        note=result["note"],
        model_used=result["model_used"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent Parkinson's screening history for a user."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear Parkinson's screening history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
