"""MCI/Alzheimer early screening API (#162)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/mci-screen", tags=["mci-screener"])


class MCIScreenInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class MCIScreenResult(BaseModel):
    user_id: str
    risk_category: str
    mci_risk_score: float
    slowing_ratio: float
    peak_alpha_freq_hz: float
    delta_burden: float
    note: str
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=MCIScreenResult)
async def analyze_mci(req: MCIScreenInput):
    """Screen for MCI/Alzheimer's risk using EEG slowing markers."""
    from models.mci_screener import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = MCIScreenResult(
        user_id=req.user_id,
        risk_category=result["risk_category"],
        mci_risk_score=result["mci_risk_score"],
        slowing_ratio=result["slowing_ratio"],
        peak_alpha_freq_hz=result["peak_alpha_freq_hz"],
        delta_burden=result["delta_burden"],
        note=result["note"],
        model_used=result["model_used"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent MCI screening history for a user."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear MCI screening history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
