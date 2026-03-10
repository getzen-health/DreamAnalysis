"""Long COVID / chronic fatigue EEG screening API (#175)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/long-covid-screen", tags=["long-covid-screener"])


class LongCOVIDInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = Field(..., min_length=1)


class LongCOVIDResult(BaseModel):
    user_id: str
    risk_category: str
    long_covid_risk_score: float
    slowing_ratio: float
    beta_deficit: float
    peak_alpha_freq_hz: float
    delta_burden: float
    note: str
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=LongCOVIDResult)
async def analyze_long_covid(req: LongCOVIDInput):
    """Screen for Long COVID / CFS neurological markers from EEG."""
    from models.long_covid_screener import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = LongCOVIDResult(
        user_id=req.user_id,
        risk_category=result["risk_category"],
        long_covid_risk_score=result["long_covid_risk_score"],
        slowing_ratio=result["slowing_ratio"],
        beta_deficit=result["beta_deficit"],
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
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
