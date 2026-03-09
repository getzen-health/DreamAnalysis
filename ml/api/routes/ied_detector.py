"""IED spike detection for epilepsy screening API (#183)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/ied-detector", tags=["ied-detector"])


class IEDInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class IEDResult(BaseModel):
    user_id: str
    risk_category: str
    spike_count: int
    spike_rate_per_min: float
    max_amplitude_uv: float
    signal_kurtosis: float
    spikes: List[Dict]
    note: str
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=IEDResult)
async def analyze_ied(req: IEDInput):
    """Screen for interictal epileptiform discharges (IED) in EEG."""
    from models.ied_detector import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = IEDResult(
        user_id=req.user_id,
        risk_category=result["risk_category"],
        spike_count=result["spike_count"],
        spike_rate_per_min=result["spike_rate_per_min"],
        max_amplitude_uv=result["max_amplitude_uv"],
        signal_kurtosis=result["signal_kurtosis"],
        spikes=result["spikes"],
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
