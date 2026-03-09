"""Addiction craving detector with alpha-theta NFB guidance API (#173)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/craving-detector", tags=["craving-detector"])


class CravingInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class CravingResult(BaseModel):
    user_id: str
    craving_level: str
    craving_score: float
    beta_alpha_ratio: float
    alpha_theta_ratio: float
    frontal_asymmetry_faa: float
    nfb_protocol: str
    nfb_target: str
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=CravingResult)
async def analyze_craving(req: CravingInput):
    """Detect craving state and recommend alpha-theta NFB protocol."""
    from models.craving_detector import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = CravingResult(
        user_id=req.user_id,
        craving_level=result["craving_level"],
        craving_score=result["craving_score"],
        beta_alpha_ratio=result["beta_alpha_ratio"],
        alpha_theta_ratio=result["alpha_theta_ratio"],
        frontal_asymmetry_faa=result["frontal_asymmetry_faa"],
        nfb_protocol=result["nfb_protocol"],
        nfb_target=result["nfb_target"],
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
