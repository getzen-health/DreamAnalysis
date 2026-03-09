"""Affect labeling efficacy tracker via LPP amplitude reduction API (#185)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/affect-labeling", tags=["affect-labeling"])


class AffectLabelingInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class AffectLabelingResult(BaseModel):
    user_id: str
    labeling_efficacy_level: str
    efficacy_score: float
    left_pfc_regulation: float
    high_beta_attenuation: float
    alpha_regulation_index: float
    frontal_asymmetry_faa: float
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=AffectLabelingResult)
async def analyze_affect_labeling(req: AffectLabelingInput):
    """Track affect labeling efficacy via EEG LPP proxy and alpha regulation."""
    from models.affect_labeling import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = AffectLabelingResult(
        user_id=req.user_id,
        labeling_efficacy_level=result["labeling_efficacy_level"],
        efficacy_score=result["efficacy_score"],
        left_pfc_regulation=result["left_pfc_regulation"],
        high_beta_attenuation=result["high_beta_attenuation"],
        alpha_regulation_index=result["alpha_regulation_index"],
        frontal_asymmetry_faa=result["frontal_asymmetry_faa"],
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
