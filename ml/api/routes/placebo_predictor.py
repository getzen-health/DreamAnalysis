"""Placebo response predictor API (#178)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/placebo-predictor", tags=["placebo-predictor"])


class PlaceboInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class PlaceboResult(BaseModel):
    user_id: str
    placebo_response_prediction: str
    responder_score: float
    alpha_dominance: float
    theta_suggestibility: float
    approach_motivation_faa: float
    expected_accuracy: str
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/predict", response_model=PlaceboResult)
async def predict_placebo_response(req: PlaceboInput):
    """Predict placebo response likelihood from resting EEG."""
    from models.placebo_predictor import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = PlaceboResult(
        user_id=req.user_id,
        placebo_response_prediction=result["placebo_response_prediction"],
        responder_score=result["responder_score"],
        alpha_dominance=result["alpha_dominance"],
        theta_suggestibility=result["theta_suggestibility"],
        approach_motivation_faa=result["approach_motivation_faa"],
        expected_accuracy=result["expected_accuracy"],
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
