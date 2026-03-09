"""EEG decision confidence and risk-taking predictor API (#165)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/decision-confidence", tags=["decision-confidence"])


class DecisionConfidenceInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class DecisionConfidenceResult(BaseModel):
    user_id: str
    confidence_label: str
    confidence_score: float
    conflict_index: float
    risk_propensity: float
    decision_readiness: float
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=DecisionConfidenceResult)
async def analyze_decision_confidence(req: DecisionConfidenceInput):
    """Predict decision confidence and conflict from pre-decision EEG."""
    from models.decision_confidence import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = DecisionConfidenceResult(
        user_id=req.user_id,
        confidence_label=result["confidence_label"],
        confidence_score=result["confidence_score"],
        conflict_index=result["conflict_index"],
        risk_propensity=result["risk_propensity"],
        decision_readiness=result["decision_readiness"],
        model_used=result["model_used"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent decision confidence history for a user."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear decision confidence history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
