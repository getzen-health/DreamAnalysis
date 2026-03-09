"""Driver drowsiness and alertness detection API (#172)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/drowsiness-alertness", tags=["drowsiness-alertness"])


class DrowsinessInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class DrowsinessResult(BaseModel):
    user_id: str
    state: str
    alertness_score: float
    drowsiness_index: float
    confidence: float
    components: Dict
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=DrowsinessResult)
async def analyze_drowsiness(req: DrowsinessInput):
    """Detect driver drowsiness and alertness state from EEG."""
    from models.drowsiness_detector import DrowsinessDetector
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    detector = DrowsinessDetector()
    result = detector.predict(signals[0], req.fs)

    # Normalize components to simple dict if needed
    components = result.get("components", {})
    if not isinstance(components, dict):
        components = {}

    out = DrowsinessResult(
        user_id=req.user_id,
        state=result.get("state", "unknown"),
        alertness_score=result.get("alertness_score", 0.5),
        drowsiness_index=result.get("drowsiness_index", 0.5),
        confidence=result.get("confidence", 0.5),
        components=components,
        model_used=result.get("model_type", "feature_based"),
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent drowsiness analysis history for a user."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear drowsiness history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
