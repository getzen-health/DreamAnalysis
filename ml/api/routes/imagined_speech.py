"""Imagined speech BCI command decoder API (#166)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/imagined-speech", tags=["imagined-speech"])


class ImaginedSpeechInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class ImaginedSpeechResult(BaseModel):
    user_id: str
    predicted_command: str
    confidence: float
    probabilities: Dict[str, float]
    lateral_asymmetry: float
    is_reliable: bool
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=ImaginedSpeechResult)
async def analyze_imagined_speech(req: ImaginedSpeechInput):
    """Decode imagined speech command from EEG spectral asymmetry."""
    from models.imagined_speech import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs)

    out = ImaginedSpeechResult(
        user_id=req.user_id,
        predicted_command=result["predicted_command"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        lateral_asymmetry=result["lateral_asymmetry"],
        is_reliable=result["is_reliable"],
        model_used=result["model_used"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent imagined speech history for a user."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear imagined speech history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
