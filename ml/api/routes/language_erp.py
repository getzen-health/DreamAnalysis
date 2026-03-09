"""N400/P600 Language ERP monitoring API (#159)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/language-erp", tags=["language-erp"])


class LanguageERPInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    word_onset_ms: float = 0.0
    user_id: str = "default"
    context: Optional[str] = None  # "expected" | "unexpected" | None


class LanguageERPResult(BaseModel):
    user_id: str
    n400_amplitude_uv: float
    p600_amplitude_uv: float
    semantic_surprise_index: float
    syntactic_load_index: float
    comprehension_score: float
    is_semantically_surprising: bool
    is_syntactically_violated: bool
    model_used: str
    processed_at: float


_history: dict = defaultdict(lambda: deque(maxlen=200))


@router.post("/analyze", response_model=LanguageERPResult)
async def analyze_language_erp(req: LanguageERPInput):
    """Extract N400 and P600 ERP components from word-locked EEG."""
    from models.language_erp import get_model
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    result = get_model().predict(signals, req.fs, req.word_onset_ms)

    out = LanguageERPResult(
        user_id=req.user_id,
        n400_amplitude_uv=result["n400_amplitude_uv"],
        p600_amplitude_uv=result["p600_amplitude_uv"],
        semantic_surprise_index=result["semantic_surprise_index"],
        syntactic_load_index=result["syntactic_load_index"],
        comprehension_score=result["comprehension_score"],
        is_semantically_surprising=result["semantic_surprise_index"] > 0.6,
        is_syntactically_violated=result["syntactic_load_index"] > 0.6,
        model_used=result["model_used"],
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent language ERP analysis history for a user."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear language ERP history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}
