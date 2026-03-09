"""Few-shot EEG emotion personalization API.

GitHub issue: #115
"""
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ._shared import _numpy_safe
from models.few_shot_personalizer import FewShotPersonalizer

router = APIRouter(tags=["few-shot"])

_personalizer = FewShotPersonalizer()


class SupportInput(BaseModel):
    features: List[float] = Field(..., description="1D feature vector")
    emotion: str = Field(..., description="Emotion label (happy/sad/angry/fear/surprise/neutral)")
    user_id: str = Field("default", description="User identifier")


class ClassifyInput(BaseModel):
    features: List[float] = Field(..., description="1D feature vector to classify")
    user_id: str = Field("default", description="User identifier")


@router.post("/few-shot/add-support")
async def add_support(data: SupportInput):
    """Add a labeled EEG feature vector as a support example for personalization."""
    feats = np.array(data.features, dtype=np.float64)
    if feats.ndim == 0 or len(feats) == 0:
        raise HTTPException(status_code=422, detail="features must be a non-empty list")
    valid_emotions = {"happy", "sad", "angry", "fear", "surprise", "neutral"}
    if data.emotion not in valid_emotions:
        raise HTTPException(
            status_code=422,
            detail=f"emotion must be one of {sorted(valid_emotions)}",
        )
    result = _personalizer.add_support(
        features=feats,
        emotion=data.emotion,
        user_id=data.user_id,
    )
    return _numpy_safe(result)


@router.post("/few-shot/classify")
async def classify_features(data: ClassifyInput):
    """Classify EEG features using per-user prototypical matching."""
    feats = np.array(data.features, dtype=np.float64)
    if feats.ndim == 0 or len(feats) == 0:
        raise HTTPException(status_code=422, detail="features must be a non-empty list")
    result = _personalizer.classify(features=feats, user_id=data.user_id)
    result["user_id"] = data.user_id
    return _numpy_safe(result)


@router.get("/few-shot/status")
async def get_status(user_id: str = "default"):
    """Get personalization adaptation status for a user."""
    result = _personalizer.get_status(user_id=user_id)
    result["user_id"] = user_id
    return _numpy_safe(result)


@router.get("/few-shot/prototypes")
async def get_prototypes(user_id: str = "default"):
    """Get computed emotion prototypes (for visualization/debugging)."""
    protos = _personalizer.get_prototypes(user_id=user_id)
    return {
        "user_id": user_id,
        "prototypes": {e: v for e, v in protos.items()},
        "n_classes": len(protos),
    }


@router.post("/few-shot/reset")
async def reset_personalizer(user_id: str = "default"):
    """Clear all support data and prototypes for a user."""
    _personalizer.reset(user_id=user_id)
    return {"status": "ok", "message": "Few-shot personalizer cleared.", "user_id": user_id}
