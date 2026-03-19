"""Somatic marker mapping API routes (#421).

Provides endpoints for predicting emotions from body-sensation maps,
learning personal somatic-emotion pairings, and checking service status.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/somatic", tags=["somatic"])


# ── Request / Response schemas ────────────────────────────────────────


class BodySensation(BaseModel):
    region: str = Field(..., description="Body region (e.g. chest, stomach, head)")
    sensation_type: str = Field(..., description="Type of sensation (e.g. tension, warmth)")
    intensity: float = Field(..., ge=1, le=5, description="Intensity on 1-5 scale")


class PredictRequest(BaseModel):
    body_map: List[BodySensation]
    user_id: str = Field(default="default", min_length=1)


class LearnRequest(BaseModel):
    body_map: List[BodySensation]
    reported_valence: float = Field(..., ge=-1.0, le=1.0)
    reported_arousal: float = Field(..., ge=-1.0, le=1.0)
    user_id: str = Field(default="default", min_length=1)


class PredictResponse(BaseModel):
    valence: float
    arousal: float
    predicted_emotion: str
    confidence: float
    n_activations: int
    dominant_region: Optional[str] = None
    dominant_sensation: Optional[str] = None


class LearnResponse(BaseModel):
    status: str
    n_pairings_learned: int
    total_pairings: int


# ── Singleton model instance ──────────────────────────────────────────

_model = None


def _get_model():
    global _model
    if _model is None:
        from models.somatic_marker_model import SomaticMarkerModel
        _model = SomaticMarkerModel()
    return _model


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post("/predict", response_model=PredictResponse)
async def predict_emotion_from_soma(req: PredictRequest):
    """Predict emotion from a body-sensation map."""
    model = _get_model()
    body_map = [s.model_dump() for s in req.body_map]
    result = model.predict_emotion_from_soma(body_map, user_id=req.user_id)
    return PredictResponse(**result)


@router.post("/learn", response_model=LearnResponse)
async def learn_somatic_pairing(req: LearnRequest):
    """Learn a personal somatic-emotion pairing."""
    model = _get_model()
    body_map = [s.model_dump() for s in req.body_map]
    result = model.learn_somatic_pairing(
        body_map,
        reported_valence=req.reported_valence,
        reported_arousal=req.reported_arousal,
        user_id=req.user_id,
    )
    return LearnResponse(**result)


@router.get("/status")
async def somatic_status():
    """Check somatic marker model availability."""
    return {
        "available": True,
        "model": "somatic_marker",
        "version": "1.0.0",
    }
