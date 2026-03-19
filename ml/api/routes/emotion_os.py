"""Emotion OS API routes -- open platform for emotion-as-a-service.

Endpoints:
  POST /emotion-os/fuse            -- fuse multiple emotion sources
  POST /emotion-os/register-app    -- register an app on the platform
  POST /emotion-os/webhook         -- register a webhook
  GET  /emotion-os/stats           -- platform statistics
  GET  /emotion-os/status          -- health check

Issue #442.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.emotion_os import (
    BASIC_EMOTIONS,
    VALID_SOURCES,
    check_webhook_triggers,
    compute_platform_stats,
    create_emotion_vector,
    fuse_emotion_sources,
    platform_to_dict,
    register_app,
    register_webhook,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/emotion-os", tags=["emotion-os"])


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------


class EmotionSourceInput(BaseModel):
    """A single emotion source for the fusion endpoint."""

    source: str = Field(
        ...,
        description="Source type: eeg, voice, text, physiological, self_report",
    )
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)
    dominance: float = Field(0.5, ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Per-emotion probabilities (happy, sad, angry, fear, surprise, neutral)",
    )
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class FuseRequest(BaseModel):
    sources: List[EmotionSourceInput] = Field(..., min_length=1)
    weights: Optional[Dict[str, float]] = Field(
        None,
        description="Optional per-source weight overrides",
    )


class RegisterAppRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    rate_limit: int = Field(60, ge=1, le=10000)


class WebhookRequest(BaseModel):
    app_id: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    emotion: str = Field(
        ...,
        description="Emotion key (happy, sad, angry, fear, surprise, neutral, valence, arousal, dominance)",
    )
    threshold: float = Field(..., description="Threshold value")
    direction: str = Field("above", description="'above' or 'below'")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@router.post("/fuse")
async def fuse_emotions(req: FuseRequest) -> Dict[str, Any]:
    """Fuse multiple emotion source vectors into a single unified EmotionVector."""
    vectors = []
    for src in req.sources:
        vec = create_emotion_vector(
            valence=src.valence,
            arousal=src.arousal,
            dominance=src.dominance,
            probabilities=src.probabilities,
            confidence=src.confidence,
            source=src.source,
        )
        vectors.append(vec)

    try:
        fused = fuse_emotion_sources(vectors, weights=req.weights)
    except Exception as exc:
        log.exception("fuse_emotion_sources failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "fused": {
            "valence": round(fused.valence, 4),
            "arousal": round(fused.arousal, 4),
            "dominance": round(fused.dominance, 4),
            "probabilities": {e: round(p, 4) for e, p in fused.probabilities.items()},
            "confidence": round(fused.confidence, 4),
            "dominant_emotion": fused.dominant_emotion(),
            "source": fused.source,
        },
        "source_count": len(req.sources),
    }


@router.post("/register-app")
async def register_application(req: RegisterAppRequest) -> Dict[str, Any]:
    """Register a new application on the Emotion OS platform."""
    try:
        app = register_app(name=req.name, rate_limit=req.rate_limit)
    except Exception as exc:
        log.exception("register_app failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "app_id": app.app_id,
        "name": app.name,
        "api_key": app.api_key,
        "rate_limit": app.rate_limit,
    }


@router.post("/webhook")
async def register_webhook_endpoint(req: WebhookRequest) -> Dict[str, Any]:
    """Register a webhook for emotion threshold notifications."""
    result = register_webhook(
        app_id=req.app_id,
        url=req.url,
        emotion=req.emotion,
        threshold=req.threshold,
        direction=req.direction,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {"status": "ok", **result}


@router.get("/stats")
async def get_platform_stats() -> Dict[str, Any]:
    """Return aggregate platform statistics."""
    return {
        "status": "ok",
        "stats": compute_platform_stats(),
    }


@router.get("/status")
async def emotion_os_status() -> Dict[str, Any]:
    """Return platform availability and summary."""
    stats = compute_platform_stats()
    return {
        "status": "available",
        "total_apps": stats["total_apps"],
        "total_webhooks": stats["total_webhooks"],
        "total_plugins": stats["total_plugins"],
        "total_fusions": stats["total_fusions"],
    }
