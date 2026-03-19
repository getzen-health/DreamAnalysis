"""Interoception training engine API routes (#422).

Provides endpoints for heartbeat counting scoring, body scan scoring,
interoceptive profile computation, and exercise generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/interoception", tags=["interoception"])


# ── Request / Response schemas ────────────────────────────────────────


class HeartbeatScoreRequest(BaseModel):
    counted_beats: int = Field(..., ge=0, description="User's reported heartbeat count")
    actual_beats: int = Field(..., ge=0, description="Sensor-measured heartbeat count")
    duration_seconds: float = Field(..., gt=0, description="Counting interval in seconds")
    condition: str = Field(default="rest", description="Task condition: rest, movement, post_exercise, paced_breathing")
    user_id: str = Field(default="default", min_length=1)


class BodyScanReport(BaseModel):
    region: str = Field(..., description="Body region name")
    reported_sensation: str = Field(..., description="User-reported sensation or 'none'")
    ground_truth_active: bool = Field(..., description="Whether physiological signal was present")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="User confidence 0-1")


class BodyScanRequest(BaseModel):
    reports: List[BodyScanReport]
    user_id: str = Field(default="default", min_length=1)


class ProfileRequest(BaseModel):
    user_id: str = Field(default="default", min_length=1)


# ── Singleton model instance ──────────────────────────────────────────

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from models.interoception_model import InteroceptionEngine
        _engine = InteroceptionEngine()
    return _engine


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post("/score-heartbeat")
async def score_heartbeat(req: HeartbeatScoreRequest):
    """Score a heartbeat counting task using the Schandry IAS formula."""
    engine = _get_engine()
    result = engine.score_heartbeat_counting(
        counted_beats=req.counted_beats,
        actual_beats=req.actual_beats,
        duration_seconds=req.duration_seconds,
        condition=req.condition,
        user_id=req.user_id,
    )
    return result


@router.post("/score-body-scan")
async def score_body_scan(req: BodyScanRequest):
    """Score a body scan exercise against physiological ground truth."""
    engine = _get_engine()
    reports = [r.model_dump() for r in req.reports]
    result = engine.score_body_scan(reports=reports, user_id=req.user_id)
    return result


@router.post("/profile")
async def compute_profile(req: ProfileRequest):
    """Compute full interoceptive profile with MAIA-like dimensions."""
    engine = _get_engine()
    profile = engine.compute_interoceptive_profile(user_id=req.user_id)
    return engine.profile_to_dict(profile)


@router.get("/next-exercise")
async def get_next_exercise(user_id: str = "default"):
    """Generate the next progressive interoception exercise."""
    engine = _get_engine()
    return engine.generate_next_exercise(user_id=user_id)


@router.get("/status")
async def interoception_status():
    """Check interoception training engine availability."""
    return {
        "available": True,
        "model": "interoception_engine",
        "version": "1.0.0",
    }
