"""Longitudinal EI growth tracker API routes.

POST /ei/growth/snapshot              — add a daily EI snapshot
GET  /ei/growth/trend/{user_id}       — weekly trend + slope
GET  /ei/growth/milestones/{user_id}  — achieved milestones
GET  /ei/growth/trajectory/{user_id}  — trajectory to target EIQ
GET  /ei/growth/dimensions/{user_id}  — per-dimension report
GET  /ei/growth/history/{user_id}     — past snapshots (last_n param)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/ei/growth", tags=["ei-growth"])

# ── Lazy singleton ─────────────────────────────────────────────────────────────

_tracker = None


def _get_tracker():
    global _tracker
    if _tracker is None:
        try:
            from models.ei_growth_tracker import get_ei_growth_tracker  # type: ignore
            _tracker = get_ei_growth_tracker()
        except Exception as exc:
            log.warning("EIGrowthTracker unavailable: %s", exc)
    return _tracker


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class SnapshotRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="User identifier")
    eiq_score: float = Field(..., ge=0.0, le=100.0, description="Overall EI composite score (0-100)")
    dimension_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension scores (0-100): self_perception, self_expression, interpersonal, decision_making, stress_management",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources that contributed, e.g. ['voice', 'eeg', 'health']",
    )
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence of the reading (0-1)")


class SnapshotResponse(BaseModel):
    added: bool
    snapshot: Dict[str, Any]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/snapshot", response_model=SnapshotResponse)
def add_snapshot(req: SnapshotRequest) -> Dict[str, Any]:
    """Add a daily EI composite snapshot for a user.

    One snapshot per day is stored; submitting multiple times on the same day
    replaces the earlier entry.
    """
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "EIGrowthTracker unavailable — check server logs")

    try:
        snap = tracker.add_snapshot(
            user_id=req.user_id,
            eiq_score=req.eiq_score,
            dimension_scores=req.dimension_scores,
            data_sources=req.data_sources,
            confidence=req.confidence,
        )
    except Exception as exc:
        log.exception("EIGrowthTracker.add_snapshot failed: %s", exc)
        raise HTTPException(500, f"Snapshot error: {exc}")

    return {
        "added": True,
        "snapshot": {
            "date": snap.date,
            "eiq_score": snap.eiq_score,
            "dimension_scores": snap.dimension_scores,
            "data_sources": snap.data_sources,
            "confidence": snap.confidence,
            "timestamp": snap.timestamp,
        },
    }


@router.get("/trend/{user_id}")
def get_trend(user_id: str) -> Dict[str, Any]:
    """Return weekly EI trend + Theil-Sen slope for the user.

    Computes weekly averages over the last 4 weeks and calculates the
    median-of-pairwise-slopes (Theil-Sen) for robustness with small samples.
    """
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "EIGrowthTracker unavailable — check server logs")

    try:
        result = tracker.get_weekly_trend(user_id=user_id)
    except Exception as exc:
        log.exception("EIGrowthTracker.get_weekly_trend failed: %s", exc)
        raise HTTPException(500, f"Trend error: {exc}")

    return {"user_id": user_id, **result}


@router.get("/milestones/{user_id}")
def get_milestones(user_id: str) -> Dict[str, Any]:
    """Return achieved EI growth milestones for the user."""
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "EIGrowthTracker unavailable — check server logs")

    try:
        milestones = tracker.get_milestones(user_id=user_id)
    except Exception as exc:
        log.exception("EIGrowthTracker.get_milestones failed: %s", exc)
        raise HTTPException(500, f"Milestones error: {exc}")

    return {
        "user_id": user_id,
        "count": len(milestones),
        "milestones": milestones,
    }


@router.get("/trajectory/{user_id}")
def get_trajectory(
    user_id: str,
    target: float = Query(80.0, ge=0.0, le=100.0, description="Target EIQ score to project towards"),
) -> Dict[str, Any]:
    """Predict weeks to reach a target EIQ score given the current growth slope."""
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "EIGrowthTracker unavailable — check server logs")

    try:
        result = tracker.predict_trajectory(user_id=user_id, target_eiq=target)
    except Exception as exc:
        log.exception("EIGrowthTracker.predict_trajectory failed: %s", exc)
        raise HTTPException(500, f"Trajectory error: {exc}")

    return {"user_id": user_id, **result}


@router.get("/dimensions/{user_id}")
def get_dimensions(user_id: str) -> Dict[str, Any]:
    """Return per-dimension 4-week slope and changeability rating."""
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "EIGrowthTracker unavailable — check server logs")

    try:
        result = tracker.get_dimension_report(user_id=user_id)
    except Exception as exc:
        log.exception("EIGrowthTracker.get_dimension_report failed: %s", exc)
        raise HTTPException(500, f"Dimension report error: {exc}")

    return {"user_id": user_id, **result}


@router.get("/history/{user_id}")
def get_history(
    user_id: str,
    last_n: int = Query(30, ge=1, le=365, description="Number of most recent snapshots to return"),
) -> Dict[str, Any]:
    """Return the last N daily EI snapshots for the user."""
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "EIGrowthTracker unavailable — check server logs")

    try:
        snapshots = tracker.get_history(user_id=user_id, last_n=last_n)
    except Exception as exc:
        log.exception("EIGrowthTracker.get_history failed: %s", exc)
        raise HTTPException(500, f"History error: {exc}")

    return {
        "user_id": user_id,
        "count": len(snapshots),
        "snapshots": [
            {
                "date": s.date,
                "eiq_score": s.eiq_score,
                "dimension_scores": s.dimension_scores,
                "data_sources": s.data_sources,
                "confidence": s.confidence,
                "timestamp": s.timestamp,
            }
            for s in snapshots
        ],
    }
