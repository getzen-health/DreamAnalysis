"""Post-traumatic growth tracking API routes.

POST /ptg/assess     — assess current PTG status (full profile)
POST /ptg/trajectory — track growth trajectory over time
GET  /ptg/status     — get user state summary

Supporting endpoints:
POST /ptg/adversity       — record an adversity event
POST /ptg/emotional-data  — add emotional snapshot for longitudinal tracking
POST /ptg/domain-ratings  — submit self-report domain ratings

GitHub issue: #456
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/ptg", tags=["post-traumatic-growth"])

# ── Lazy singleton ──────────────────────────────────────────────────────────

_tracker = None


def _get_tracker():
    global _tracker
    if _tracker is None:
        try:
            from models.post_traumatic_growth import get_ptg_tracker
            _tracker = get_ptg_tracker()
        except Exception as exc:
            log.warning("PTGTracker unavailable: %s", exc)
    return _tracker


# ── Pydantic schemas ────────────────────────────────────────────────────────

class AdversityRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    date: str = Field(..., description="YYYY-MM-DD")
    severity: float = Field(..., ge=0.0, le=10.0, description="Adversity severity 0-10")
    adversity_type: str = Field(..., description="One of: loss, illness, trauma, relationship, career")
    description: str = Field("", description="Optional free-text description")


class EmotionalDataRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    date: str = Field(..., description="YYYY-MM-DD")
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    stress_level: float = Field(..., ge=0.0, le=1.0)
    social_engagement: float = Field(0.5, ge=0.0, le=1.0)


class DomainRatingsRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    ratings: Dict[str, float] = Field(
        ...,
        description="Domain ratings (0-5 Likert): relating_to_others, new_possibilities, personal_strength, spiritual_change, appreciation_of_life",
    )


class AssessRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    domain_ratings: Optional[Dict[str, float]] = Field(
        None,
        description="Optional fresh domain ratings (0-5 Likert) to include in assessment",
    )


class TrajectoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/adversity")
def record_adversity(req: AdversityRequest) -> Dict[str, Any]:
    """Record an adversity event for a user.

    Adversity events are reference points for computing growth trajectory
    and distinguishing resilience from growth.
    """
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "PTGTracker unavailable")

    try:
        event = tracker.record_adversity(
            user_id=req.user_id,
            date=req.date,
            severity=req.severity,
            adversity_type=req.adversity_type,
            description=req.description,
        )
    except Exception as exc:
        log.exception("record_adversity failed: %s", exc)
        raise HTTPException(500, f"Error: {exc}")

    return {
        "recorded": True,
        "event": {
            "date": event.date,
            "severity": event.severity,
            "type": event.adversity_type,
            "description": event.description,
        },
    }


@router.post("/emotional-data")
def add_emotional_data(req: EmotionalDataRequest) -> Dict[str, Any]:
    """Add an emotional snapshot for longitudinal PTG tracking.

    Feed daily emotional readings to build the longitudinal dataset
    used for detecting growth indicators.
    """
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "PTGTracker unavailable")

    try:
        snapshot = tracker.add_emotional_snapshot(
            user_id=req.user_id,
            date=req.date,
            valence=req.valence,
            arousal=req.arousal,
            stress_level=req.stress_level,
            social_engagement=req.social_engagement,
        )
    except Exception as exc:
        log.exception("add_emotional_snapshot failed: %s", exc)
        raise HTTPException(500, f"Error: {exc}")

    return {
        "added": True,
        "snapshot": {
            "date": snapshot.date,
            "valence": snapshot.valence,
            "arousal": snapshot.arousal,
            "stress_level": snapshot.stress_level,
            "social_engagement": snapshot.social_engagement,
        },
    }


@router.post("/domain-ratings")
def submit_domain_ratings(req: DomainRatingsRequest) -> Dict[str, Any]:
    """Submit self-report PTG domain ratings (0-5 Likert scale).

    Domains: relating_to_others, new_possibilities, personal_strength,
    spiritual_change, appreciation_of_life.
    """
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "PTGTracker unavailable")

    try:
        normalized = tracker.set_domain_ratings(
            user_id=req.user_id, ratings=req.ratings
        )
    except Exception as exc:
        log.exception("set_domain_ratings failed: %s", exc)
        raise HTTPException(500, f"Error: {exc}")

    return {
        "user_id": req.user_id,
        "normalized_ratings": normalized,
    }


@router.post("/assess")
def assess_ptg(req: AssessRequest) -> Dict[str, Any]:
    """Assess current PTG status -- full growth profile.

    Combines self-report domain ratings, longitudinal emotional data,
    adversity events, and growth indicators into a comprehensive profile.
    Also stores the assessment for trajectory tracking.
    """
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "PTGTracker unavailable")

    try:
        profile = tracker.compute_growth_profile(req.user_id)
    except Exception as exc:
        log.exception("compute_growth_profile failed: %s", exc)
        raise HTTPException(500, f"Assessment error: {exc}")

    return profile


@router.post("/trajectory")
def track_trajectory(req: TrajectoryRequest) -> Dict[str, Any]:
    """Track PTG growth trajectory over time.

    Requires at least 2 prior assessments (via POST /ptg/assess).
    Computes Theil-Sen slope of PTG total scores.
    """
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "PTGTracker unavailable")

    try:
        result = tracker.track_growth_trajectory(req.user_id)
    except Exception as exc:
        log.exception("track_growth_trajectory failed: %s", exc)
        raise HTTPException(500, f"Trajectory error: {exc}")

    return {"user_id": req.user_id, **result}


@router.get("/status")
def get_status(
    user_id: str = Query("default", min_length=1, description="User identifier"),
) -> Dict[str, Any]:
    """Get PTG tracking status for a user.

    Returns adversity history, emotional data count, domain ratings,
    assessment count, and domain definitions.
    """
    tracker = _get_tracker()
    if tracker is None:
        raise HTTPException(503, "PTGTracker unavailable")

    try:
        result = tracker.profile_to_dict(user_id)
    except Exception as exc:
        log.exception("profile_to_dict failed: %s", exc)
        raise HTTPException(500, f"Status error: {exc}")

    return result
