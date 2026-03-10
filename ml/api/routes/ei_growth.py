"""Longitudinal EI growth tracker API.

POST /ei-growth/log/{user_id}         — log a session
GET  /ei-growth/report/{user_id}      — full growth report
GET  /ei-growth/domain/{user_id}/{d}  — single domain time-series
DELETE /ei-growth/reset/{user_id}     — clear all data
GET  /ei-growth/domains               — list valid domain names
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["EI Growth"])

_DOMAINS = [
    "intrapersonal",
    "interpersonal",
    "stress_management",
    "adaptability",
    "general_mood",
]


class SessionData(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)
    stress_index: float = Field(0.5, ge=0.0, le=1.0)
    focus_index: float = Field(0.5, ge=0.0, le=1.0)
    relaxation_index: float = Field(0.5, ge=0.0, le=1.0)
    anger_index: float = Field(0.3, ge=0.0, le=1.0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    probabilities: dict = Field(default_factory=dict)


@router.post("/ei-growth/log/{user_id}")
def log_session(user_id: str, session: SessionData) -> dict:
    """Record a session's emotion metrics and return current EI snapshot.

    Typically called after each voice check-in or EEG analysis session.
    Requires at least 7 sessions to establish a baseline.
    """
    try:
        from models.ei_growth_tracker import get_tracker

        return get_tracker().log_session(user_id, session.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/ei-growth/report/{user_id}")
def get_report(user_id: str) -> dict:
    """Return full longitudinal EI growth report.

    Includes:
    - Current 5-domain scores
    - Growth vs baseline (Cohen's d effect size)
    - Weekly trend
    - Strongest/weakest domains
    - Plain-language insight
    """
    try:
        from models.ei_growth_tracker import get_tracker

        return get_tracker().get_report(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/ei-growth/domain/{user_id}/{domain}")
def get_domain_history(user_id: str, domain: str, limit: int = 30) -> dict:
    """Return time-series data for one EI domain.

    Args:
        domain: One of intrapersonal, interpersonal, stress_management,
                adaptability, general_mood.
        limit: Number of recent sessions to return (default 30).
    """
    if domain not in _DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid domain. Valid options: {_DOMAINS}",
        )
    try:
        from models.ei_growth_tracker import get_tracker

        return get_tracker().get_domain_history(user_id, domain, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/ei-growth/reset/{user_id}")
def reset_user(user_id: str) -> dict:
    """Clear all EI growth data for a user. Irreversible."""
    try:
        from models.ei_growth_tracker import get_tracker

        return get_tracker().reset(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/ei-growth/domains")
def list_domains() -> dict:
    """Return valid EI domain names and their descriptions."""
    return {
        "domains": {
            "intrapersonal": "Self-awareness and emotional insight",
            "interpersonal": "Empathy and social emotional awareness",
            "stress_management": "Stress recovery speed and regulation",
            "adaptability": "Emotional flexibility and cognitive reframing",
            "general_mood": "Baseline positive affect and optimism",
        },
        "reference": "Bar-On EQ-i 2.0 behavioral proxy model",
        "note": "Scores are [0,1] derived from voice/EEG emotion metrics. Requires 7+ sessions for baseline.",
    }
