"""Epigenetic emotional inheritance API routes.

POST /epigenetic/analyze        -- analyze family history against user's emotional data
POST /epigenetic/family-history -- submit family history
GET  /epigenetic/status         -- get user state summary

GitHub issue: #449
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/epigenetic", tags=["epigenetic-inheritance"])

# ── Lazy singleton ──────────────────────────────────────────────────────────

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        try:
            from models.epigenetic_model import get_epigenetic_engine
            _engine = get_epigenetic_engine()
        except Exception as exc:
            log.warning("EpigeneticEngine unavailable: %s", exc)
    return _engine


# ── Pydantic schemas ────────────────────────────────────────────────────────

class FamilyMemberSchema(BaseModel):
    relation: str = Field(..., description="One of: mother, father, maternal_grandmother, etc.")
    conditions: List[str] = Field(..., description="List of conditions: anxiety, depression, trauma, addiction, resilience, anger, grief, phobia")
    severity: float = Field(..., ge=0.0, le=10.0, description="Overall severity 0-10")
    notes: str = Field("", description="Optional notes")


class FamilyHistoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    members: List[FamilyMemberSchema] = Field(..., description="List of family members with emotional history")


class EmotionalDataRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    date: str = Field(..., description="YYYY-MM-DD")
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    stress_level: float = Field(..., ge=0.0, le=1.0)
    trigger: str = Field("", description="Optional trigger label")


class AnalyzeRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/family-history")
def submit_family_history(req: FamilyHistoryRequest) -> Dict[str, Any]:
    """Submit family emotional history for epigenetic analysis.

    Records structured questionnaire data about parents/grandparents
    including conditions (anxiety, depression, trauma, etc.) and severity.
    """
    engine = _get_engine()
    if engine is None:
        raise HTTPException(503, "EpigeneticEngine unavailable")

    try:
        from models.epigenetic_model import FamilyMember
        members = [
            FamilyMember(
                relation=m.relation,
                conditions=m.conditions,
                severity=m.severity,
                notes=m.notes,
            )
            for m in req.members
        ]
        stored = engine.intake_family_history(user_id=req.user_id, members=members)
    except Exception as exc:
        log.exception("intake_family_history failed: %s", exc)
        raise HTTPException(500, f"Error: {exc}")

    return {
        "user_id": req.user_id,
        "members_stored": len(stored),
        "members": [
            {
                "relation": m.relation,
                "conditions": m.conditions,
                "severity": m.severity,
            }
            for m in stored
        ],
    }


@router.post("/analyze")
def analyze_epigenetic(req: AnalyzeRequest) -> Dict[str, Any]:
    """Analyze family history against user's emotional data.

    Computes inherited patterns, risk/attenuation scores, protective factors,
    and generational healing metrics in a single comprehensive profile.
    """
    engine = _get_engine()
    if engine is None:
        raise HTTPException(503, "EpigeneticEngine unavailable")

    try:
        profile = engine.compute_epigenetic_profile(req.user_id)
    except Exception as exc:
        log.exception("compute_epigenetic_profile failed: %s", exc)
        raise HTTPException(500, f"Analysis error: {exc}")

    return profile


@router.get("/status")
def get_status(
    user_id: str = Query("default", min_length=1, description="User identifier"),
) -> Dict[str, Any]:
    """Get epigenetic tracking status for a user.

    Returns family history summary, emotional data count,
    and available condition/relation type definitions.
    """
    engine = _get_engine()
    if engine is None:
        raise HTTPException(503, "EpigeneticEngine unavailable")

    try:
        result = engine.profile_to_dict(user_id)
    except Exception as exc:
        log.exception("profile_to_dict failed: %s", exc)
        raise HTTPException(500, f"Status error: {exc}")

    return result
