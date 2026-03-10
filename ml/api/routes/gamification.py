"""Gamification API endpoints for NeuralDreamWorkshop.

Endpoints (prefix: /gamification, tag: Gamification)
------------------------------------------------------
POST /gamification/checkin
    Record a daily check-in for a user.

GET  /gamification/status/{user_id}
    Return full gamification status (streak, unlocks, milestones).

GET  /gamification/insights/{user_id}
    List insight features the user has unlocked via their streak.

GET  /gamification/challenges/{user_id}
    Return 3 active challenges with current progress.

GET  /gamification/leaderboard
    Anonymized community streak leaderboard (mock top-10).
"""
from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.gamification import GamificationEngine

router = APIRouter(prefix="/gamification", tags=["Gamification"])

# Module-level singleton — shared across all requests
_engine = GamificationEngine()


# ── Request models ───────────────────────────────────────────────────────────


class CheckinRequest(BaseModel):
    """Request body for recording a daily check-in."""

    user_id: str = Field(..., description="User identifier")
    timestamp: Optional[float] = Field(
        default=None,
        description="Unix timestamp of the check-in. Defaults to server time.",
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/checkin")
async def record_checkin(req: CheckinRequest) -> dict:
    """Record a daily check-in for the user.

    Updates the streak (flexible: gap ≤ 1.5 days keeps streak alive),
    checks for new insight unlocks, and detects milestone completions.

    Returns the current streak, total check-ins, any newly unlocked
    insight features, and a milestone celebration message if applicable.
    """
    ts = req.timestamp if req.timestamp is not None else time.time()
    result = _engine.record_checkin(user_id=req.user_id, timestamp=ts)
    return result


@router.get("/status/{user_id}")
async def get_status(user_id: str) -> dict:
    """Return the full gamification status for a user.

    Includes:
    - Current streak and total check-ins
    - List of unlocked insight features
    - Milestones already achieved
    - Next milestone target (days remaining)
    """
    return _engine.get_status(user_id=user_id)


@router.get("/insights/{user_id}")
async def get_insights(user_id: str) -> dict:
    """List insight features the user has unlocked via their streak.

    Unlocks:
     7 days  → weekly_patterns
    14 days  → supplement_correlations
    30 days  → monthly_brain_report
    60 days  → personalized_model
    """
    unlocked: List[str] = _engine.get_insights_unlocked(user_id=user_id)
    return {"user_id": user_id, "unlocked_insights": unlocked, "count": len(unlocked)}


@router.get("/challenges/{user_id}")
async def get_challenges(user_id: str) -> dict:
    """Return 3 active challenges with current progress for the user.

    Challenges include morning routine tracking, weekly streak,
    and the monthly warrior milestone.
    """
    challenges = _engine.get_challenges(user_id=user_id)
    return {"user_id": user_id, "challenges": challenges}


@router.get("/leaderboard")
async def get_leaderboard() -> dict:
    """Return anonymized community streak leaderboard.

    Top-10 streak lengths from the community without any user
    identifiers — designed to inspire without enabling surveillance.
    """
    # Mock community leaderboard — no real user IDs exposed
    mock_streaks = [
        {"rank": 1, "streak_days": 187, "badge": "Neuronaut"},
        {"rank": 2, "streak_days": 142, "badge": "Dream Architect"},
        {"rank": 3, "streak_days": 98, "badge": "Flow Master"},
        {"rank": 4, "streak_days": 75, "badge": "Alpha Rider"},
        {"rank": 5, "streak_days": 63, "badge": "Theta Seeker"},
        {"rank": 6, "streak_days": 51, "badge": "Beta Breaker"},
        {"rank": 7, "streak_days": 44, "badge": "Delta Diver"},
        {"rank": 8, "streak_days": 38, "badge": "Spindle Surfer"},
        {"rank": 9, "streak_days": 30, "badge": "Monthly Warrior"},
        {"rank": 10, "streak_days": 21, "badge": "Pattern Scout"},
    ]
    return {
        "leaderboard": mock_streaks,
        "note": "Anonymized community streaks — no user IDs are stored or shared.",
    }
