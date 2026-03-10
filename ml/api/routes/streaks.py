"""Ethical gamification — streak tracking with forgiving logic.

POST /streaks/checkin           — record a check-in for today
GET  /streaks/status/{user_id} — current streak, milestones, unlocked features
GET  /streaks/history/{user_id} — last 90 days of check-in dates
"""
from __future__ import annotations

import logging
from datetime import date, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/streaks", tags=["streaks"])

# ── In-memory storage ─────────────────────────────────────────────────────────

# Keyed by user_id.  Each entry:
#   checkin_dates : set of ISO date strings (YYYY-MM-DD) that had at least one check-in
#   flexible_used : number of grace days consumed in the *current* streak window
#   window_start  : date when the current 7-day grace window started
_store: Dict[str, Dict[str, Any]] = {}


def _user_data(user_id: str) -> Dict[str, Any]:
    if user_id not in _store:
        _store[user_id] = {
            "checkin_dates": set(),
            "flexible_used": 0,
            "window_start": None,  # date or None
        }
    return _store[user_id]


# ── Milestone definitions ─────────────────────────────────────────────────────

MILESTONES: List[int] = [3, 7, 14, 30, 60, 90]

# Features unlocked at specific milestone thresholds
UNLOCKS: Dict[int, str] = {
    7: "weekly_patterns",
    14: "supplement_correlations",
    30: "monthly_report",
}
# 30 days also unlocks personal_model
UNLOCKS_EXTRA: Dict[int, List[str]] = {
    30: ["monthly_report", "personal_model"],
}


def _unlocked_features(streak_days: int) -> List[str]:
    """Return the list of feature keys unlocked at or below current streak."""
    features: List[str] = []
    # 7 days
    if streak_days >= 7:
        features.append("weekly_patterns")
    # 14 days
    if streak_days >= 14:
        features.append("supplement_correlations")
    # 30 days
    if streak_days >= 30:
        features.append("monthly_report")
        features.append("personal_model")
    return features


def _milestones_achieved(streak_days: int) -> List[str]:
    return [f"{m}_days" for m in MILESTONES if streak_days >= m]


def _next_milestone(streak_days: int) -> Optional[int]:
    for m in MILESTONES:
        if streak_days < m:
            return m
    return None


# ── Core streak computation ───────────────────────────────────────────────────

def _today_utc() -> date:
    return date.today()  # server UTC date (uvicorn runs in UTC)


def _compute_streak(checkin_dates: set, today: date) -> Dict[str, Any]:
    """Compute forgiving streak: 1 grace day per 7-day cycle.

    Walk backward from today (or yesterday if today has no check-in yet).
    Within each 7-day window, one missed day is allowed without breaking
    the streak (flexible day).  The grace counter resets every 7
    consecutive days.

    Returns a dict with:
        streak_days, longest_streak, today_checked_in,
        flexible_days_used, last_checkin_date
    """
    today_str = today.isoformat()
    today_checked_in = today_str in checkin_dates

    if not checkin_dates:
        return {
            "streak_days": 0,
            "longest_streak": 0,
            "today_checked_in": False,
            "flexible_days_used": 0,
            "last_checkin_date": None,
        }

    sorted_dates = sorted(checkin_dates, reverse=True)
    last_checkin_date = sorted_dates[0]

    # Start counting from today if checked in, else from yesterday
    start = today if today_checked_in else today - timedelta(days=1)

    # If the last check-in is older than yesterday+1 (i.e. 2+ days ago from
    # today and we haven't checked in today), the streak is broken — unless
    # a flexible day can cover the gap.
    last_date = date.fromisoformat(last_checkin_date)
    gap_from_start = (start - last_date).days

    # If gap > 1 already on the starting day, the streak can't begin
    if not today_checked_in and gap_from_start > 2:
        return {
            "streak_days": 0,
            "longest_streak": _longest_streak(checkin_dates),
            "today_checked_in": False,
            "flexible_days_used": 0,
            "last_checkin_date": last_checkin_date,
        }

    streak = 0
    flexible_used = 0
    window_days = 0  # consecutive days (with or without grace) in current 7-day window

    current = start
    # Walk backward until we hit a gap we can't bridge
    while True:
        current_str = current.isoformat()
        if current_str in checkin_dates:
            streak += 1
            window_days += 1
        else:
            # One grace day allowed per 7-day cycle
            if flexible_used < 1:
                flexible_used += 1
                streak += 1
                window_days += 1
            else:
                break

        # Reset flexible allowance every 7 consecutive streak days
        if window_days > 0 and window_days % 7 == 0:
            flexible_used = 0

        current = current - timedelta(days=1)

        # Safety: don't walk further than 365 days
        if (start - current).days > 365:
            break

    return {
        "streak_days": streak,
        "longest_streak": max(streak, _longest_streak(checkin_dates)),
        "today_checked_in": today_checked_in,
        "flexible_days_used": flexible_used,
        "last_checkin_date": last_checkin_date,
    }


def _longest_streak(checkin_dates: set) -> int:
    """Compute all-time longest strict consecutive-day streak."""
    if not checkin_dates:
        return 0
    sorted_d = sorted(date.fromisoformat(s) for s in checkin_dates)
    best = 1
    current = 1
    for i in range(1, len(sorted_d)):
        if (sorted_d[i] - sorted_d[i - 1]).days == 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class CheckinRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    checkin_type: Literal["voice", "eeg", "health"] = Field(
        "voice", description="Type of check-in"
    )


class CheckinResponse(BaseModel):
    user_id: str
    date: str
    checkin_type: str
    streak_days: int
    today_checked_in: bool
    message: str


class StreakStatusResponse(BaseModel):
    user_id: str
    streak_days: int
    longest_streak: int
    today_checked_in: bool
    flexible_days_used: int
    milestones_achieved: List[str]
    next_milestone: Optional[int]
    unlocked_features: List[str]
    last_checkin_date: Optional[str]


class StreakHistoryResponse(BaseModel):
    user_id: str
    days_with_checkins: List[str]   # ISO date strings, last 90 days only
    total_checkin_days: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/checkin", response_model=CheckinResponse)
def record_checkin(req: CheckinRequest) -> Dict[str, Any]:
    """Record a check-in for today (idempotent — multiple calls on same day are fine)."""
    data = _user_data(req.user_id)
    today = _today_utc()
    today_str = today.isoformat()
    data["checkin_dates"].add(today_str)

    streak_info = _compute_streak(data["checkin_dates"], today)

    log.info(
        "Check-in recorded: user=%s type=%s date=%s streak=%d",
        req.user_id,
        req.checkin_type,
        today_str,
        streak_info["streak_days"],
    )

    streak_days = streak_info["streak_days"]
    msg = f"Day {streak_days} streak!"
    if streak_days in MILESTONES:
        msg = f"Milestone reached: {streak_days} days!"
    elif streak_days == 1:
        msg = "Streak started — keep it going!"

    return {
        "user_id": req.user_id,
        "date": today_str,
        "checkin_type": req.checkin_type,
        "streak_days": streak_days,
        "today_checked_in": True,
        "message": msg,
    }


@router.get("/status/{user_id}", response_model=StreakStatusResponse)
def get_streak_status(user_id: str) -> Dict[str, Any]:
    """Return current streak, milestones reached, and unlocked features."""
    data = _user_data(user_id)
    today = _today_utc()
    info = _compute_streak(data["checkin_dates"], today)

    streak_days = info["streak_days"]
    return {
        "user_id": user_id,
        "streak_days": streak_days,
        "longest_streak": info["longest_streak"],
        "today_checked_in": info["today_checked_in"],
        "flexible_days_used": info["flexible_days_used"],
        "milestones_achieved": _milestones_achieved(streak_days),
        "next_milestone": _next_milestone(streak_days),
        "unlocked_features": _unlocked_features(streak_days),
        "last_checkin_date": info["last_checkin_date"],
    }


@router.get("/history/{user_id}", response_model=StreakHistoryResponse)
def get_streak_history(user_id: str) -> Dict[str, Any]:
    """Return the last 90 days that had at least one check-in."""
    data = _user_data(user_id)
    today = _today_utc()
    cutoff = today - timedelta(days=90)

    recent = sorted(
        d for d in data["checkin_dates"] if date.fromisoformat(d) >= cutoff
    )

    return {
        "user_id": user_id,
        "days_with_checkins": recent,
        "total_checkin_days": len(data["checkin_dates"]),
    }
