"""Ethical gamification engine for NeuralDreamWorkshop.

Tracks daily check-ins, streaks, insight unlocks, and milestones without
any external database — persists to a local JSON file using threading.Lock
for safe concurrent access.

Unlock thresholds
-----------------
 7 days  → weekly_patterns
14 days  → supplement_correlations
30 days  → monthly_brain_report
60 days  → personalized_model

Flexible streak rule: a streak continues if the gap between the last
check-in and the current one is ≤ 1.5 days (36 hours), allowing users
to miss a single day without losing their streak.
"""
from __future__ import annotations

import json
import logging
import os
import time
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_DATA_DIR = "data"
_DATA_FILE = os.path.join(_DATA_DIR, "gamification.json")

# seconds in a day
_DAY_SEC = 86_400.0

# streak continues if gap ≤ this many days (allows one missed day)
_STREAK_GAP_DAYS = 1.5

UNLOCK_THRESHOLDS: Dict[int, str] = {
    7: "weekly_patterns",
    14: "supplement_correlations",
    30: "monthly_brain_report",
    60: "personalized_model",
}

MILESTONE_DAYS = [7, 14, 30, 60, 90]

MILESTONE_MESSAGES: Dict[int, str] = {
    7: "One week strong! Your brain data is starting to reveal weekly patterns.",
    14: "Two weeks in! You've unlocked supplement correlation analysis.",
    30: "One month! A full monthly brain report is now available for you.",
    60: "60 days — your personalized model is now training on your data.",
    90: "90 days of consistent brain monitoring. You're in the top tier.",
}

CHALLENGE_DEFINITIONS: List[dict] = [
    {
        "id": "morning_checkin",
        "name": "Morning Routine",
        "description": "Check in before 9 AM 3 days in a row",
        "target": 3,
    },
    {
        "id": "weekly_streak",
        "name": "Seven-Day Streak",
        "description": "Check in every day for 7 consecutive days",
        "target": 7,
    },
    {
        "id": "monthly_warrior",
        "name": "Monthly Warrior",
        "description": "Reach a 30-day streak",
        "target": 30,
    },
]

# ── Persistence helpers ───────────────────────────────────────────────────────


def _load_data() -> dict:
    """Load gamification data from JSON file; return empty dict on failure."""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_DATA_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("gamification: failed to load %s (%s), starting fresh", _DATA_FILE, exc)
        return {}


def _save_data(data: dict) -> None:
    """Persist gamification data to JSON file; log on failure."""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        tmp = _DATA_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, _DATA_FILE)
    except OSError as exc:
        logger.error("gamification: failed to save %s (%s)", _DATA_FILE, exc)


# ── Empty user record ─────────────────────────────────────────────────────────


def _empty_user() -> dict:
    return {
        "streak": 0,
        "total_checkins": 0,
        "last_checkin_ts": None,
        "checkin_timestamps": [],
        "unlocked_insights": [],
        "milestones_hit": [],
    }


# ── GamificationEngine ────────────────────────────────────────────────────────


class GamificationEngine:
    """Core gamification logic with JSON file persistence.

    Thread-safe: all mutations are protected by a threading.Lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def record_checkin(
        self,
        user_id: str,
        timestamp: Optional[float] = None,
    ) -> dict:
        """Record a daily check-in for *user_id*.

        Returns a dict with:
            streak           (int)   current streak after this check-in
            total_checkins   (int)   total check-ins for this user
            new_unlocks      (list)  insight keys unlocked by this check-in
            milestone_hit    (str|None) celebration message if a milestone was reached
            already_checked_in (bool) True if the user already checked in today
        """
        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            data = _load_data()
            user = data.get(user_id, _empty_user())

            last_ts = user.get("last_checkin_ts")
            already_checked_in = False

            if last_ts is not None:
                gap_days = (ts - last_ts) / _DAY_SEC
                if gap_days < 1.0:
                    # Already checked in during this calendar-day window
                    already_checked_in = True
                elif gap_days <= _STREAK_GAP_DAYS:
                    # Within the flexible window — streak continues
                    user["streak"] = user.get("streak", 0) + 1
                else:
                    # Streak broken — start fresh
                    user["streak"] = 1
            else:
                # First ever check-in
                user["streak"] = 1

            if not already_checked_in:
                user["total_checkins"] = user.get("total_checkins", 0) + 1
                user["last_checkin_ts"] = ts
                checkin_list = user.get("checkin_timestamps", [])
                checkin_list.append(ts)
                user["checkin_timestamps"] = checkin_list

            current_streak = user["streak"]

            # ── Insight unlocks ───────────────────────────────────────────
            already_unlocked = set(user.get("unlocked_insights", []))
            new_unlocks: List[str] = []
            for threshold, insight_key in sorted(UNLOCK_THRESHOLDS.items()):
                if current_streak >= threshold and insight_key not in already_unlocked:
                    new_unlocks.append(insight_key)
                    already_unlocked.add(insight_key)
            user["unlocked_insights"] = sorted(already_unlocked)

            # ── Milestones ────────────────────────────────────────────────
            milestones_hit = set(user.get("milestones_hit", []))
            milestone_hit_msg: Optional[str] = None
            for day in MILESTONE_DAYS:
                if current_streak >= day and day not in milestones_hit:
                    milestones_hit.add(day)
                    milestone_hit_msg = MILESTONE_MESSAGES.get(day)
                    break  # report one milestone celebration per check-in
            user["milestones_hit"] = sorted(milestones_hit)

            data[user_id] = user
            _save_data(data)

        return {
            "streak": current_streak,
            "total_checkins": user["total_checkins"],
            "new_unlocks": new_unlocks,
            "milestone_hit": milestone_hit_msg,
            "already_checked_in": already_checked_in,
        }

    def get_status(self, user_id: str) -> dict:
        """Return full gamification status for *user_id*.

        Returns:
            user_id          (str)
            streak           (int)
            total_checkins   (int)
            unlocked_insights (list[str])
            milestones_hit   (list[int])
            next_milestone   (int|None)  days until next milestone
        """
        with self._lock:
            data = _load_data()
        user = data.get(user_id, _empty_user())

        current_streak = user.get("streak", 0)
        milestones_hit = user.get("milestones_hit", [])
        next_milestone: Optional[int] = None
        for day in MILESTONE_DAYS:
            if day > current_streak:
                next_milestone = day
                break

        return {
            "user_id": user_id,
            "streak": current_streak,
            "total_checkins": user.get("total_checkins", 0),
            "unlocked_insights": user.get("unlocked_insights", []),
            "milestones_hit": milestones_hit,
            "next_milestone": next_milestone,
        }

    def get_insights_unlocked(self, user_id: str) -> List[str]:
        """Return list of insight feature keys unlocked for *user_id*."""
        with self._lock:
            data = _load_data()
        user = data.get(user_id, _empty_user())
        return user.get("unlocked_insights", [])

    def get_challenges(self, user_id: str) -> List[dict]:
        """Return 3 active challenges with current progress for *user_id*.

        Challenges:
            morning_checkin  — Check in before 9 AM 3 days in a row
            weekly_streak    — Check in every day for 7 consecutive days
            monthly_warrior  — Reach a 30-day streak
        """
        with self._lock:
            data = _load_data()
        user = data.get(user_id, _empty_user())

        streak = user.get("streak", 0)
        checkin_timestamps = user.get("checkin_timestamps", [])

        # Compute morning check-in progress (before 09:00 local time)
        morning_count = _count_morning_checkins_streak(checkin_timestamps)

        results: List[dict] = []
        for defn in CHALLENGE_DEFINITIONS:
            cid = defn["id"]
            if cid == "morning_checkin":
                progress = min(morning_count, defn["target"])
            elif cid == "weekly_streak":
                progress = min(streak, defn["target"])
            elif cid == "monthly_warrior":
                progress = min(streak, defn["target"])
            else:
                progress = 0

            results.append(
                {
                    "id": cid,
                    "name": defn["name"],
                    "description": defn["description"],
                    "progress": progress,
                    "target": defn["target"],
                    "completed": progress >= defn["target"],
                }
            )

        return results


# ── Helpers ───────────────────────────────────────────────────────────────────


def _count_morning_checkins_streak(timestamps: List[float]) -> int:
    """Return the current consecutive morning check-in streak.

    A check-in counts as "morning" if it falls before 09:00 local time.
    The streak resets whenever a check-in is not before 09:00.
    """
    import datetime

    if not timestamps:
        return 0

    streak = 0
    for ts in reversed(timestamps):
        dt = datetime.datetime.fromtimestamp(ts)
        if dt.hour < 9:
            streak += 1
        else:
            break
    return streak
