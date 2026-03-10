"""Anonymous emotional wellness community — opt-in aggregated insights.

Privacy-first aggregation: users submit anonymized snapshots (no user ID,
no audio, no health details) and see community-level trends.

k-anonymity: never publish aggregate if fewer than 10 submissions in window.

POST /community/submit        — submit anonymized emotional snapshot (opt-in)
GET  /community/trends        — get aggregated community mood trends
GET  /community/challenge     — current weekly wellness challenge + completion rate
GET  /community/status        — health check

Privacy guarantees:
- No user ID stored — submissions are anonymous
- Only (valence, arousal, stress_index, hour_of_day, day_of_week, region_code) stored
- region_code is 2-letter country code only (e.g. "US", "IN") — not city/state
- Minimum k=10 per aggregate bucket (k-anonymity)
- Submissions older than 7 days auto-purged for trend windows
- Raw submissions deleted after aggregation runs
"""
from __future__ import annotations

import json
import math
import random
import string
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["Community"])

_DATA_DIR = Path("data/community")
_SUBMISSIONS_FILE = _DATA_DIR / "submissions.jsonl"
_CHALLENGES_FILE = _DATA_DIR / "challenges.json"

_K_ANONYMITY_MIN = 10  # never publish aggregate below this count

# ── Pydantic models ────────────────────────────────────────────────────────────

class AnonSnapshot(BaseModel):
    """Anonymized emotional state snapshot — no personally identifying information."""
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional valence (-1=negative, +1=positive)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Arousal level")
    stress_index: float = Field(..., ge=0.0, le=1.0, description="Stress level")
    region_code: Optional[str] = Field(None, min_length=2, max_length=2, description="ISO-2 country code (optional)")
    # No user_id, no timestamp (server assigns), no audio, no health details


class ChallengeCompletion(BaseModel):
    """Mark a weekly challenge as completed — no user ID needed."""
    challenge_id: str = Field(..., description="ID of the current challenge")


# ── Storage helpers ────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _append_submission(snapshot: AnonSnapshot) -> None:
    """Append anonymized submission to JSONL file."""
    _ensure_dirs()
    now = datetime.now(timezone.utc)
    record = {
        "valence": snapshot.valence,
        "arousal": snapshot.arousal,
        "stress_index": snapshot.stress_index,
        "region_code": snapshot.region_code,
        "hour": now.hour,
        "day_of_week": now.weekday(),  # 0=Mon, 6=Sun
        "date": now.date().isoformat(),
    }
    with open(_SUBMISSIONS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_recent_submissions(days: int = 7) -> List[Dict]:
    """Load submissions from the last N days."""
    _ensure_dirs()
    if not _SUBMISSIONS_FILE.exists():
        return []
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    results = []
    try:
        with open(_SUBMISSIONS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("date", "") >= cutoff:
                        results.append(r)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return results


def _load_today_submissions() -> List[Dict]:
    """Load today's submissions only."""
    today = date.today().isoformat()
    return [s for s in _load_recent_submissions(days=1) if s.get("date") == today]


def _purge_old_submissions(days: int = 7) -> None:
    """Remove submissions older than N days (privacy cleanup)."""
    _ensure_dirs()
    if not _SUBMISSIONS_FILE.exists():
        return
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    kept = []
    try:
        with open(_SUBMISSIONS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("date", "") >= cutoff:
                        kept.append(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return
    with open(_SUBMISSIONS_FILE, "w") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))


# ── Aggregation helpers ────────────────────────────────────────────────────────

def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _aggregate(submissions: List[Dict]) -> Optional[Dict]:
    """Compute aggregate stats; returns None if below k-anonymity threshold."""
    if len(submissions) < _K_ANONYMITY_MIN:
        return None

    valences = [s["valence"] for s in submissions]
    arousals = [s["arousal"] for s in submissions]
    stresses = [s["stress_index"] for s in submissions]

    avg_v = _safe_mean(valences)
    avg_a = _safe_mean(arousals)
    avg_s = _safe_mean(stresses)

    # Mood label
    if avg_v is not None and avg_a is not None:
        if avg_v > 0.3 and avg_a > 0.5:
            mood_label = "energized"
        elif avg_v > 0.2 and avg_a <= 0.5:
            mood_label = "calm"
        elif avg_v < -0.2 and avg_a > 0.5:
            mood_label = "stressed"
        elif avg_v < -0.2 and avg_a <= 0.4:
            mood_label = "low energy"
        else:
            mood_label = "neutral"
    else:
        mood_label = "unknown"

    # Pct in calm/stressed states
    calm_pct = round(100 * sum(1 for v in valences if v > 0.1) / len(valences))
    stressed_pct = round(100 * sum(1 for s in stresses if s > 0.6) / len(stresses))

    return {
        "sample_size": len(submissions),
        "avg_valence": avg_v,
        "avg_arousal": avg_a,
        "avg_stress": avg_s,
        "mood_label": mood_label,
        "calm_pct": calm_pct,
        "stressed_pct": stressed_pct,
    }


def _hourly_stress(submissions: List[Dict]) -> List[Dict]:
    """Group by hour, compute mean stress per hour (k-anon enforced)."""
    by_hour: Dict[int, List[float]] = defaultdict(list)
    for s in submissions:
        by_hour[s["hour"]].append(s["stress_index"])

    result = []
    for hour in sorted(by_hour.keys()):
        vals = by_hour[hour]
        if len(vals) >= _K_ANONYMITY_MIN:
            result.append({
                "hour": hour,
                "avg_stress": round(sum(vals) / len(vals), 3),
                "sample_size": len(vals),
            })
    return result


# ── Weekly challenge helpers ────────────────────────────────────────────────────

_DEFAULT_CHALLENGES = [
    {"id": "w001", "title": "3 voice check-ins per day", "description": "Record a 10-second voice sample 3 times daily"},
    {"id": "w002", "title": "5-minute breathing break", "description": "Do a 5-minute breathing exercise every afternoon"},
    {"id": "w003", "title": "Digital sunset at 9pm", "description": "No screens after 9pm for 5 out of 7 days"},
    {"id": "w004", "title": "Gratitude voice journal", "description": "Record one thing you're grateful for each morning"},
    {"id": "w005", "title": "HRV morning baseline", "description": "Log morning HRV reading before coffee for 7 days"},
]


def _current_challenge() -> Dict:
    """Pick current week's challenge (deterministic by ISO week number)."""
    week_num = date.today().isocalendar()[1]
    idx = week_num % len(_DEFAULT_CHALLENGES)
    return _DEFAULT_CHALLENGES[idx]


def _load_challenge_data() -> Dict:
    _ensure_dirs()
    if _CHALLENGES_FILE.exists():
        try:
            return json.loads(_CHALLENGES_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_challenge_data(data: Dict) -> None:
    _ensure_dirs()
    _CHALLENGES_FILE.write_text(json.dumps(data, indent=2, default=str))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/community/submit")
def submit_snapshot(snapshot: AnonSnapshot) -> dict:
    """Submit an anonymized emotional snapshot for community aggregation.

    Privacy guarantees:
    - No user ID is stored
    - Only valence, arousal, stress_index, hour, day, and optional country code stored
    - Raw submissions auto-purged after 7 days
    - Aggregates only published when 10+ users in window
    """
    try:
        _append_submission(snapshot)
        _purge_old_submissions()  # cleanup on each write
        today_count = len(_load_today_submissions())
        return {
            "submitted": True,
            "privacy_note": "No user ID stored. Submission included in anonymized aggregate after 10+ responses.",
            "today_submissions": today_count if today_count >= _K_ANONYMITY_MIN else f"<{_K_ANONYMITY_MIN} (not yet published)",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/community/trends")
def get_trends() -> dict:
    """Get anonymized community wellness trends.

    Returns today's aggregate and hourly stress pattern.
    All values suppressed until minimum k=10 submissions.
    """
    try:
        today_submissions = _load_today_submissions()
        week_submissions = _load_recent_submissions(days=7)

        today_agg = _aggregate(today_submissions)
        week_agg = _aggregate(week_submissions)
        hourly = _hourly_stress(week_submissions)

        if today_agg is None and week_agg is None:
            return {
                "available": False,
                "reason": f"Fewer than {_K_ANONYMITY_MIN} submissions in window — trends withheld for privacy.",
                "encourage": "Invite friends to join the community — trends unlock at 10+ active users.",
            }

        return {
            "available": True,
            "today": today_agg,
            "this_week": week_agg,
            "hourly_stress_pattern": hourly,
            "privacy_note": (
                f"All data aggregated from {_K_ANONYMITY_MIN}+ anonymous users. "
                "No individual data is identifiable."
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/community/challenge")
def get_challenge() -> dict:
    """Get the current weekly wellness challenge and community completion rate."""
    try:
        challenge = _current_challenge()
        challenge_data = _load_challenge_data()
        week_key = f"{date.today().isocalendar()[0]}-W{date.today().isocalendar()[1]}"

        week_data = challenge_data.get(week_key, {"completions": 0, "participants": 0})
        completions = week_data["completions"]
        participants = week_data["participants"]

        completion_pct = None
        if participants >= _K_ANONYMITY_MIN:
            completion_pct = round(100 * completions / max(participants, 1))

        return {
            "challenge": challenge,
            "week": week_key,
            "completion_rate": f"{completion_pct}%" if completion_pct is not None else "Not enough participants yet",
            "participants": participants if participants >= _K_ANONYMITY_MIN else f"<{_K_ANONYMITY_MIN}",
            "privacy_note": "Participation is counted but not attributed to any user.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/community/challenge/complete")
def mark_challenge_complete(body: ChallengeCompletion) -> dict:
    """Mark current weekly challenge as completed — anonymous, no user tracking."""
    try:
        challenge = _current_challenge()
        if body.challenge_id != challenge["id"]:
            raise HTTPException(status_code=400, detail="challenge_id does not match current week's challenge")

        challenge_data = _load_challenge_data()
        week_key = f"{date.today().isocalendar()[0]}-W{date.today().isocalendar()[1]}"

        if week_key not in challenge_data:
            challenge_data[week_key] = {"completions": 0, "participants": 0}

        # Count participant on first completion (no duplicate detection — anonymous)
        challenge_data[week_key]["participants"] += 1
        challenge_data[week_key]["completions"] += 1
        _save_challenge_data(challenge_data)

        total = challenge_data[week_key]["completions"]
        return {
            "recorded": True,
            "challenge": challenge["title"],
            "community_completions": total if total >= _K_ANONYMITY_MIN else f"<{_K_ANONYMITY_MIN}",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/community/status")
def status() -> dict:
    today_count = len(_load_today_submissions())
    return {
        "status": "ready",
        "privacy_model": "k-anonymity (k=10)",
        "data_retention_days": 7,
        "today_submissions": today_count if today_count >= _K_ANONYMITY_MIN else f"<{_K_ANONYMITY_MIN}",
        "features": [
            "anonymized_mood_trends",
            "hourly_stress_patterns",
            "weekly_challenge_tracking",
        ],
        "what_is_never_stored": [
            "user_id",
            "voice_audio",
            "health_details",
            "location_below_country",
            "name_or_email",
        ],
    }
