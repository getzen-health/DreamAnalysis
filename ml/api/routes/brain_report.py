"""Daily Brain Report — no-EEG mode.

Generates a daily brain report from voice check-ins + Apple Health data when
no EEG session is available.  Falls back gracefully when health or voice data
is sparse.

Endpoints
---------
GET /brain-report/{user_id}
    Returns a structured report with:
    - data_sources: list of what was used ("voice", "health", "eeg")
    - sleep_quality: 0-100 or null
    - focus_forecast: 0-100
    - stress_risk: 0-100
    - dominant_mood: emotion label or null
    - recommended_action: string
    - peak_focus_window: "HH:MM–HH:MM" or null
    - insight: one-liner about yesterday's patterns

GET /brain-report/readiness-score/{user_id}
    Returns Brain Readiness Score (0-100) computed from:
    - sleep quality (40%), stress avg (25%), HRV trend (20%), voice emotion (15%)
    - factor breakdown and 7-day history

GET /brain-report/streak/{user_id}
    Returns check-in streak computed from sessions + voice check-ins.
    Tracks current streak, best streak, and milestone progress.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from .voice_watch import _VOICE_CACHE, _VOICE_CACHE_TTL, _VOICE_HISTORY

log = logging.getLogger(__name__)
router = APIRouter(prefix="/brain-report", tags=["Brain Report"])

# ── In-memory streak store (keyed by participant_id → list of ISO date strings) ──
# Persisted to a simple JSON file next to this module so it survives restarts.
_STREAK_FILE = os.path.join(os.path.dirname(__file__), "_streak_data.json")
_streak_store: Dict[str, List[str]] = {}


def _load_streak_store() -> None:
    global _streak_store
    try:
        if os.path.exists(_STREAK_FILE):
            with open(_STREAK_FILE, "r") as f:
                _streak_store = json.load(f)
    except Exception as exc:
        log.debug("Could not load streak store: %s", exc)
        _streak_store = {}


def _save_streak_store() -> None:
    try:
        with open(_STREAK_FILE, "w") as f:
            json.dump(_streak_store, f)
    except Exception as exc:
        log.debug("Could not save streak store: %s", exc)


_load_streak_store()


# ── Response model ────────────────────────────────────────────────────────────

class BrainReport(BaseModel):
    user_id: str
    date: str
    data_sources: List[str]          # ["voice", "health"] or ["eeg", "voice", "health"]
    sleep_quality: Optional[float]   # 0-100 or null
    focus_forecast: float            # 0-100
    stress_risk: float               # 0-100
    dominant_mood: Optional[str]
    mood_valence: Optional[float]    # -1 to 1
    recommended_action: str
    peak_focus_window: Optional[str] # "09:00–11:00" or null
    insight: str
    has_eeg: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _yesterday_voice(user_id: str) -> Dict[str, Any]:
    """Pull latest voice result from the canonical voice-watch cache."""
    import time
    entry = _VOICE_CACHE.get(user_id)
    if not entry:
        return {}
    # Honour TTL — stale cache is treated as no data
    if time.time() - entry.get("ts", 0) > _VOICE_CACHE_TTL:
        return {}
    r = entry.get("result", {})
    stress_raw = r.get("stress_from_watch")
    stress_index = min(1.0, float(stress_raw) / 10.0) if stress_raw is not None else r.get("stress_index", 0.0)
    return {
        "avg_valence":      float(r.get("valence", 0.0)),
        "avg_stress":       float(stress_index),
        "avg_arousal":      float(r.get("arousal", 0.5)),
        "evening_stress":   float(stress_index),
        "dominant_emotion": r.get("emotion", "neutral"),
        "count": 1,
    }


def _peak_focus_from_arousal(user_id: str) -> Optional[str]:
    """Find the hour-of-day with highest historical arousal from voice history."""
    import numpy as np
    from collections import defaultdict

    hour_arousals: Dict[int, list] = defaultdict(list)
    for entry in _VOICE_HISTORY.get(user_id, []):
        h = datetime.datetime.utcfromtimestamp(entry["timestamp"]).hour
        hour_arousals[h].append(entry.get("arousal", 0.0))

    if not hour_arousals:
        return None

    best_hour = max(hour_arousals, key=lambda h: float(np.mean(hour_arousals[h])))
    return f"{best_hour:02d}:00–{best_hour + 2:02d}:00"


def _health_daily_summary(user_id: str) -> Dict[str, Any]:
    """Pull today's health summary from the health DB (best-effort)."""
    try:
        from .health import _health_db  # type: ignore[attr-defined]
        result = _health_db.get_daily_summary(user_id, None)
        if isinstance(result, dict):
            return result
    except Exception as exc:
        log.debug("Health DB not available: %s", exc)
    return {}


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/{user_id}", response_model=BrainReport)
async def get_brain_report(user_id: str) -> BrainReport:
    """Generate a daily brain report using voice + health (EEG optional)."""
    today = datetime.date.today().isoformat()
    data_sources: List[str] = []

    # ── Voice data ────────────────────────────────────────────────────────────
    voice = _yesterday_voice(user_id)
    if voice:
        data_sources.append("voice")

    # ── Health data ───────────────────────────────────────────────────────────
    health = _health_daily_summary(user_id)
    if health:
        data_sources.append("health")

    # ── EEG presence check (simple: any EEG in health summary) ───────────────
    has_eeg = bool(health.get("eeg_sessions") or health.get("alpha_power"))
    if has_eeg:
        data_sources.append("eeg")

    # ── Sleep quality ─────────────────────────────────────────────────────────
    sleep_quality: Optional[float] = None
    if "sleep_efficiency" in health and health["sleep_efficiency"] is not None:
        sleep_quality = _clamp(float(health["sleep_efficiency"]))
    elif "sleep_score" in health and health["sleep_score"] is not None:
        sleep_quality = _clamp(float(health["sleep_score"]))

    # ── HRV ───────────────────────────────────────────────────────────────────
    hrv_sdnn: Optional[float] = health.get("hrv_sdnn")
    hrv_norm: float = 0.5  # neutral default
    if hrv_sdnn and hrv_sdnn > 0:
        # Normalise: 70 ms = excellent (1.0), 20 ms = poor (0.0)
        hrv_norm = _clamp((float(hrv_sdnn) - 20) / 50, 0.0, 1.0)

    # ── Focus forecast (0-100) ────────────────────────────────────────────────
    focus_components: List[float] = []
    weights: List[float] = []

    if sleep_quality is not None:
        focus_components.append(sleep_quality / 100)
        weights.append(0.40)

    if hrv_sdnn:
        focus_components.append(hrv_norm)
        weights.append(0.30)

    if voice.get("avg_valence") is not None:
        # Map valence [-1, 1] → [0, 1]
        focus_components.append((voice["avg_valence"] + 1) / 2)
        weights.append(0.30)

    if focus_components:
        total_w = sum(weights)
        focus_forecast = _clamp(sum(c * w for c, w in zip(focus_components, weights)) / total_w * 100)
    else:
        focus_forecast = 50.0  # neutral default

    # ── Stress risk (0-100) ───────────────────────────────────────────────────
    stress_components: List[float] = []
    stress_weights: List[float] = []

    stress_components.append(1 - hrv_norm)
    stress_weights.append(0.50)

    if voice.get("evening_stress") is not None:
        stress_components.append(float(voice["evening_stress"]))
        stress_weights.append(0.30)

    if sleep_quality is not None:
        poor_sleep_flag = _clamp(1 - sleep_quality / 100, 0.0, 1.0)
        stress_components.append(poor_sleep_flag)
        stress_weights.append(0.20)

    total_sw = sum(stress_weights)
    stress_risk = _clamp(sum(c * w for c, w in zip(stress_components, stress_weights)) / total_sw * 100)

    # ── Dominant mood ─────────────────────────────────────────────────────────
    dominant_mood: Optional[str] = voice.get("dominant_emotion")
    mood_valence: Optional[float] = voice.get("avg_valence")

    # ── Recommended action ────────────────────────────────────────────────────
    if stress_risk > 65:
        recommended_action = "Take a 5-minute breathing break before starting work"
    elif focus_forecast < 45:
        recommended_action = "Start with simple tasks — today's focus forecast is low"
    elif sleep_quality is not None and sleep_quality < 60:
        recommended_action = "Short nap (20 min) may help recover from poor sleep"
    elif focus_forecast > 75:
        recommended_action = "Good time for deep work — block 90 minutes this morning"
    else:
        recommended_action = "Steady day ahead — maintain regular breaks"

    # ── Peak focus window ─────────────────────────────────────────────────────
    peak_focus_window = _peak_focus_from_arousal(user_id)

    # ── Insight ───────────────────────────────────────────────────────────────
    if not voice and not health:
        insight = "Complete a voice check-in or sync health data for a personalised report"
    elif voice and not health:
        n = voice["count"]
        mood = voice.get("dominant_emotion", "neutral")
        insight = f"Yesterday's {n} voice check-in{'s' if n != 1 else ''} showed {mood} as the dominant mood"
    elif health and not voice:
        insight = f"Sleep efficiency {sleep_quality:.0f}%" if sleep_quality else "Health data synced — add voice check-ins for mood insights"
    else:
        v = voice.get("avg_valence", 0.0)
        polarity = "positive" if v > 0.1 else "negative" if v < -0.1 else "neutral"
        insight = f"Yesterday's voice mood was {polarity}; focus forecast today is {focus_forecast:.0f}/100"

    return BrainReport(
        user_id=user_id,
        date=today,
        data_sources=data_sources,
        sleep_quality=round(sleep_quality, 1) if sleep_quality is not None else None,
        focus_forecast=round(focus_forecast, 1),
        stress_risk=round(stress_risk, 1),
        dominant_mood=dominant_mood,
        mood_valence=round(mood_valence, 3) if mood_valence is not None else None,
        recommended_action=recommended_action,
        peak_focus_window=peak_focus_window,
        insight=insight,
        has_eeg=has_eeg,
    )


# ── Readiness Score endpoint ───────────────────────────────────────────────────

class ReadinessFactors(BaseModel):
    sleep_quality: Optional[float]    # 0-100 or null
    stress_avg: Optional[float]       # 0-100 or null (lower = better)
    hrv_trend: Optional[float]        # 0-100 or null
    voice_emotion: Optional[float]    # 0-100 or null


class ReadinessScoreResponse(BaseModel):
    user_id: str
    score: int                        # 0-100
    factors: ReadinessFactors
    history: List[Dict[str, Any]]     # last 7 days [{date, score}, ...]
    color: str                        # "red" | "yellow" | "green"
    label: str                        # human-readable tier


def _compute_readiness_score(
    sleep_quality: Optional[float],
    stress_avg: Optional[float],
    hrv_norm: float,
    voice_valence: Optional[float],
) -> int:
    """Compute readiness 0-100 from weighted factors.

    Weights: sleep 40%, stress (inverted) 25%, HRV 20%, voice emotion 15%.
    Falls back to neutral (50) for any missing factor.
    """
    components: List[float] = []
    weights: List[float] = []

    if sleep_quality is not None:
        components.append(sleep_quality / 100.0)
        weights.append(0.40)

    if stress_avg is not None:
        # invert stress: low stress = high readiness
        components.append(1.0 - stress_avg / 100.0)
        weights.append(0.25)

    # HRV always contributes (defaults to 0.5 neutral if unknown)
    components.append(hrv_norm)
    weights.append(0.20)

    if voice_valence is not None:
        # Map valence [-1, 1] → [0, 1]
        components.append((voice_valence + 1.0) / 2.0)
        weights.append(0.15)

    if not components:
        return 50

    total_w = sum(weights)
    raw = sum(c * w for c, w in zip(components, weights)) / total_w
    return int(_clamp(raw * 100))


@router.get("/readiness-score/{user_id}", response_model=ReadinessScoreResponse)
async def get_readiness_score(user_id: str) -> ReadinessScoreResponse:
    """Return Brain Readiness Score (0-100) plus a 7-day history."""
    voice = _yesterday_voice(user_id)
    health = _health_daily_summary(user_id)

    # Sleep quality
    sleep_quality: Optional[float] = None
    if "sleep_efficiency" in health and health["sleep_efficiency"] is not None:
        sleep_quality = _clamp(float(health["sleep_efficiency"]))
    elif "sleep_score" in health and health["sleep_score"] is not None:
        sleep_quality = _clamp(float(health["sleep_score"]))

    # HRV
    hrv_sdnn: Optional[float] = health.get("hrv_sdnn")
    hrv_norm: float = 0.5
    if hrv_sdnn and hrv_sdnn > 0:
        hrv_norm = _clamp((float(hrv_sdnn) - 20) / 50, 0.0, 1.0)
    hrv_trend_score: Optional[float] = round(hrv_norm * 100) if hrv_sdnn else None

    # Stress from voice
    stress_avg: Optional[float] = None
    if voice.get("avg_stress") is not None:
        stress_avg = round(_clamp(float(voice["avg_stress"]) * 100))

    # Voice emotion valence
    voice_valence: Optional[float] = voice.get("avg_valence")

    score = _compute_readiness_score(sleep_quality, stress_avg, hrv_norm, voice_valence)

    if score >= 70:
        color, label = "green", "Ready"
    elif score >= 50:
        color, label = "yellow", "Fair"
    else:
        color, label = "red", "Low"

    # 7-day history — approximate from voice history arousal (best available signal)
    history: List[Dict[str, Any]] = []
    today = datetime.date.today()
    for i in range(6, -1, -1):
        day = today - datetime.timedelta(days=i)
        day_str = day.isoformat()
        # Look for a voice entry on that day
        day_score: Optional[int] = None
        for entry in _VOICE_HISTORY.get(user_id, []):
            entry_date = datetime.datetime.utcfromtimestamp(entry["timestamp"]).date()
            if entry_date == day:
                v = entry.get("valence", 0.0)
                s = min(1.0, float(entry.get("stress_index", 0.5)))
                day_score = int(_clamp(((1.0 - s) * 0.60 + (v + 1.0) / 2.0 * 0.40) * 100))
                break
        if day_str == today.isoformat():
            day_score = score  # use today's computed score
        history.append({"date": day_str, "score": day_score})

    return ReadinessScoreResponse(
        user_id=user_id,
        score=score,
        factors=ReadinessFactors(
            sleep_quality=round(sleep_quality, 1) if sleep_quality is not None else None,
            stress_avg=stress_avg,
            hrv_trend=hrv_trend_score,
            voice_emotion=round((voice_valence + 1.0) / 2.0 * 100) if voice_valence is not None else None,
        ),
        history=history,
        color=color,
        label=label,
    )


# ── Streak endpoint ────────────────────────────────────────────────────────────

class StreakResponse(BaseModel):
    user_id: str
    current_streak: int
    best_streak: int
    today_checked_in: bool
    milestones: List[int]             # upcoming milestones e.g. [7, 30, 100]
    next_milestone: Optional[int]
    total_checkins: int


def _record_checkin(user_id: str) -> None:
    """Record today as a check-in day for this user."""
    today = datetime.date.today().isoformat()
    dates = _streak_store.setdefault(user_id, [])
    if today not in dates:
        dates.append(today)
        _save_streak_store()


def _compute_streak(dates: List[str]) -> tuple[int, int]:
    """Return (current_streak, best_streak) from a list of ISO date strings."""
    if not dates:
        return 0, 0

    day_set = {datetime.date.fromisoformat(d) for d in dates}
    today = datetime.date.today()
    one_day = datetime.timedelta(days=1)

    # current streak: walk backwards from today (allow yesterday as start)
    start = today if today in day_set else (today - one_day if (today - one_day) in day_set else None)
    current = 0
    if start:
        check = start
        while check in day_set:
            current += 1
            check -= one_day

    # best streak: sort and walk
    sorted_days = sorted(day_set)
    best = 1
    run = 1
    for i in range(1, len(sorted_days)):
        if sorted_days[i] - sorted_days[i - 1] == one_day:
            run += 1
            best = max(best, run)
        else:
            run = 1

    return current, max(best, current)


_MILESTONES = [7, 30, 100]


@router.get("/streak/{user_id}", response_model=StreakResponse)
async def get_streak(user_id: str) -> StreakResponse:
    """Return current/best streak and milestone info for this participant."""
    # Auto-record today as checked in if there is fresh voice or health data
    voice = _yesterday_voice(user_id)
    if voice:
        _record_checkin(user_id)

    dates = _streak_store.get(user_id, [])
    current, best = _compute_streak(dates)
    today_str = datetime.date.today().isoformat()
    today_checked_in = today_str in dates

    next_ms: Optional[int] = None
    for m in _MILESTONES:
        if current < m:
            next_ms = m
            break

    return StreakResponse(
        user_id=user_id,
        current_streak=current,
        best_streak=best,
        today_checked_in=today_checked_in,
        milestones=_MILESTONES,
        next_milestone=next_ms,
        total_checkins=len(dates),
    )


@router.post("/streak/{user_id}/checkin")
async def record_checkin(user_id: str) -> Dict[str, Any]:
    """Manually record a check-in for today. Returns updated streak."""
    _record_checkin(user_id)
    dates = _streak_store.get(user_id, [])
    current, best = _compute_streak(dates)
    return {"status": "ok", "current_streak": current, "best_streak": best}
