"""Daily Wellness Score endpoint — composite 0-100 score with components and insight.

Implements GET /wellness/daily-score/{user_id} as specified in issue #464.
Reuses the in-memory stores from health_summary to avoid duplicating state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .health_summary import _daily_records, _trend_history

router = APIRouter(prefix="/wellness", tags=["wellness"])

_WELLNESS_DISCLAIMER = (
    "This wellness score is for informational purposes only and does not "
    "constitute medical advice."
)

# ---------------------------------------------------------------------------
# Component weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_WEIGHTS: Dict[str, float] = {
    "sleep_quality":       0.30,
    "emotional_baseline":  0.25,
    "hrv_trend":           0.20,
    "stress_recovery":     0.15,
    "activity_rest":       0.10,
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class WellnessComponent(BaseModel):
    name: str
    sub_score: float       # 0-100
    weight: float          # 0.0-1.0
    weighted_contribution: float
    available: bool        # False when the source metric was missing


class DailyWellnessScore(BaseModel):
    user_id: str
    date: str
    score: float           # 0-100 composite
    color: str             # "green" | "yellow" | "red"
    insight: str           # one-sentence actionable text
    trend_7d: List[float]  # last 7 days' scores (oldest first)
    trend_direction: str   # "improving" | "stable" | "declining"
    components: List[WellnessComponent]
    disclaimer: str


# ---------------------------------------------------------------------------
# Composite wellness computation
# ---------------------------------------------------------------------------

def _score_sleep_quality(record: dict) -> tuple[float, bool]:
    """30% weight — sleep quality score (0-100 field from daily record)."""
    km = record.get("key_metrics", {})
    val = km.get("sleep_quality")
    if val is None:
        return 50.0, False
    # sleep_quality_score is already 0-100
    return float(np.clip(val, 0, 100)), True


def _score_emotional_baseline(record: dict) -> tuple[float, bool]:
    """25% weight — mean_valence [-1,1] and mean_stress [0,1] from morning EEG."""
    km = record.get("key_metrics", {})
    valence = km.get("valence")
    stress  = km.get("stress")
    if valence is None and stress is None:
        return 50.0, False
    # Map valence [-1, 1] → [0, 100]: 50 + valence*50
    v_score = (50.0 + (valence or 0.0) * 50.0)
    # Map stress [0, 1] → calm score [100, 0]: 100 - stress*100
    s_score = (100.0 - (stress or 0.5) * 100.0)
    # Average of available signals
    parts = []
    if valence is not None:
        parts.append(v_score)
    if stress is not None:
        parts.append(s_score)
    return float(np.clip(np.mean(parts), 0, 100)), True


def _score_hrv(record: dict) -> tuple[float, bool]:
    """20% weight — HRV in ms; typical healthy adult range 20-100 ms."""
    km = record.get("key_metrics", {})
    hrv = km.get("hrv_ms") or record.get("hrv_ms")
    if hrv is None:
        return 50.0, False
    # Sigmoid-like mapping: 20ms→~10, 55ms→50, 100ms→~90
    score = float(np.clip((hrv - 20) / 80 * 100, 0, 100))
    return score, True


def _score_stress_recovery(
    record: dict, prev_record: Optional[dict]
) -> tuple[float, bool]:
    """15% weight — previous-day stress recovery: did stress fall from yesterday?"""
    km = record.get("key_metrics", {})
    stress_today = km.get("stress")
    if stress_today is None or prev_record is None:
        return 50.0, False
    prev_km = prev_record.get("key_metrics", {})
    stress_prev = prev_km.get("stress")
    if stress_prev is None:
        return 50.0, False
    # Recovery = how much stress dropped (positive delta = good)
    delta = float(stress_prev) - float(stress_today)   # positive = improved
    # Scale: delta in [-1, 1] → score in [0, 100]
    score = float(np.clip(50.0 + delta * 100.0, 0, 100))
    return score, True


def _score_activity_rest(record: dict) -> tuple[float, bool]:
    """10% weight — meditation_minutes as proxy for activity/rest balance."""
    km  = record.get("key_metrics", {})
    med = km.get("meditation_minutes")
    if med is None:
        # Try the top-level key that may come from DailySummary.dict()
        med = record.get("meditation_minutes")
    if med is None:
        return 50.0, False
    # 20+ minutes per day = optimal; 0 = minimal; cap at 60
    score = float(np.clip(med / 30.0 * 100.0, 0, 100))
    return score, True


def _compute_composite_wellness(
    record: dict, prev_record: Optional[dict]
) -> tuple[float, List[WellnessComponent]]:
    """Return composite 0-100 score and per-component breakdown."""
    components: List[WellnessComponent] = []

    sleep_sub,    sleep_avail    = _score_sleep_quality(record)
    emotion_sub,  emotion_avail  = _score_emotional_baseline(record)
    hrv_sub,      hrv_avail      = _score_hrv(record)
    recovery_sub, recovery_avail = _score_stress_recovery(record, prev_record)
    activity_sub, activity_avail = _score_activity_rest(record)

    sub_scores = [
        ("Sleep quality",        sleep_sub,    _WEIGHTS["sleep_quality"],      sleep_avail),
        ("Emotional baseline",   emotion_sub,  _WEIGHTS["emotional_baseline"],  emotion_avail),
        ("HRV trend",            hrv_sub,      _WEIGHTS["hrv_trend"],           hrv_avail),
        ("Stress recovery",      recovery_sub, _WEIGHTS["stress_recovery"],     recovery_avail),
        ("Activity/rest balance",activity_sub, _WEIGHTS["activity_rest"],       activity_avail),
    ]

    composite = 0.0
    for name, sub, weight, avail in sub_scores:
        contribution = sub * weight
        composite   += contribution
        components.append(WellnessComponent(
            name=name,
            sub_score=round(sub, 1),
            weight=weight,
            weighted_contribution=round(contribution, 1),
            available=avail,
        ))

    return float(np.clip(composite, 0, 100)), components


def _color(score: float) -> str:
    if score >= 70:
        return "green"
    if score >= 40:
        return "yellow"
    return "red"


def _generate_insight(
    components: List[WellnessComponent],
    trend_7d: List[float],
    record: dict,
) -> str:
    """One-sentence actionable insight targeting the weakest component."""
    # Find lowest-scoring available component
    available = [c for c in components if c.available]
    if not available:
        return "Submit your daily metrics to receive a personalised wellness insight."

    weakest = min(available, key=lambda c: c.sub_score)

    # Sleep-specific insight: compare to weekly average
    if weakest.name == "Sleep quality" and len(trend_7d) >= 3:
        weekly_avg = float(np.mean(trend_7d[:-1])) if len(trend_7d) > 1 else trend_7d[0]
        today_score = weakest.sub_score
        diff = weekly_avg - today_score
        if diff > 5:
            return (
                f"Your sleep quality score is {diff:.0f} points below your weekly average "
                "— expect lower focus today; prioritise an early bedtime tonight."
            )
        return "Your sleep quality is below optimal — aim for 7-9 hours with minimal light exposure."

    if weakest.name == "Emotional baseline":
        km = record.get("key_metrics", {})
        stress = km.get("stress")
        if stress is not None and float(stress) > 0.6:
            return "Elevated morning stress detected — a 10-minute breathing exercise can lower cortisol before your day begins."
        valence = km.get("valence")
        if valence is not None and float(valence) < -0.2:
            return "Negative emotional tone detected — check sleep quality and hydration, and consider a short walk outside."
        return "Emotional baseline is lower than ideal — mindful journaling or light movement may help shift your mood."

    if weakest.name == "HRV trend":
        return (
            "Your HRV is below your target range — prioritise recovery: avoid intense exercise today and "
            "consider a short meditation session to activate the parasympathetic nervous system."
        )

    if weakest.name == "Stress recovery":
        return (
            "Yesterday's stress hasn't fully recovered — protect your energy today with scheduled breaks "
            "and limit high-demand tasks to your peak-focus window."
        )

    if weakest.name == "Activity/rest balance":
        return (
            "Low mindfulness activity detected — even 10 minutes of meditation or a short walk "
            "can significantly improve your afternoon focus and mood."
        )

    return "Keep monitoring your daily metrics to receive more targeted wellness insights."


def _trend_direction(scores: List[float]) -> str:
    if len(scores) < 3:
        return "stable"
    slope = float(np.polyfit(range(len(scores)), scores, 1)[0])
    if slope > 0.5:
        return "improving"
    if slope < -0.5:
        return "declining"
    return "stable"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/daily-score/{user_id}", response_model=DailyWellnessScore)
async def get_daily_wellness_score(user_id: str):
    """Return the composite daily wellness score with per-component breakdown and trend."""
    user_records = _daily_records.get(user_id)
    if not user_records:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No daily records found for user '{user_id}'. "
                "Submit metrics via POST /health-summary/daily first."
            ),
        )

    # Sort dates and pick the latest record
    sorted_dates = sorted(user_records.keys())
    latest_date  = sorted_dates[-1]
    record       = user_records[latest_date]
    prev_record  = user_records.get(sorted_dates[-2]) if len(sorted_dates) >= 2 else None

    # 7-day trend from the shared deque
    all_scores = list(_trend_history.get(user_id, []))
    trend_7d   = all_scores[-7:] if len(all_scores) >= 7 else all_scores

    # Compute composite score
    score, components = _compute_composite_wellness(record, prev_record)
    direction = _trend_direction(trend_7d)
    insight   = _generate_insight(components, trend_7d, record)

    return DailyWellnessScore(
        user_id=user_id,
        date=latest_date,
        score=round(score, 1),
        color=_color(score),
        insight=insight,
        trend_7d=[round(s, 1) for s in trend_7d],
        trend_direction=direction,
        components=components,
        disclaimer=_WELLNESS_DISCLAIMER,
    )
