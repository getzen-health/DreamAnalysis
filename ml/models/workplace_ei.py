"""Workplace emotional intelligence tracker.

Tracks emotional patterns in work contexts:
- Meeting emotional climate (pre/post voice check-ins)
- Burnout risk detection from progressive HRV/voice/sleep degradation
- Productivity-emotion correlation
- Work-life emotional boundary tracking

Based on:
- Baird et al. (2024): Voice burnout markers d=0.62 (pitch variability, speech rate, pauses)
- Oswald et al. (2015): Positive affect → +12% productivity
- HRV burnout prediction 2-4 weeks early from RMSSD decline (Golkar et al. 2014)
"""
from __future__ import annotations

import json
import math
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_DIR = Path("data/workplace_ei")


def _user_path(user_id: str) -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR / f"{user_id}.json"


def _load(user_id: str) -> Dict:
    p = _user_path(user_id)
    if p.exists():
        return json.loads(p.read_text())
    return {
        "user_id": user_id,
        "meetings": [],
        "daily_snapshots": [],
        "burnout_risk": 0.0,
        "hrv_baseline": None,
        "session_count": 0,
    }


def _save(user_id: str, data: Dict) -> None:
    _user_path(user_id).write_text(json.dumps(data, indent=2, default=str))


# ── Burnout scoring ────────────────────────────────────────────────────────────

def _compute_burnout_risk(snapshots: List[Dict]) -> float:
    """Estimate burnout risk 0-1 from recent daily snapshots.

    Markers (Baird 2024, Golkar 2014):
    - Progressive HRV decline over last 14 days
    - Increasing average stress_index
    - Decreasing average focus_index
    - Reduced positive valence trend
    """
    if len(snapshots) < 7:
        return 0.0  # not enough data

    recent = snapshots[-14:]

    # HRV trend (decline = risk)
    hrv_values = [s.get("hrv_rmssd") for s in recent if s.get("hrv_rmssd") is not None]
    hrv_score = 0.0
    if len(hrv_values) >= 4:
        first_half_avg = sum(hrv_values[: len(hrv_values) // 2]) / (len(hrv_values) // 2)
        second_half_avg = sum(hrv_values[len(hrv_values) // 2 :]) / (len(hrv_values) // 2)
        if first_half_avg > 0:
            decline_pct = (first_half_avg - second_half_avg) / first_half_avg
            hrv_score = max(0.0, min(1.0, decline_pct * 3.0))  # 33% decline → 1.0

    # Stress trend (increase = risk)
    stress_values = [s.get("stress_index", 0.5) for s in recent]
    avg_stress = sum(stress_values) / len(stress_values)
    stress_score = max(0.0, (avg_stress - 0.40) / 0.60)  # 0.4→0 baseline, 1.0→1.0

    # Focus trend (decrease = risk)
    focus_values = [s.get("focus_index", 0.5) for s in recent]
    avg_focus = sum(focus_values) / len(focus_values)
    focus_score = max(0.0, (0.60 - avg_focus) / 0.60)  # 0.6→0 baseline, 0→1.0

    # Valence trend (decrease = risk)
    valence_values = [s.get("valence", 0.0) for s in recent]
    avg_valence = sum(valence_values) / len(valence_values)
    valence_score = max(0.0, (-avg_valence + 0.3) / 1.3)  # +0.3→0, -1→1.0

    # Weighted burnout risk
    risk = (
        0.35 * hrv_score
        + 0.25 * stress_score
        + 0.20 * focus_score
        + 0.20 * valence_score
    )
    return round(min(1.0, max(0.0, risk)), 3)


def _burnout_level(risk: float) -> str:
    if risk < 0.25:
        return "low"
    if risk < 0.50:
        return "moderate"
    if risk < 0.75:
        return "high"
    return "critical"


# ── Meeting climate ───────────────────────────────────────────────────────────

def _meeting_climate(pre: Dict, post: Dict) -> Dict:
    """Compute meeting emotional impact from pre/post check-ins."""
    valence_delta = post.get("valence", 0.0) - pre.get("valence", 0.0)
    stress_delta = post.get("stress_index", 0.5) - pre.get("stress_index", 0.5)
    energy_delta = post.get("focus_index", 0.5) - pre.get("focus_index", 0.5)

    if valence_delta > 0.15 and stress_delta < 0:
        climate = "energizing"
    elif valence_delta < -0.15 or stress_delta > 0.20:
        climate = "draining"
    elif abs(valence_delta) < 0.10 and abs(stress_delta) < 0.10:
        climate = "neutral"
    else:
        climate = "mixed"

    return {
        "climate": climate,
        "valence_delta": round(valence_delta, 3),
        "stress_delta": round(stress_delta, 3),
        "energy_delta": round(energy_delta, 3),
        "summary": _climate_summary(climate, valence_delta, stress_delta),
    }


def _climate_summary(climate: str, v_delta: float, s_delta: float) -> str:
    if climate == "energizing":
        return "This meeting was energizing — mood improved and stress decreased."
    if climate == "draining":
        adj = "significantly" if abs(v_delta) > 0.30 or s_delta > 0.30 else "somewhat"
        return f"This meeting was {adj} draining — consider its frequency or format."
    if climate == "neutral":
        return "Neutral meeting impact — neither energizing nor draining."
    return "Mixed emotional impact from this meeting."


# ── Productivity correlation ──────────────────────────────────────────────────

def _productivity_insight(snapshots: List[Dict]) -> str:
    if len(snapshots) < 5:
        return "Not enough data yet — log more sessions to see productivity-emotion patterns."

    high_focus = [s for s in snapshots if s.get("focus_index", 0) > 0.65]
    high_valence = [s for s in snapshots if s.get("valence", 0) > 0.2]

    if not snapshots:
        return "No data available."

    pct_focus = len(high_focus) / len(snapshots) * 100
    pct_positive = len(high_valence) / len(snapshots) * 100

    if pct_focus > 60 and pct_positive > 50:
        return (
            f"Strong productivity conditions: {pct_focus:.0f}% of sessions at high focus, "
            f"{pct_positive:.0f}% in positive mood. Research: +12% productivity when affect is positive."
        )
    if pct_focus < 35:
        return (
            f"Focus is low in {100 - pct_focus:.0f}% of sessions. "
            "Common causes: poor sleep, high stress, meeting overload."
        )
    return (
        f"Moderate conditions: {pct_focus:.0f}% high-focus sessions. "
        "Aim for 60%+ to maximize cognitive performance."
    )


# ── Public API ────────────────────────────────────────────────────────────────

class WorkplaceEITracker:
    """Track workplace emotional intelligence patterns."""

    def log_daily(self, user_id: str, snapshot: Dict) -> Dict:
        """Log a daily emotion/biometric snapshot.

        Args:
            user_id: User identifier.
            snapshot: Dict with keys: valence, arousal, stress_index, focus_index,
                relaxation_index, hrv_rmssd (optional), date (optional ISO string).

        Returns:
            Dict with burnout_risk, burnout_level, streak, insight.
        """
        data = _load(user_id)
        record = {
            "date": snapshot.get("date", date.today().isoformat()),
            "valence": float(snapshot.get("valence", 0.0)),
            "arousal": float(snapshot.get("arousal", 0.5)),
            "stress_index": float(snapshot.get("stress_index", 0.5)),
            "focus_index": float(snapshot.get("focus_index", 0.5)),
            "relaxation_index": float(snapshot.get("relaxation_index", 0.5)),
            "hrv_rmssd": snapshot.get("hrv_rmssd"),
            "work_hours": snapshot.get("work_hours"),
        }
        data["daily_snapshots"].append(record)
        data["session_count"] = data.get("session_count", 0) + 1

        # Update HRV baseline (rolling 30-day mean)
        hrv_vals = [
            s["hrv_rmssd"]
            for s in data["daily_snapshots"][-30:]
            if s.get("hrv_rmssd") is not None
        ]
        if hrv_vals:
            data["hrv_baseline"] = sum(hrv_vals) / len(hrv_vals)

        risk = _compute_burnout_risk(data["daily_snapshots"])
        data["burnout_risk"] = risk

        _save(user_id, data)
        return {
            "logged": True,
            "session_count": data["session_count"],
            "burnout_risk": risk,
            "burnout_level": _burnout_level(risk),
            "productivity_insight": _productivity_insight(data["daily_snapshots"]),
        }

    def log_meeting(
        self,
        user_id: str,
        meeting_name: str,
        pre_state: Dict,
        post_state: Dict,
    ) -> Dict:
        """Log a meeting with pre/post emotional check-ins.

        Args:
            user_id: User identifier.
            meeting_name: Short name/title of the meeting.
            pre_state: Emotion snapshot before meeting.
            post_state: Emotion snapshot after meeting.

        Returns:
            Meeting climate analysis dict.
        """
        data = _load(user_id)
        climate = _meeting_climate(pre_state, post_state)
        record = {
            "date": date.today().isoformat(),
            "meeting_name": meeting_name,
            **climate,
        }
        data["meetings"].append(record)
        _save(user_id, data)
        return {
            "meeting": meeting_name,
            **climate,
            "total_meetings_tracked": len(data["meetings"]),
        }

    def get_report(self, user_id: str) -> Dict:
        """Return full workplace EI report.

        Returns:
            Dict with burnout_risk, burnout_level, meeting_stats, weekly_pattern,
            productivity_insight, recommendations.
        """
        data = _load(user_id)
        snapshots = data.get("daily_snapshots", [])
        meetings = data.get("meetings", [])

        risk = _compute_burnout_risk(snapshots)
        level = _burnout_level(risk)

        # Meeting climate breakdown
        climate_counts: Dict[str, int] = {}
        for m in meetings:
            c = m.get("climate", "unknown")
            climate_counts[c] = climate_counts.get(c, 0) + 1

        # Weekly pattern (last 7 days)
        weekly = snapshots[-7:] if len(snapshots) >= 7 else snapshots
        avg_stress = sum(s.get("stress_index", 0.5) for s in weekly) / max(1, len(weekly))
        avg_focus = sum(s.get("focus_index", 0.5) for s in weekly) / max(1, len(weekly))
        avg_valence = sum(s.get("valence", 0.0) for s in weekly) / max(1, len(weekly))

        recommendations = _recommendations(risk, avg_stress, avg_focus, len(meetings), climate_counts)

        return {
            "user_id": user_id,
            "sessions_logged": len(snapshots),
            "burnout_risk": risk,
            "burnout_level": level,
            "hrv_baseline_ms": data.get("hrv_baseline"),
            "weekly_average": {
                "stress_index": round(avg_stress, 3),
                "focus_index": round(avg_focus, 3),
                "valence": round(avg_valence, 3),
            },
            "meeting_stats": {
                "total": len(meetings),
                "climate_breakdown": climate_counts,
                "draining_pct": round(
                    climate_counts.get("draining", 0) / max(1, len(meetings)) * 100, 1
                ),
            },
            "productivity_insight": _productivity_insight(snapshots),
            "recommendations": recommendations,
        }

    def get_meeting_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Return recent meeting climate history."""
        data = _load(user_id)
        return data.get("meetings", [])[-limit:]

    def reset(self, user_id: str) -> Dict:
        """Clear all data for a user."""
        p = _user_path(user_id)
        if p.exists():
            p.unlink()
        return {"reset": True, "user_id": user_id}


def _recommendations(
    risk: float,
    avg_stress: float,
    avg_focus: float,
    n_meetings: int,
    climate_counts: Dict[str, int],
) -> List[str]:
    recs = []
    if risk > 0.60:
        recs.append(
            "Burnout risk is HIGH. Consider: 1-2 day offline break, reduce meeting load, "
            "prioritize sleep (HRV recovery depends on it)."
        )
    elif risk > 0.35:
        recs.append(
            "Moderate burnout risk. Monitor for declining HRV trend. "
            "Protect at least 2h of deep work time daily."
        )

    if avg_stress > 0.65:
        recs.append(
            "Elevated sustained stress. Evidence: HRV-based breathing (coherence training) "
            "reduces cortisol 15-25% within 4 weeks (Lehrer 2003)."
        )

    if avg_focus < 0.40:
        recs.append(
            "Chronically low focus. Common causes: poor sleep, high cortisol, "
            "notification overload. Try: 90-min deep work blocks before 12pm."
        )

    draining = climate_counts.get("draining", 0)
    if n_meetings > 0 and draining / n_meetings > 0.50:
        recs.append(
            f"{draining}/{n_meetings} recent meetings were draining. "
            "Consider: async alternatives, meeting-free afternoons, standing meeting policy."
        )

    if not recs:
        recs.append(
            "Workplace emotional state looks healthy. "
            "Maintaining positive affect → +12% productivity (Oswald 2015)."
        )
    return recs


# Module-level singleton
_tracker: Optional[WorkplaceEITracker] = None


def get_tracker() -> WorkplaceEITracker:
    global _tracker
    if _tracker is None:
        _tracker = WorkplaceEITracker()
    return _tracker
