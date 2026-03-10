"""Longitudinal EI (Emotional Intelligence) growth tracker.

Tracks daily EI composite scores over time and computes growth trajectory.
Science: Bar-On EQ-i 2.0 shows EI is trainable (d=0.46, Mattingly & Kraiger 2019).
Measurable changes in 4-8 weeks of consistent practice.

Key dimensions tracked:
- Self-perception (self-awareness, most changeable d=0.55)
- Self-expression (assertiveness, independence)
- Interpersonal (empathy, social responsibility, slowest d=0.28)
- Decision-making (problem-solving, impulse control)
- Stress management (flexibility, optimism)
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Changeability ratings from Mattingly & Kraiger (2019) meta-analysis d-values
_DIMENSION_CHANGEABILITY: Dict[str, Dict[str, str]] = {
    "self_perception": {
        "label": "Self-Perception",
        "changeability": "fastest",
        "d_value": 0.55,
        "description": "Self-awareness, self-regard, self-actualization",
    },
    "self_expression": {
        "label": "Self-Expression",
        "changeability": "fast",
        "d_value": 0.50,
        "description": "Assertiveness, independence, emotional expression",
    },
    "decision_making": {
        "label": "Decision-Making",
        "changeability": "moderate",
        "d_value": 0.45,
        "description": "Problem-solving, reality-testing, impulse control",
    },
    "stress_management": {
        "label": "Stress Management",
        "changeability": "moderate",
        "d_value": 0.42,
        "description": "Flexibility, stress tolerance, optimism",
    },
    "interpersonal": {
        "label": "Interpersonal",
        "changeability": "slowest",
        "d_value": 0.28,
        "description": "Empathy, social responsibility, interpersonal relationships",
    },
}

_MAX_SNAPSHOTS_PER_USER = 365


@dataclass
class DailyEISnapshot:
    date: str                          # YYYY-MM-DD
    eiq_score: float                   # 0-100 overall EI composite
    dimension_scores: Dict[str, float] # 5 dimensions, each 0-100
    data_sources: List[str]            # e.g. ["voice", "eeg", "health"]
    confidence: float                  # 0-1
    timestamp: float                   # Unix epoch


class EIGrowthTracker:
    """Tracks daily EI composite scores and computes longitudinal growth trajectory."""

    def __init__(self) -> None:
        # user_id -> list of DailyEISnapshot (oldest first)
        self._store: Dict[str, List[DailyEISnapshot]] = defaultdict(list)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_snapshot(
        self,
        user_id: str,
        eiq_score: float,
        dimension_scores: Dict[str, float],
        data_sources: List[str],
        confidence: float,
    ) -> DailyEISnapshot:
        """Add a daily EI reading for user_id.

        If a snapshot for today already exists it is replaced (one per day).
        """
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        snapshot = DailyEISnapshot(
            date=today,
            eiq_score=float(eiq_score),
            dimension_scores={k: float(v) for k, v in dimension_scores.items()},
            data_sources=list(data_sources),
            confidence=float(confidence),
            timestamp=time.time(),
        )

        user_data = self._store[user_id]

        # Replace existing entry for today if present
        for i, s in enumerate(user_data):
            if s.date == today:
                user_data[i] = snapshot
                return snapshot

        user_data.append(snapshot)

        # Enforce cap: drop oldest if over limit
        if len(user_data) > _MAX_SNAPSHOTS_PER_USER:
            self._store[user_id] = user_data[-_MAX_SNAPSHOTS_PER_USER:]

        return snapshot

    def get_weekly_trend(self, user_id: str) -> Dict:
        """Compute weekly averages + Theil-Sen slope over the last 4 weeks.

        Returns dict with:
          weekly_averages   List[float]  — average EIQ per week (oldest → newest)
          slope_per_week    float        — Theil-Sen median slope (EIQ pts / week)
          trend             str          — "improving" | "declining" | "stable"
          p_significant     bool         — True if |slope| >= 0.5 pts/week
          weeks_of_data     int          — how many weeks contributed data
        """
        snapshots = self._store.get(user_id, [])
        if len(snapshots) < 2:
            return {
                "weekly_averages": [],
                "slope_per_week": 0.0,
                "trend": "stable",
                "p_significant": False,
                "weeks_of_data": 0,
            }

        # Group into ISO weeks (most recent 4 weeks)
        weekly: Dict[str, List[float]] = defaultdict(list)
        for s in snapshots:
            try:
                dt = datetime.strptime(s.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                week_key = dt.strftime("%Y-W%W")
                weekly[week_key].append(s.eiq_score)
            except ValueError:
                continue

        sorted_weeks = sorted(weekly.keys())[-4:]
        weekly_averages = [
            sum(weekly[w]) / len(weekly[w]) for w in sorted_weeks
        ]
        weeks_of_data = len(sorted_weeks)

        if weeks_of_data < 2:
            return {
                "weekly_averages": weekly_averages,
                "slope_per_week": 0.0,
                "trend": "stable",
                "p_significant": False,
                "weeks_of_data": weeks_of_data,
            }

        slope = _theil_sen_slope(weekly_averages)
        trend = (
            "improving" if slope > 0.3
            else "declining" if slope < -0.3
            else "stable"
        )
        p_significant = abs(slope) >= 0.5

        return {
            "weekly_averages": [round(v, 2) for v in weekly_averages],
            "slope_per_week": round(slope, 4),
            "trend": trend,
            "p_significant": p_significant,
            "weeks_of_data": weeks_of_data,
        }

    def get_milestones(self, user_id: str) -> List[Dict]:
        """Return achieved milestones for the user."""
        snapshots = self._store.get(user_id, [])
        milestones: List[Dict] = []

        if not snapshots:
            return milestones

        dates = [s.date for s in snapshots]
        first_date = dates[0]

        # Milestone 1: Baseline established (7+ days of data)
        if len(snapshots) >= 7:
            milestones.append({
                "type": "baseline_established",
                "label": "Baseline Established",
                "achieved_at": dates[6],
                "description": "7 days of EI data collected — your personal baseline is ready.",
            })

        # Milestone 2: 30-day mark
        if len(snapshots) >= 30:
            milestones.append({
                "type": "thirty_day_mark",
                "label": "30-Day Journey",
                "achieved_at": dates[29],
                "description": "30 consecutive days of EI tracking. Measurable changes typically emerge now.",
            })

        # Milestone 3: First improvement (2+ weeks with positive slope)
        if len(snapshots) >= 14:
            recent_scores = [s.eiq_score for s in snapshots[-14:]]
            slope_2w = _theil_sen_slope(recent_scores)
            if slope_2w > 0.3:
                milestones.append({
                    "type": "first_improvement",
                    "label": "First Improvement",
                    "achieved_at": dates[-1],
                    "description": "Your EI score is trending upward over the past 2 weeks.",
                })

        # Milestone 4: Dimension breakthrough (+5 pts in any dimension over 2 weeks)
        if len(snapshots) >= 14:
            early = snapshots[-14]
            latest = snapshots[-1]
            for dim in _DIMENSION_CHANGEABILITY:
                early_score = early.dimension_scores.get(dim)
                latest_score = latest.dimension_scores.get(dim)
                if (
                    early_score is not None
                    and latest_score is not None
                    and latest_score - early_score >= 5.0
                ):
                    dim_label = _DIMENSION_CHANGEABILITY[dim]["label"]
                    milestones.append({
                        "type": f"dimension_breakthrough_{dim}",
                        "label": f"{dim_label} Breakthrough",
                        "achieved_at": latest.date,
                        "description": (
                            f"{dim_label} improved by "
                            f"{latest_score - early_score:.1f} points over 2 weeks."
                        ),
                    })

        return milestones

    def predict_trajectory(self, user_id: str, target_eiq: float) -> Dict:
        """Predict weeks to reach target_eiq given current slope.

        Returns dict with:
          predicted_weeks     int | None
          confidence          float (0-1)
          current_eiq         float
          target_eiq          float
          trajectory_points   List[float]
        """
        snapshots = self._store.get(user_id, [])
        if not snapshots:
            return {
                "predicted_weeks": None,
                "confidence": 0.0,
                "current_eiq": 0.0,
                "target_eiq": float(target_eiq),
                "trajectory_points": [],
            }

        current_eiq = snapshots[-1].eiq_score
        trend = self.get_weekly_trend(user_id)
        slope = trend["slope_per_week"]
        weeks_of_data = trend["weeks_of_data"]

        # Confidence scales with data quality
        base_conf = min(0.9, weeks_of_data / 4.0 * 0.6 + 0.1)

        if slope <= 0 or current_eiq >= target_eiq:
            predicted_weeks = None if slope <= 0 else 0
            trajectory_points: List[float] = [
                min(100.0, current_eiq + slope * w) for w in range(13)
            ]
            return {
                "predicted_weeks": predicted_weeks,
                "confidence": round(base_conf, 2),
                "current_eiq": round(current_eiq, 2),
                "target_eiq": float(target_eiq),
                "trajectory_points": [round(v, 2) for v in trajectory_points],
            }

        weeks_needed = int((target_eiq - current_eiq) / slope) + 1
        # Project 12-week trajectory
        trajectory_points = [
            min(100.0, current_eiq + slope * w) for w in range(min(weeks_needed + 4, 52))
        ]

        return {
            "predicted_weeks": weeks_needed,
            "confidence": round(base_conf, 2),
            "current_eiq": round(current_eiq, 2),
            "target_eiq": float(target_eiq),
            "trajectory_points": [round(v, 2) for v in trajectory_points],
        }

    def get_dimension_report(self, user_id: str) -> Dict:
        """Return per-dimension 4-week slope + changeability rating.

        Returns dict with key 'dimensions', each entry containing:
          label, current_score, slope_per_week, trend, changeability, d_value, description
        """
        snapshots = self._store.get(user_id, [])
        report: Dict[str, Dict] = {}

        for dim, meta in _DIMENSION_CHANGEABILITY.items():
            # Collect weekly dim averages (last 4 weeks)
            if not snapshots:
                report[dim] = {
                    "label": meta["label"],
                    "current_score": None,
                    "slope_per_week": 0.0,
                    "trend": "stable",
                    "changeability": meta["changeability"],
                    "d_value": meta["d_value"],
                    "description": meta["description"],
                }
                continue

            weekly: Dict[str, List[float]] = defaultdict(list)
            for s in snapshots:
                score = s.dimension_scores.get(dim)
                if score is None:
                    continue
                try:
                    dt = datetime.strptime(s.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    week_key = dt.strftime("%Y-W%W")
                    weekly[week_key].append(score)
                except ValueError:
                    continue

            sorted_weeks = sorted(weekly.keys())[-4:]
            if not sorted_weeks:
                slope = 0.0
                current: Optional[float] = None
            else:
                weekly_avgs = [sum(weekly[w]) / len(weekly[w]) for w in sorted_weeks]
                slope = _theil_sen_slope(weekly_avgs) if len(weekly_avgs) >= 2 else 0.0
                current = weekly_avgs[-1]

            current_score = snapshots[-1].dimension_scores.get(dim) if snapshots else None

            trend = (
                "improving" if slope > 0.3
                else "declining" if slope < -0.3
                else "stable"
            )

            report[dim] = {
                "label": meta["label"],
                "current_score": round(current_score, 2) if current_score is not None else None,
                "slope_per_week": round(slope, 4),
                "trend": trend,
                "changeability": meta["changeability"],
                "d_value": meta["d_value"],
                "description": meta["description"],
            }

        return {"dimensions": report}

    def get_history(self, user_id: str, last_n: int = 30) -> List[DailyEISnapshot]:
        """Return the last N snapshots for user_id (most recent last)."""
        snapshots = self._store.get(user_id, [])
        return snapshots[-last_n:]


# ── Math helpers ─────────────────────────────────────────────────────────────

def _theil_sen_slope(values: List[float]) -> float:
    """Compute Theil-Sen slope (median of all pairwise slopes).

    Pure Python — no scipy dependency.
    Returns 0.0 for fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return 0.0

    slopes: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = j - i
            if dx != 0:
                slopes.append((values[j] - values[i]) / dx)

    if not slopes:
        return 0.0

    slopes.sort()
    mid = len(slopes) // 2
    if len(slopes) % 2 == 1:
        return slopes[mid]
    return (slopes[mid - 1] + slopes[mid]) / 2.0


# ── Module-level singleton ────────────────────────────────────────────────────

_tracker: Optional[EIGrowthTracker] = None


def get_ei_growth_tracker() -> EIGrowthTracker:
    global _tracker
    if _tracker is None:
        _tracker = EIGrowthTracker()
    return _tracker
