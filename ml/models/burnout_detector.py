"""Predictive burnout detection model (issue #418).

Analyzes longitudinal emotional trajectory (weeks-to-months) to detect
the slow-moving signature of burnout onset. Computes a burnout risk score
(0-100) from multiple converging signals.

References:
- Maslach & Leiter (2016) — Understanding the burnout experience
- Sonnentag (2017) — A task-level perspective on work engagement
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class DailySnapshot:
    """One day's aggregated emotional data."""

    date: str                        # ISO date string
    mean_valence: float              # daily average (-1 to 1)
    max_valence: float               # daily high
    min_valence: float               # daily low
    mean_arousal: float              # daily average (0 to 1)
    arousal_variance: float          # within-day arousal variance
    mean_stress: float               # daily average (0 to 1)
    mean_focus: float                # daily average (0 to 1)
    sleep_quality: Optional[float] = None   # 0–100 if available
    check_in_count: int = 0          # number of readings that day
    is_weekend: bool = False


@dataclass
class BurnoutSignal:
    """A single burnout trajectory signal with its current value and trend."""

    name: str
    description: str
    value: float            # current value (0-1 normalized)
    trend_slope: float      # per-week slope (negative = worsening)
    weeks_trending: int     # how many weeks the trend has persisted
    severity: str           # "normal" | "watch" | "warning" | "critical"


@dataclass
class BurnoutAssessment:
    """Full burnout risk assessment."""

    risk_score: int                          # 0–100
    risk_level: str                          # "green" | "yellow" | "orange" | "red"
    signals: List[BurnoutSignal]
    warning_signals_count: int               # signals at "warning" or "critical"
    primary_concern: str                     # most significant signal
    recommended_actions: List[str]
    data_weeks: int                          # weeks of data analyzed
    confidence: float                        # 0–1 based on data sufficiency


def _compute_rolling_stats(
    values: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rolling mean and std with given window size."""
    if len(values) < window:
        return np.array([np.mean(values)]), np.array([np.std(values)])

    means = []
    stds = []
    for i in range(len(values) - window + 1):
        chunk = values[i:i + window]
        means.append(np.mean(chunk))
        stds.append(np.std(chunk))
    return np.array(means), np.array(stds)


def _compute_trend_slope(values: np.ndarray) -> float:
    """Compute linear trend slope using OLS."""
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    x_mean = np.mean(x)
    y_mean = np.mean(values)
    ss_xy = np.sum((x - x_mean) * (values - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx < 1e-10:
        return 0.0
    return float(ss_xy / ss_xx)


def _classify_signal_severity(
    value: float,
    trend_slope: float,
    weeks: int,
    threshold_watch: float = -0.02,
    threshold_warning: float = -0.04,
) -> str:
    """Classify signal severity based on value and trend."""
    if weeks >= 4 and trend_slope < threshold_warning:
        return "critical"
    if weeks >= 3 and trend_slope < threshold_watch:
        return "warning"
    if weeks >= 2 and trend_slope < 0:
        return "watch"
    return "normal"


def analyze_burnout_trajectory(
    daily_snapshots: List[DailySnapshot],
) -> BurnoutAssessment:
    """Analyze emotional trajectory for burnout indicators.

    Parameters
    ----------
    daily_snapshots : list of DailySnapshot, ordered chronologically

    Returns
    -------
    BurnoutAssessment with risk score, signals, and recommendations.
    """
    n_days = len(daily_snapshots)
    n_weeks = n_days // 7

    if n_days < 7:
        return BurnoutAssessment(
            risk_score=0,
            risk_level="green",
            signals=[],
            warning_signals_count=0,
            primary_concern="insufficient data",
            recommended_actions=["Continue tracking for at least 2 weeks"],
            data_weeks=0,
            confidence=0.0,
        )

    # Extract daily time series
    valences = np.array([s.mean_valence for s in daily_snapshots])
    arousals = np.array([s.mean_arousal for s in daily_snapshots])
    stresses = np.array([s.mean_stress for s in daily_snapshots])
    focuses = np.array([s.mean_focus for s in daily_snapshots])
    ranges = np.array([s.max_valence - s.min_valence for s in daily_snapshots])
    arousal_vars = np.array([s.arousal_variance for s in daily_snapshots])
    check_ins = np.array([s.check_in_count for s in daily_snapshots])

    signals: List[BurnoutSignal] = []

    # ── Signal 1: Emotional range compression ──
    range_slope = _compute_trend_slope(ranges)
    range_weeks = max(1, n_weeks)
    signals.append(BurnoutSignal(
        name="emotional_range",
        description="Gap between daily emotional highs and lows",
        value=round(float(np.mean(ranges[-7:])), 3),
        trend_slope=round(range_slope, 5),
        weeks_trending=range_weeks if range_slope < 0 else 0,
        severity=_classify_signal_severity(float(np.mean(ranges[-7:])), range_slope, range_weeks),
    ))

    # ── Signal 2: Valence drift (slow decline) ──
    valence_slope = _compute_trend_slope(valences)
    signals.append(BurnoutSignal(
        name="valence_drift",
        description="Trend in overall emotional positivity",
        value=round(float(np.mean(valences[-7:])), 3),
        trend_slope=round(valence_slope, 5),
        weeks_trending=max(1, n_weeks) if valence_slope < -0.01 else 0,
        severity=_classify_signal_severity(float(np.mean(valences[-7:])), valence_slope, n_weeks),
    ))

    # ── Signal 3: Arousal flattening ──
    arousal_var_slope = _compute_trend_slope(arousal_vars)
    signals.append(BurnoutSignal(
        name="arousal_flattening",
        description="Declining variability in energy/arousal levels",
        value=round(float(np.mean(arousal_vars[-7:])), 3),
        trend_slope=round(arousal_var_slope, 5),
        weeks_trending=max(1, n_weeks) if arousal_var_slope < 0 else 0,
        severity=_classify_signal_severity(
            float(np.mean(arousal_vars[-7:])),
            arousal_var_slope,
            n_weeks,
        ),
    ))

    # ── Signal 4: Recovery failure (weekend vs weekday) ──
    weekday_vals = [s.mean_valence for s in daily_snapshots if not s.is_weekend]
    weekend_vals = [s.mean_valence for s in daily_snapshots if s.is_weekend]
    if weekday_vals and weekend_vals:
        recovery = float(np.mean(weekend_vals)) - float(np.mean(weekday_vals))
        # Normal: weekends higher valence. Burnout: no difference or weekends worse
        recovery_signal = max(0.0, min(1.0, recovery + 0.5))  # normalize
        signals.append(BurnoutSignal(
            name="recovery_failure",
            description="Weekend emotional recovery vs weekday baseline",
            value=round(recovery, 3),
            trend_slope=0.0,  # single value, not trended
            weeks_trending=0 if recovery > 0.05 else max(1, n_weeks),
            severity="warning" if recovery < -0.05 else ("watch" if recovery < 0.05 else "normal"),
        ))

    # ── Signal 5: Engagement decay (check-in frequency) ──
    if np.sum(check_ins) > 0:
        checkin_slope = _compute_trend_slope(check_ins)
        signals.append(BurnoutSignal(
            name="engagement_decay",
            description="Declining app check-in frequency",
            value=round(float(np.mean(check_ins[-7:])), 1),
            trend_slope=round(checkin_slope, 4),
            weeks_trending=max(1, n_weeks) if checkin_slope < 0 else 0,
            severity=_classify_signal_severity(
                float(np.mean(check_ins[-7:])),
                checkin_slope * 0.1,  # scale down since it's count-based
                n_weeks,
            ),
        ))

    # ── Signal 6: Sleep-mood decoupling ──
    sleep_data = [(s.sleep_quality, s.mean_valence) for s in daily_snapshots if s.sleep_quality is not None]
    if len(sleep_data) >= 7:
        sleep_arr = np.array([s[0] for s in sleep_data])
        mood_arr = np.array([s[1] for s in sleep_data])
        # Normalize
        if np.std(sleep_arr) > 0 and np.std(mood_arr) > 0:
            correlation = float(np.corrcoef(sleep_arr, mood_arr)[0, 1])
        else:
            correlation = 0.0
        # Normal: r ~ 0.3-0.4. Burnout: r drops toward 0
        signals.append(BurnoutSignal(
            name="sleep_mood_decoupling",
            description="Sleep quality stops predicting mood (normally r~0.3)",
            value=round(correlation, 3),
            trend_slope=0.0,
            weeks_trending=0 if correlation > 0.15 else max(1, n_weeks),
            severity="warning" if correlation < 0.05 else ("watch" if correlation < 0.15 else "normal"),
        ))

    # ── Compute composite risk score ──
    signal_weights = {
        "emotional_range": 0.20,
        "valence_drift": 0.25,
        "arousal_flattening": 0.15,
        "recovery_failure": 0.15,
        "engagement_decay": 0.10,
        "sleep_mood_decoupling": 0.15,
    }

    risk_points = 0.0
    total_weight = 0.0
    for sig in signals:
        w = signal_weights.get(sig.name, 0.1)
        severity_score = {"normal": 0, "watch": 25, "warning": 60, "critical": 100}
        risk_points += w * severity_score.get(sig.severity, 0)
        total_weight += w

    risk_score = int(risk_points / total_weight) if total_weight > 0 else 0
    risk_score = max(0, min(100, risk_score))

    # Risk level
    if risk_score < 30:
        risk_level = "green"
    elif risk_score < 60:
        risk_level = "yellow"
    elif risk_score < 80:
        risk_level = "orange"
    else:
        risk_level = "red"

    # Count warning+ signals
    warning_count = sum(1 for s in signals if s.severity in ("warning", "critical"))

    # Primary concern
    severity_rank = {"critical": 4, "warning": 3, "watch": 2, "normal": 1}
    worst_signal = max(signals, key=lambda s: severity_rank.get(s.severity, 0)) if signals else None
    primary_concern = worst_signal.description if worst_signal and worst_signal.severity != "normal" else "no significant concerns"

    # Recommendations
    actions = []
    if risk_level == "green":
        actions.append("Continue your current patterns — your emotional trajectory looks healthy")
    if risk_level in ("yellow", "orange", "red"):
        actions.append("Consider a workload audit — identify tasks that can be deferred or delegated")
    if any(s.name == "recovery_failure" and s.severity in ("warning", "critical") for s in signals):
        actions.append("Protect your recovery time — schedule restorative activities on weekends")
    if any(s.name == "sleep_mood_decoupling" and s.severity in ("warning", "critical") for s in signals):
        actions.append("Sleep quality is not translating to mood improvement — consider professional support")
    if risk_level in ("orange", "red"):
        actions.append("Your emotional trajectory matches burnout patterns — speak with a professional if possible")

    if not actions:
        actions.append("Keep tracking — more data improves prediction accuracy")

    # Confidence based on data amount
    confidence = min(1.0, n_weeks / 8.0)  # full confidence at 8 weeks

    return BurnoutAssessment(
        risk_score=risk_score,
        risk_level=risk_level,
        signals=signals,
        warning_signals_count=warning_count,
        primary_concern=primary_concern,
        recommended_actions=actions,
        data_weeks=n_weeks,
        confidence=round(confidence, 2),
    )


def assessment_to_dict(assessment: BurnoutAssessment) -> Dict[str, Any]:
    """Serialize BurnoutAssessment to JSON-safe dict."""
    return {
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "signals": [
            {
                "name": s.name,
                "description": s.description,
                "value": s.value,
                "trend_slope": s.trend_slope,
                "weeks_trending": s.weeks_trending,
                "severity": s.severity,
            }
            for s in assessment.signals
        ],
        "warning_signals_count": assessment.warning_signals_count,
        "primary_concern": assessment.primary_concern,
        "recommended_actions": assessment.recommended_actions,
        "data_weeks": assessment.data_weeks,
        "confidence": assessment.confidence,
    }
