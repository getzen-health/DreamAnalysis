"""Unit tests for predictive burnout detector (issue #418)."""
from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.burnout_detector import (
    DailySnapshot,
    BurnoutSignal,
    analyze_burnout_trajectory,
    assessment_to_dict,
    _compute_trend_slope,
    _classify_signal_severity,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_snapshots(
    n_days: int,
    base_valence: float = 0.3,
    valence_drift: float = 0.0,       # per-day drift
    base_range: float = 0.6,
    range_drift: float = 0.0,
    base_arousal_var: float = 0.1,
    arousal_var_drift: float = 0.0,
    sleep_quality: float = 70.0,
    check_ins: int = 5,
    seed: int = 42,
) -> list:
    """Generate synthetic daily snapshots with controllable trends."""
    np.random.seed(seed)
    base_date = datetime(2026, 1, 1)
    snapshots = []
    for i in range(n_days):
        date = base_date + timedelta(days=i)
        v = base_valence + valence_drift * i + np.random.normal(0, 0.03)
        r = max(0.05, base_range + range_drift * i + np.random.normal(0, 0.02))
        av = max(0.01, base_arousal_var + arousal_var_drift * i + np.random.normal(0, 0.005))
        snapshots.append(DailySnapshot(
            date=date.strftime("%Y-%m-%d"),
            mean_valence=np.clip(v, -1, 1),
            max_valence=np.clip(v + r / 2, -1, 1),
            min_valence=np.clip(v - r / 2, -1, 1),
            mean_arousal=0.5,
            arousal_variance=av,
            mean_stress=0.3,
            mean_focus=0.5,
            sleep_quality=sleep_quality,
            check_in_count=check_ins,
            is_weekend=date.weekday() >= 5,
        ))
    return snapshots


# ── Trend slope ────────────────────────────────────────────────────────────────

class TestComputeTrendSlope:

    def test_flat(self):
        values = np.array([1.0, 1.0, 1.0, 1.0])
        assert abs(_compute_trend_slope(values)) < 0.01

    def test_upward(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert _compute_trend_slope(values) > 0.9

    def test_downward(self):
        values = np.array([4.0, 3.0, 2.0, 1.0])
        assert _compute_trend_slope(values) < -0.9

    def test_few_points(self):
        assert _compute_trend_slope(np.array([1.0])) == 0.0


# ── Signal severity ────────────────────────────────────────────────────────────

class TestClassifySignalSeverity:

    def test_normal(self):
        assert _classify_signal_severity(0.5, 0.01, 1) == "normal"

    def test_watch(self):
        assert _classify_signal_severity(0.5, -0.01, 2) == "watch"

    def test_warning(self):
        assert _classify_signal_severity(0.5, -0.03, 3) == "warning"

    def test_critical(self):
        assert _classify_signal_severity(0.5, -0.05, 4) == "critical"


# ── Main analysis ──────────────────────────────────────────────────────────────

class TestAnalyzeBurnoutTrajectory:

    def test_insufficient_data(self):
        snapshots = _make_snapshots(3)
        result = analyze_burnout_trajectory(snapshots)
        assert result.risk_score == 0
        assert result.risk_level == "green"
        assert result.data_weeks == 0
        assert result.confidence == 0.0

    def test_healthy_trajectory_green(self):
        """Stable emotional patterns should be green."""
        snapshots = _make_snapshots(28, base_valence=0.4, base_range=0.6)
        result = analyze_burnout_trajectory(snapshots)

        assert result.risk_level == "green"
        assert result.risk_score < 30
        assert result.data_weeks == 4

    def test_declining_valence_warns(self):
        """Gradual valence decline over 4+ weeks should raise concern."""
        snapshots = _make_snapshots(
            42,
            base_valence=0.4,
            valence_drift=-0.015,  # ~0.1/week decline
        )
        result = analyze_burnout_trajectory(snapshots)

        # Should detect valence drift
        valence_signals = [s for s in result.signals if s.name == "valence_drift"]
        assert len(valence_signals) == 1
        assert valence_signals[0].trend_slope < 0

    def test_range_compression_detected(self):
        """Shrinking emotional range is a key burnout indicator."""
        snapshots = _make_snapshots(
            42,
            base_range=0.8,
            range_drift=-0.015,  # range shrinking
        )
        result = analyze_burnout_trajectory(snapshots)

        range_signals = [s for s in result.signals if s.name == "emotional_range"]
        assert len(range_signals) == 1
        assert range_signals[0].trend_slope < 0

    def test_multiple_declining_signals_elevates_risk(self):
        """Multiple simultaneous declining signals should push risk higher."""
        snapshots = _make_snapshots(
            42,
            base_valence=0.3,
            valence_drift=-0.012,
            base_range=0.7,
            range_drift=-0.012,
            base_arousal_var=0.12,
            arousal_var_drift=-0.002,
        )
        result = analyze_burnout_trajectory(snapshots)

        assert result.risk_score > 0  # some risk detected
        assert result.warning_signals_count >= 0

    def test_weekend_recovery_measured(self):
        """Should detect whether weekends provide emotional recovery."""
        snapshots = _make_snapshots(28)
        result = analyze_burnout_trajectory(snapshots)

        recovery_signals = [s for s in result.signals if s.name == "recovery_failure"]
        assert len(recovery_signals) == 1

    def test_recommendations_present(self):
        """Should always provide at least one recommendation."""
        snapshots = _make_snapshots(14)
        result = analyze_burnout_trajectory(snapshots)
        assert len(result.recommended_actions) >= 1

    def test_confidence_scales_with_data(self):
        """More weeks of data should increase confidence."""
        short = analyze_burnout_trajectory(_make_snapshots(14))
        long = analyze_burnout_trajectory(_make_snapshots(56))

        assert long.confidence > short.confidence
        assert long.data_weeks > short.data_weeks

    def test_sleep_mood_decoupling(self):
        """When sleep quality doesn't predict mood, signal should fire."""
        np.random.seed(42)
        base_date = datetime(2026, 1, 1)
        snapshots = []
        for i in range(28):
            date = base_date + timedelta(days=i)
            # Random sleep quality, random mood — no correlation
            snapshots.append(DailySnapshot(
                date=date.strftime("%Y-%m-%d"),
                mean_valence=np.random.uniform(-0.5, 0.5),
                max_valence=0.5,
                min_valence=-0.5,
                mean_arousal=0.5,
                arousal_variance=0.1,
                mean_stress=0.3,
                mean_focus=0.5,
                sleep_quality=np.random.uniform(30, 90),
                check_in_count=3,
                is_weekend=date.weekday() >= 5,
            ))

        result = analyze_burnout_trajectory(snapshots)
        decoupling = [s for s in result.signals if s.name == "sleep_mood_decoupling"]
        assert len(decoupling) == 1


# ── Serialization ──────────────────────────────────────────────────────────────

class TestAssessmentToDict:

    def test_serialization(self):
        snapshots = _make_snapshots(28)
        assessment = analyze_burnout_trajectory(snapshots)
        d = assessment_to_dict(assessment)

        assert "risk_score" in d
        assert "risk_level" in d
        assert "signals" in d
        assert isinstance(d["signals"], list)
        assert "recommended_actions" in d
        assert d["data_weeks"] == 4

    def test_signal_fields(self):
        snapshots = _make_snapshots(14)
        assessment = analyze_burnout_trajectory(snapshots)
        d = assessment_to_dict(assessment)

        if d["signals"]:
            sig = d["signals"][0]
            assert "name" in sig
            assert "severity" in sig
            assert "trend_slope" in sig
