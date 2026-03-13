"""Tests for CGMAnalyzer — overnight glucose / sleep quality correlation."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from health.cgm_analyzer import CGMAnalyzer, OvernightGlucoseMetrics


# ─── Helpers ─────────────────────────────────────────────────────────────────

SLEEP_START = datetime(2026, 3, 10, 22, 0)   # 10 PM
WAKE_TIME   = datetime(2026, 3, 11, 6, 0)    # 6 AM


def _make_readings(values: list, start: datetime = SLEEP_START) -> list:
    """Create glucose readings at 5-minute intervals starting from `start`."""
    readings = []
    for i, v in enumerate(values):
        readings.append({
            "timestamp": start + timedelta(minutes=5 * i),
            "value": v,
        })
    return readings


# ─── Overnight filtering ─────────────────────────────────────────────────────

class TestOvernightFiltering:
    """Readings outside the sleep window must be excluded."""

    def test_readings_inside_window_included(self):
        analyzer = CGMAnalyzer()
        # All 3 readings are within the sleep window
        readings = _make_readings([90, 95, 100], start=SLEEP_START + timedelta(hours=1))
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.readings_count == 3

    def test_readings_before_sleep_excluded(self):
        analyzer = CGMAnalyzer()
        before = [
            {"timestamp": SLEEP_START - timedelta(hours=1), "value": 150},
            {"timestamp": SLEEP_START - timedelta(minutes=1), "value": 140},
        ]
        inside = _make_readings([90, 95], start=SLEEP_START + timedelta(minutes=10))
        metrics = analyzer.analyze_overnight(before + inside, SLEEP_START, WAKE_TIME)
        assert metrics.readings_count == 2

    def test_readings_after_wake_excluded(self):
        analyzer = CGMAnalyzer()
        inside  = _make_readings([85, 90], start=SLEEP_START + timedelta(hours=1))
        after   = [
            {"timestamp": WAKE_TIME + timedelta(minutes=30), "value": 130},
        ]
        metrics = analyzer.analyze_overnight(inside + after, SLEEP_START, WAKE_TIME)
        assert metrics.readings_count == 2

    def test_boundary_timestamps_included(self):
        """Readings exactly at sleep_start and wake_time must be included."""
        analyzer = CGMAnalyzer()
        readings = [
            {"timestamp": SLEEP_START, "value": 88},
            {"timestamp": WAKE_TIME,   "value": 92},
        ]
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.readings_count == 2

    def test_no_readings_raises_value_error(self):
        analyzer = CGMAnalyzer()
        before = [{"timestamp": SLEEP_START - timedelta(hours=2), "value": 110}]
        with pytest.raises(ValueError, match="No glucose readings found"):
            analyzer.analyze_overnight(before, SLEEP_START, WAKE_TIME)

    def test_iso_string_timestamps_parsed(self):
        """Timestamps supplied as ISO-8601 strings should be handled."""
        analyzer = CGMAnalyzer()
        readings = [
            {"timestamp": (SLEEP_START + timedelta(hours=1)).isoformat(), "value": 95},
            {"timestamp": (SLEEP_START + timedelta(hours=2)).isoformat(), "value": 100},
        ]
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.readings_count == 2


# ─── Metric calculations ──────────────────────────────────────────────────────

class TestMetricCalculations:
    """Verify computed metrics against known input data."""

    def test_mean_glucose(self):
        analyzer = CGMAnalyzer()
        readings = _make_readings([80, 100, 120])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.mean_glucose == pytest.approx(100.0, abs=0.1)

    def test_min_max_glucose(self):
        analyzer = CGMAnalyzer()
        readings = _make_readings([70, 95, 110, 140])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.min_glucose == pytest.approx(70.0, abs=0.1)
        assert metrics.max_glucose == pytest.approx(140.0, abs=0.1)

    def test_std_zero_for_constant_values(self):
        analyzer = CGMAnalyzer()
        readings = _make_readings([90, 90, 90, 90])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.glucose_std == pytest.approx(0.0, abs=0.01)

    def test_std_nonzero_for_variable_values(self):
        analyzer = CGMAnalyzer()
        readings = _make_readings([70, 130, 70, 130])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.glucose_std > 0

    def test_time_in_range_all_in_range(self):
        """All readings 70–120 → 100 % TIR."""
        analyzer = CGMAnalyzer()
        readings = _make_readings([75, 85, 95, 115, 70, 120])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.time_in_range_pct == pytest.approx(100.0, abs=0.1)

    def test_time_in_range_none_in_range(self):
        """All readings outside 70–120 → 0 % TIR."""
        analyzer = CGMAnalyzer()
        readings = _make_readings([130, 140, 150, 60, 65])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.time_in_range_pct == pytest.approx(0.0, abs=0.1)

    def test_time_in_range_partial(self):
        """2 of 4 readings in range → 50 %."""
        analyzer = CGMAnalyzer()
        readings = _make_readings([90, 100, 130, 140])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.time_in_range_pct == pytest.approx(50.0, abs=0.1)

    def test_single_reading(self):
        """Exactly one reading in window — std should be 0.0."""
        analyzer = CGMAnalyzer()
        readings = [{"timestamp": SLEEP_START + timedelta(hours=1), "value": 95}]
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.readings_count == 1
        assert metrics.glucose_std == 0.0


# ─── Excursion counting ───────────────────────────────────────────────────────

class TestExcursionCounting:
    """Readings above 120 mg/dL are excursions; at/below 120 are not."""

    def test_no_excursions(self):
        analyzer = CGMAnalyzer()
        readings = _make_readings([70, 90, 110, 120])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.excursion_count == 0

    def test_all_excursions(self):
        analyzer = CGMAnalyzer()
        readings = _make_readings([125, 130, 145, 160])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.excursion_count == 4

    def test_mixed_excursions(self):
        analyzer = CGMAnalyzer()
        readings = _make_readings([90, 121, 110, 135, 80])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.excursion_count == 2

    def test_threshold_boundary_120_not_excursion(self):
        """Exactly 120 mg/dL is in-range, not an excursion."""
        analyzer = CGMAnalyzer()
        readings = _make_readings([120])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.excursion_count == 0

    def test_threshold_boundary_121_is_excursion(self):
        """121 mg/dL exceeds the 120 mg/dL ceiling."""
        analyzer = CGMAnalyzer()
        readings = _make_readings([121])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.excursion_count == 1


# ─── Correlation direction ────────────────────────────────────────────────────

class TestCorrelationDirection:
    """High variability should correlate with poor sleep metrics."""

    def test_high_variability_negative_direction_vs_deep_sleep(self):
        """Variability > 20 mg/dL → direction reported as 'negative'."""
        analyzer = CGMAnalyzer()
        # glucose_std >> 20: readings jump between 60 and 160
        readings = _make_readings([60, 160, 60, 160, 60, 160])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.glucose_std > 20

        correlations = analyzer.correlate_with_sleep(
            metrics,
            {"deep_sleep_pct": 15.0, "awakenings": 4, "sleep_efficiency": 70.0},
        )
        assert correlations["glucose_variability_vs_deep_sleep"]["direction"] == "negative"

    def test_low_variability_neutral_direction_vs_deep_sleep(self):
        """Variability < 6 mg/dL → direction reported as 'neutral'."""
        analyzer = CGMAnalyzer()
        readings = _make_readings([88, 90, 91, 89, 90, 88])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        assert metrics.glucose_std < 6

        correlations = analyzer.correlate_with_sleep(
            metrics,
            {"deep_sleep_pct": 22.0},
        )
        assert correlations["glucose_variability_vs_deep_sleep"]["direction"] == "neutral"

    def test_excursion_awakening_ratio_present_when_awakenings_given(self):
        analyzer = CGMAnalyzer()
        readings = _make_readings([130, 140, 150, 90])  # 3 excursions
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)

        correlations = analyzer.correlate_with_sleep(metrics, {"awakenings": 3})
        assert "excursions_vs_awakenings" in correlations
        assert correlations["excursions_vs_awakenings"]["excursion_count"] == 3

    def test_elevated_mean_glucose_flags_efficiency_penalty(self):
        """Mean > 110 should produce a non-zero efficiency penalty."""
        analyzer = CGMAnalyzer()
        readings = _make_readings([130, 135, 140, 130])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)

        correlations = analyzer.correlate_with_sleep(
            metrics, {"sleep_efficiency": 75.0}
        )
        penalty = correlations["mean_glucose_vs_sleep_efficiency"]["efficiency_penalty_score"]
        assert penalty > 0

    def test_summary_score_higher_for_good_control(self):
        """Well-controlled glucose (high TIR, low SD) scores better than poor control."""
        analyzer = CGMAnalyzer()

        good_readings = _make_readings([85, 90, 88, 92, 87, 90])
        poor_readings = _make_readings([80, 150, 70, 160, 75, 145])

        good_metrics = analyzer.analyze_overnight(good_readings, SLEEP_START, WAKE_TIME)
        poor_metrics = analyzer.analyze_overnight(poor_readings, SLEEP_START, WAKE_TIME)

        good_corr = analyzer.correlate_with_sleep(good_metrics, {"sleep_efficiency": 85})
        poor_corr = analyzer.correlate_with_sleep(poor_metrics, {"sleep_efficiency": 60})

        assert (
            good_corr["summary"]["glucose_sleep_score"]
            > poor_corr["summary"]["glucose_sleep_score"]
        )

    def test_correlate_with_no_optional_sleep_fields(self):
        """correlate_with_sleep should not crash when sleep_data is empty."""
        analyzer = CGMAnalyzer()
        readings = _make_readings([90, 95, 100])
        metrics = analyzer.analyze_overnight(readings, SLEEP_START, WAKE_TIME)
        result = analyzer.correlate_with_sleep(metrics, {})
        assert "summary" in result

    def test_weekly_patterns_returns_expected_keys(self):
        """get_weekly_patterns should return trend, best/worst, and summary keys."""
        analyzer = CGMAnalyzer()
        daily = []
        base = SLEEP_START
        for i in range(7):
            readings = _make_readings(
                [85 + i * 3, 90 + i * 2, 88 + i],
                start=base + timedelta(days=i, hours=1),
            )
            sleep_end = base + timedelta(days=i, hours=8)
            metrics = analyzer.analyze_overnight(
                readings,
                base + timedelta(days=i),
                sleep_end,
            )
            daily.append({
                "date": (base + timedelta(days=i)).date().isoformat(),
                "glucose_metrics": metrics,
                "sleep_quality": max(0, 80 - i * 3),
            })

        patterns = analyzer.get_weekly_patterns(daily)
        assert patterns["days_analyzed"] == 7
        assert "variability_trend" in patterns
        assert "time_in_range_trend" in patterns
        assert "best_night" in patterns
        assert "worst_night" in patterns
        assert "weekly_summary" in patterns
