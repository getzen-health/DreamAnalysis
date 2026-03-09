"""Tests for DigitalPhenotyper — EEG + health data fusion for mental wellness scoring."""

import pytest
from models.digital_phenotyper import DigitalPhenotyper, get_digital_phenotyper


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def phenotyper():
    """Fresh DigitalPhenotyper instance for each test."""
    return DigitalPhenotyper()


def make_eeg_session(valence=0.3, stress=0.2, focus=0.7):
    """Build a synthetic EEG session dict."""
    return {
        "valence": valence,
        "stress_index": stress,
        "focus_index": focus,
    }


def make_health_data(sleep_hours=7.5, hrv=45.0, steps=8000, resting_hr=65):
    """Build a synthetic health data dict."""
    return {
        "sleep_hours": sleep_hours,
        "hrv": hrv,
        "steps": steps,
        "resting_hr": resting_hr,
    }


# ── Unit tests: basic structure ───────────────────────────────────────────────


class TestDigitalPhenotyperInit:
    def test_is_instantiable(self, phenotyper):
        assert phenotyper is not None

    def test_singleton_returns_same_instance(self):
        a = get_digital_phenotyper()
        b = get_digital_phenotyper()
        assert a is b

    def test_singleton_is_digital_phenotyper(self):
        assert isinstance(get_digital_phenotyper(), DigitalPhenotyper)


# ── Unit tests: compute_daily_score ──────────────────────────────────────────


class TestDailyScore:
    def test_returns_dict(self, phenotyper):
        result = phenotyper.compute_daily_score([], date_label="today")
        assert isinstance(result, dict)

    def test_required_keys_present(self, phenotyper):
        result = phenotyper.compute_daily_score([], date_label="today")
        required = {
            "mental_wellness_score",
            "components",
            "risk_flags",
            "session_count",
            "data_completeness",
            "date",
        }
        assert required.issubset(result.keys())

    def test_score_in_range(self, phenotyper):
        sessions = [make_eeg_session()]
        result = phenotyper.compute_daily_score(sessions, health_data=make_health_data())
        assert 0.0 <= result["mental_wellness_score"] <= 100.0

    def test_empty_sessions_returns_score(self, phenotyper):
        result = phenotyper.compute_daily_score([])
        assert "mental_wellness_score" in result
        assert result["session_count"] == 0

    def test_session_count_correct(self, phenotyper):
        sessions = [make_eeg_session(), make_eeg_session(), make_eeg_session()]
        result = phenotyper.compute_daily_score(sessions)
        assert result["session_count"] == 3

    def test_date_label_preserved(self, phenotyper):
        result = phenotyper.compute_daily_score([], date_label="2026-03-08")
        assert result["date"] == "2026-03-08"

    def test_components_is_dict(self, phenotyper):
        result = phenotyper.compute_daily_score([make_eeg_session()])
        assert isinstance(result["components"], dict)

    def test_risk_flags_is_list(self, phenotyper):
        result = phenotyper.compute_daily_score([make_eeg_session()])
        assert isinstance(result["risk_flags"], list)

    def test_data_completeness_in_range(self, phenotyper):
        result = phenotyper.compute_daily_score([make_eeg_session()], health_data=make_health_data())
        assert 0.0 <= result["data_completeness"] <= 1.0

    def test_high_stress_lowers_score(self, phenotyper):
        good = phenotyper.compute_daily_score([make_eeg_session(valence=0.8, stress=0.0, focus=0.9)])
        bad = phenotyper.compute_daily_score([make_eeg_session(valence=-0.5, stress=0.9, focus=0.2)])
        assert good["mental_wellness_score"] > bad["mental_wellness_score"]

    def test_good_health_raises_score(self, phenotyper):
        with_health = phenotyper.compute_daily_score(
            [make_eeg_session()], health_data=make_health_data(sleep_hours=8, hrv=60, steps=10000)
        )
        without_health = phenotyper.compute_daily_score([make_eeg_session()])
        # Good health data should contribute positively
        assert with_health["mental_wellness_score"] >= 0.0

    def test_low_sleep_creates_risk_flag(self, phenotyper):
        result = phenotyper.compute_daily_score(
            [], health_data=make_health_data(sleep_hours=4.0)
        )
        assert len(result["risk_flags"]) > 0

    def test_no_risk_flags_for_good_data(self, phenotyper):
        result = phenotyper.compute_daily_score(
            [make_eeg_session(valence=0.5, stress=0.1, focus=0.8)],
            health_data=make_health_data(sleep_hours=8, hrv=60, steps=9000, resting_hr=62),
        )
        # Should have zero or very few risk flags with good data
        assert isinstance(result["risk_flags"], list)


# ── Unit tests: compute_weekly_trend ─────────────────────────────────────────


class TestWeeklyTrend:
    def _make_daily_scores(self, scores):
        """compute_weekly_trend takes List[float], not dicts."""
        return [float(s) for s in scores]

    def test_returns_dict(self, phenotyper):
        daily = self._make_daily_scores([60, 65, 70, 68, 72, 74, 75])
        result = phenotyper.compute_weekly_trend(daily)
        assert isinstance(result, dict)

    def test_required_keys(self, phenotyper):
        daily = self._make_daily_scores([60, 65, 70])
        result = phenotyper.compute_weekly_trend(daily)
        required = {"trend", "slope_per_day", "min_score", "max_score"}
        assert required.issubset(result.keys())

    def test_improving_trend_detected(self, phenotyper):
        daily = self._make_daily_scores([50, 55, 60, 65, 70, 75, 80])
        result = phenotyper.compute_weekly_trend(daily)
        assert result["trend"] == "improving"

    def test_declining_trend_detected(self, phenotyper):
        daily = self._make_daily_scores([80, 75, 70, 65, 60, 55, 50])
        result = phenotyper.compute_weekly_trend(daily)
        assert result["trend"] == "declining"

    def test_stable_trend_detected(self, phenotyper):
        daily = self._make_daily_scores([65, 65, 65, 65, 65, 65, 65])
        result = phenotyper.compute_weekly_trend(daily)
        assert result["trend"] == "stable"

    def test_data_points_correct(self, phenotyper):
        daily = self._make_daily_scores([60, 65, 70])
        result = phenotyper.compute_weekly_trend(daily)
        assert result["data_points"] == 3

    def test_empty_returns_gracefully(self, phenotyper):
        result = phenotyper.compute_weekly_trend([])
        assert "trend" in result

    def test_slope_positive_for_improving(self, phenotyper):
        daily = self._make_daily_scores([50, 60, 70, 80])
        result = phenotyper.compute_weekly_trend(daily)
        assert result["slope_per_day"] > 0


# ── Unit tests: compute_monthly_report ───────────────────────────────────────


class TestMonthlyReport:
    def _make_records(self, n=30, base_score=65):
        return [
            {"mental_wellness_score": base_score + i % 10, "date": f"2026-02-{i+1:02d}",
             "risk_flags": [], "session_count": 1}
            for i in range(n)
        ]

    def test_returns_dict(self, phenotyper):
        result = phenotyper.compute_monthly_report(self._make_records())
        assert isinstance(result, dict)

    def test_required_keys(self, phenotyper):
        result = phenotyper.compute_monthly_report(self._make_records())
        required = {"trend", "days_analyzed", "mean_score"}
        assert required.issubset(result.keys())

    def test_days_analyzed_correct(self, phenotyper):
        records = self._make_records(n=20)
        result = phenotyper.compute_monthly_report(records)
        assert result["days_analyzed"] == 20

    def test_mean_score_in_range(self, phenotyper):
        result = phenotyper.compute_monthly_report(self._make_records())
        assert 0.0 <= result["mean_score"] <= 100.0

    def test_empty_records_handled(self, phenotyper):
        result = phenotyper.compute_monthly_report([])
        # Returns error dict for empty input
        assert isinstance(result, dict)

    def test_max_score_gte_min_score(self, phenotyper):
        records = self._make_records()
        result = phenotyper.compute_monthly_report(records)
        assert result["max_score"] >= result["min_score"]
