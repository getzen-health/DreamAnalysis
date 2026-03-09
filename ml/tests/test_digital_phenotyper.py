"""Tests for digital phenotyper."""
import pytest

from models.digital_phenotyper import DigitalPhenotyper


@pytest.fixture
def phenotyper():
    return DigitalPhenotyper()


def _good_eeg():
    return {"mean_valence": 0.5, "mean_arousal": 0.5, "mean_stress": 0.2, "session_count": 3}


def _bad_eeg():
    return {"mean_valence": -0.6, "mean_arousal": 0.8, "mean_stress": 0.8, "session_count": 1}


def _good_health():
    return {"steps": 10000, "sleep_hours": 8.0, "resting_hr": 58, "hrv_ms": 55, "active_energy_kcal": 500}


def _bad_health():
    return {"steps": 1500, "sleep_hours": 4.0, "resting_hr": 95, "hrv_ms": 15}


class TestDailyScore:
    def test_output_keys(self, phenotyper):
        result = phenotyper.compute_daily_score(_good_eeg(), _good_health())
        expected = {"mental_health_score", "eeg_component", "health_component",
                    "risk_level", "risk_flags", "data_quality"}
        assert expected.issubset(set(result.keys()))

    def test_score_range(self, phenotyper):
        result = phenotyper.compute_daily_score(_good_eeg(), _good_health())
        assert 0 <= result["mental_health_score"] <= 100

    def test_good_data_high_score(self, phenotyper):
        result = phenotyper.compute_daily_score(_good_eeg(), _good_health())
        assert result["mental_health_score"] > 60

    def test_bad_data_low_score(self, phenotyper):
        result = phenotyper.compute_daily_score(_bad_eeg(), _bad_health())
        assert result["mental_health_score"] < 50

    def test_full_quality_with_both(self, phenotyper):
        result = phenotyper.compute_daily_score(_good_eeg(), _good_health())
        assert result["data_quality"] == "full"

    def test_eeg_only_quality(self, phenotyper):
        result = phenotyper.compute_daily_score(_good_eeg(), None)
        assert result["data_quality"] == "eeg_only"

    def test_health_only_quality(self, phenotyper):
        result = phenotyper.compute_daily_score(None, _good_health())
        assert result["data_quality"] == "health_only"

    def test_no_data_quality(self, phenotyper):
        result = phenotyper.compute_daily_score(None, None)
        assert result["data_quality"] == "no_data"
        assert result["mental_health_score"] == 50.0


class TestRiskFlags:
    def test_no_flags_good_data(self, phenotyper):
        result = phenotyper.compute_daily_score(_good_eeg(), _good_health())
        assert result["risk_flags"] == []

    def test_negative_valence_flag(self, phenotyper):
        result = phenotyper.compute_daily_score(_bad_eeg(), _good_health())
        assert "persistent_negative_valence" in result["risk_flags"]

    def test_stress_flag(self, phenotyper):
        result = phenotyper.compute_daily_score(_bad_eeg(), _good_health())
        assert "elevated_stress" in result["risk_flags"]

    def test_sleep_flag(self, phenotyper):
        result = phenotyper.compute_daily_score(None, _bad_health())
        assert "severe_sleep_deficit" in result["risk_flags"]

    def test_low_activity_flag(self, phenotyper):
        result = phenotyper.compute_daily_score(None, _bad_health())
        assert "very_low_activity" in result["risk_flags"]


class TestRiskLevels:
    def test_low_risk(self, phenotyper):
        result = phenotyper.compute_daily_score(_good_eeg(), _good_health())
        assert result["risk_level"] == "low"

    def test_high_risk(self, phenotyper):
        result = phenotyper.compute_daily_score(_bad_eeg(), _bad_health())
        assert result["risk_level"] in ("elevated", "high")


class TestTrend:
    def test_no_data_stable(self, phenotyper):
        trend = phenotyper.compute_trend()
        assert trend["trend"] == "stable"

    def test_improving_trend(self, phenotyper):
        for i in range(7):
            phenotyper.compute_daily_score(
                {"mean_valence": -0.5 + i * 0.15, "mean_stress": 0.7 - i * 0.08, "session_count": 1},
                {"steps": 5000 + i * 1000, "sleep_hours": 6 + i * 0.3, "resting_hr": 75 - i * 2, "hrv_ms": 30 + i * 5}
            )
        trend = phenotyper.compute_trend()
        assert trend["trend"] == "improving"
        assert trend["slope"] > 0

    def test_declining_trend(self, phenotyper):
        for i in range(7):
            phenotyper.compute_daily_score(
                {"mean_valence": 0.5 - i * 0.15, "mean_stress": 0.2 + i * 0.08, "session_count": 1},
                {"steps": 10000 - i * 1000, "sleep_hours": 8 - i * 0.4, "resting_hr": 55 + i * 3, "hrv_ms": 55 - i * 5}
            )
        trend = phenotyper.compute_trend()
        assert trend["trend"] == "declining"
        assert trend["slope"] < 0

    def test_trend_has_stats(self, phenotyper):
        for _ in range(5):
            phenotyper.compute_daily_score(_good_eeg(), _good_health())
        trend = phenotyper.compute_trend()
        assert "mean_score" in trend
        assert "n_days" in trend


class TestMonthlyReport:
    def test_empty_report(self, phenotyper):
        assert phenotyper.get_monthly_report()["n_days"] == 0

    def test_report_with_data(self, phenotyper):
        for _ in range(10):
            phenotyper.compute_daily_score(_good_eeg(), _good_health())
        report = phenotyper.get_monthly_report()
        assert report["n_days"] == 10
        assert "mean_score" in report
        assert "risk_distribution" in report
        assert "days_at_risk" in report


class TestHistory:
    def test_get_scores(self, phenotyper):
        phenotyper.compute_daily_score(_good_eeg(), _good_health())
        phenotyper.compute_daily_score(_bad_eeg(), _bad_health())
        assert len(phenotyper.get_scores()) == 2

    def test_get_scores_last_n(self, phenotyper):
        for _ in range(10):
            phenotyper.compute_daily_score(_good_eeg(), _good_health())
        assert len(phenotyper.get_scores(last_n=3)) == 3


class TestMultiUser:
    def test_independent_users(self, phenotyper):
        phenotyper.compute_daily_score(_good_eeg(), _good_health(), user_id="a")
        phenotyper.compute_daily_score(_bad_eeg(), _bad_health(), user_id="b")
        a = phenotyper.get_scores("a")
        b = phenotyper.get_scores("b")
        assert len(a) == 1 and len(b) == 1
        assert a[0]["mental_health_score"] > b[0]["mental_health_score"]


class TestReset:
    def test_reset_clears(self, phenotyper):
        phenotyper.compute_daily_score(_good_eeg(), _good_health())
        phenotyper.reset()
        assert phenotyper.get_monthly_report()["n_days"] == 0
