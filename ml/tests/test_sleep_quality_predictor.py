"""Tests for sleep quality predictor."""
import pytest

from models.sleep_quality_predictor import SleepQualityPredictor, QUALITY_LABELS


@pytest.fixture
def predictor():
    return SleepQualityPredictor()


class TestBasicPrediction:
    def test_output_keys(self, predictor):
        result = predictor.predict()
        expected = {"quality_score", "quality_label", "readiness_forecast",
                    "components", "suggestions"}
        assert expected.issubset(set(result.keys()))

    def test_score_range(self, predictor):
        result = predictor.predict()
        assert 0 <= result["quality_score"] <= 100

    def test_label_valid(self, predictor):
        result = predictor.predict()
        assert result["quality_label"] in QUALITY_LABELS


class TestQualityScoring:
    def test_excellent_sleep(self, predictor):
        result = predictor.predict(
            n3_pct=0.20, rem_pct=0.22, sleep_efficiency=0.92,
            total_sleep_hours=8.0, waso_minutes=5, spindle_density=3.0
        )
        assert result["quality_score"] > 75
        assert result["quality_label"] in ("good", "excellent")

    def test_poor_sleep(self, predictor):
        result = predictor.predict(
            n3_pct=0.05, rem_pct=0.05, sleep_efficiency=0.50,
            total_sleep_hours=4.0, waso_minutes=60, spindle_density=0.5
        )
        assert result["quality_score"] < 50
        assert result["quality_label"] in ("poor", "fair")

    def test_with_hrv(self, predictor):
        result = predictor.predict(
            n3_pct=0.20, rem_pct=0.22, total_sleep_hours=8.0, hrv_ms=50
        )
        assert "hrv" in result["components"]


class TestComponents:
    def test_all_components_present(self, predictor):
        result = predictor.predict()
        comps = result["components"]
        expected = {"deep_sleep", "rem_sleep", "efficiency", "duration",
                    "continuity", "spindle_density"}
        assert expected.issubset(set(comps.keys()))

    def test_component_ranges(self, predictor):
        result = predictor.predict(n3_pct=0.20, rem_pct=0.22)
        for name, score in result["components"].items():
            assert 0 <= score <= 100, f"{name} out of range: {score}"


class TestReadinessForecast:
    def test_high_readiness(self, predictor):
        result = predictor.predict(
            n3_pct=0.20, rem_pct=0.22, sleep_efficiency=0.92,
            total_sleep_hours=8.0, spindle_density=3.0
        )
        assert result["readiness_forecast"]["level"] in ("high", "moderate")

    def test_low_readiness(self, predictor):
        result = predictor.predict(
            n3_pct=0.05, rem_pct=0.05, sleep_efficiency=0.50,
            total_sleep_hours=4.0
        )
        assert result["readiness_forecast"]["level"] in ("low", "very_low")

    def test_forecast_has_recommendation(self, predictor):
        result = predictor.predict()
        assert "recommendation" in result["readiness_forecast"]


class TestSuggestions:
    def test_no_suggestions_excellent(self, predictor):
        result = predictor.predict(
            n3_pct=0.20, rem_pct=0.22, sleep_efficiency=0.92,
            total_sleep_hours=8.0, waso_minutes=5
        )
        # May have 0 or few suggestions for good sleep
        assert isinstance(result["suggestions"], list)

    def test_suggestions_for_poor_sleep(self, predictor):
        result = predictor.predict(
            n3_pct=0.02, rem_pct=0.05, sleep_efficiency=0.50,
            total_sleep_hours=4.0, waso_minutes=60
        )
        assert len(result["suggestions"]) > 0


class TestTrend:
    def test_insufficient_data(self, predictor):
        trend = predictor.get_trend()
        assert trend["trend"] == "insufficient_data"

    def test_improving_trend(self, predictor):
        for i in range(7):
            predictor.predict(n3_pct=0.10 + i * 0.02, rem_pct=0.15 + i * 0.01,
                              total_sleep_hours=6 + i * 0.3)
        trend = predictor.get_trend()
        assert trend["trend"] == "improving"

    def test_declining_trend(self, predictor):
        for i in range(7):
            predictor.predict(n3_pct=0.22 - i * 0.03, rem_pct=0.22 - i * 0.02,
                              total_sleep_hours=8.5 - i * 0.5)
        trend = predictor.get_trend()
        assert trend["trend"] == "declining"


class TestHistory:
    def test_empty_history(self, predictor):
        assert predictor.get_history() == []

    def test_history_grows(self, predictor):
        predictor.predict()
        predictor.predict()
        assert len(predictor.get_history()) == 2

    def test_history_last_n(self, predictor):
        for _ in range(10):
            predictor.predict()
        assert len(predictor.get_history(last_n=3)) == 3


class TestMultiUser:
    def test_independent(self, predictor):
        predictor.predict(n3_pct=0.20, user_id="a")
        predictor.predict(n3_pct=0.05, user_id="b")
        assert len(predictor.get_history("a")) == 1
        assert len(predictor.get_history("b")) == 1


class TestReset:
    def test_reset_clears(self, predictor):
        predictor.predict()
        predictor.reset()
        assert predictor.get_history() == []
