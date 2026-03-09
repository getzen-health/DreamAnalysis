"""Tests for circadian normalizer."""
from datetime import datetime
import pytest

from processing.circadian_normalizer import CircadianNormalizer, CHRONOTYPE_OFFSETS


@pytest.fixture
def normalizer():
    return CircadianNormalizer()


class TestChronotype:
    def test_morning_chronotype(self, normalizer):
        """Early mid-sleep → morning chronotype."""
        records = [{"wake_time_hours": 5.5, "sleep_time_hours": 21.5}] * 7
        result = normalizer.estimate_chronotype(records)
        assert result["chronotype"] == "morning"
        assert result["offset_hours"] < 0

    def test_evening_chronotype(self, normalizer):
        """Late mid-sleep → evening chronotype."""
        records = [{"wake_time_hours": 10.0, "sleep_time_hours": 2.0}] * 7
        result = normalizer.estimate_chronotype(records)
        assert result["chronotype"] == "evening"
        assert result["offset_hours"] > 0

    def test_intermediate_chronotype(self, normalizer):
        """Average mid-sleep → intermediate."""
        records = [{"wake_time_hours": 7.0, "sleep_time_hours": 23.0}] * 7
        result = normalizer.estimate_chronotype(records)
        assert result["chronotype"] == "intermediate"
        assert result["offset_hours"] == 0

    def test_insufficient_data(self, normalizer):
        """Too few records → default intermediate."""
        result = normalizer.estimate_chronotype([{"wake_time_hours": 7}])
        assert result["chronotype"] == "intermediate"
        assert result["confidence"] == "low"

    def test_empty_records(self, normalizer):
        """Empty records → default."""
        result = normalizer.estimate_chronotype([])
        assert result["chronotype"] == "intermediate"

    def test_high_confidence_with_enough_data(self, normalizer):
        """7+ records → high confidence."""
        records = [{"wake_time_hours": 7.0, "sleep_time_hours": 23.0}] * 10
        result = normalizer.estimate_chronotype(records)
        assert result["confidence"] == "high"

    def test_uses_last_14_days(self, normalizer):
        """Should use only last 14 records."""
        old = [{"wake_time_hours": 5.0, "sleep_time_hours": 21.0}] * 20
        recent = [{"wake_time_hours": 10.0, "sleep_time_hours": 2.0}] * 14
        result = normalizer.estimate_chronotype(old + recent)
        assert result["chronotype"] == "evening"


class TestCorrection:
    def test_correction_applied(self, normalizer):
        """Features should be modified by correction."""
        normalizer.set_wake_time(7.0)
        features = {"theta": 0.5, "alpha": 0.4, "beta": 0.3}
        session = datetime(2026, 3, 8, 15, 0)  # 3 PM, 8 hours awake
        corrected = normalizer.correct_features(features, session)
        assert corrected["circadian_correction_applied"] is True
        assert corrected["hours_awake_estimate"] > 0

    def test_theta_reduced_late(self, normalizer):
        """Late session theta should be reduced (removing sleep pressure drift)."""
        normalizer.set_wake_time(7.0)
        features = {"theta": 0.5}
        late = datetime(2026, 3, 8, 22, 0)  # 10 PM, 15 hours awake
        corrected = normalizer.correct_features(features, late)
        assert corrected["theta"] < features["theta"]

    def test_alpha_increased_late(self, normalizer):
        """Late session alpha should be increased (compensating for natural decline)."""
        normalizer.set_wake_time(7.0)
        features = {"alpha": 0.4}
        late = datetime(2026, 3, 8, 22, 0)
        corrected = normalizer.correct_features(features, late)
        assert corrected["alpha"] > features["alpha"]

    def test_morning_minimal_correction(self, normalizer):
        """Shortly after wake → minimal correction."""
        normalizer.set_wake_time(7.0)
        features = {"theta": 0.5, "alpha": 0.4}
        morning = datetime(2026, 3, 8, 7, 30)
        corrected = normalizer.correct_features(features, morning)
        assert abs(corrected["theta"] - features["theta"]) < 0.02

    def test_non_band_features_preserved(self, normalizer):
        """Non-band features should pass through unchanged."""
        features = {"theta": 0.5, "custom_metric": 42, "label": "test"}
        corrected = normalizer.correct_features(features)
        assert corrected["custom_metric"] == 42
        assert corrected["label"] == "test"

    def test_power_suffix_keys(self, normalizer):
        """Should handle both 'theta' and 'theta_power' keys."""
        features = {"theta_power": 0.5, "alpha_power": 0.4}
        normalizer.set_wake_time(7.0)
        corrected = normalizer.correct_features(features, datetime(2026, 3, 8, 15, 0))
        assert corrected["circadian_correction_applied"] is True

    def test_default_time_is_now(self, normalizer):
        """Should use current time if none provided."""
        features = {"theta": 0.5}
        corrected = normalizer.correct_features(features)
        assert "hours_awake_estimate" in corrected


class TestExpectedDrift:
    def test_drift_values(self, normalizer):
        """Expected drift should scale with hours awake."""
        drift_4h = normalizer.get_expected_drift(4)
        drift_8h = normalizer.get_expected_drift(8)
        assert abs(drift_8h["theta"]) > abs(drift_4h["theta"])

    def test_drift_zero_at_wake(self, normalizer):
        """Zero hours awake → zero drift."""
        drift = normalizer.get_expected_drift(0)
        for band, value in drift.items():
            assert value == 0.0


class TestWakeTime:
    def test_set_wake_time(self, normalizer):
        """Setting wake time should affect hours awake calculation."""
        normalizer.set_wake_time(6.0)
        features = {"theta": 0.5}
        noon = datetime(2026, 3, 8, 12, 0)
        corrected = normalizer.correct_features(features, noon)
        assert corrected["hours_awake_estimate"] == pytest.approx(6.0, abs=0.1)

    def test_wake_time_clamped(self, normalizer):
        """Wake time should be clamped to valid range."""
        normalizer.set_wake_time(30.0)  # invalid
        # Should not crash


class TestEdgeCases:
    def test_midnight_wrap(self, normalizer):
        """Session after midnight should handle wrap correctly."""
        normalizer.set_wake_time(7.0)
        features = {"theta": 0.5}
        late = datetime(2026, 3, 9, 1, 0)  # 1 AM
        corrected = normalizer.correct_features(features, late)
        assert corrected["hours_awake_estimate"] > 15  # 18 hours awake
