"""Tests for circadian rhythm monitor."""
import numpy as np
import pytest

from models.circadian_monitor import CircadianMonitor


@pytest.fixture
def monitor():
    return CircadianMonitor(fs=256.0)


def _make_signal(fs=256, duration=4, n_channels=4, alpha_amp=10.0, theta_amp=5.0,
                 beta_amp=5.0, noise_amp=2.0):
    """EEG with controllable band amplitudes."""
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + ch * 0.5)
        beta = beta_amp * np.sin(2 * np.pi * 20 * t + ch * 0.7)
        noise = noise_amp * np.random.randn(len(t))
        signals.append(alpha + theta + beta + noise)
    return np.array(signals)


class TestBaseline:
    def test_set_baseline(self, monitor):
        np.random.seed(42)
        result = monitor.set_morning_baseline(_make_signal(), hour_of_day=9)
        assert result["baseline_set"] is True
        assert result["hour_of_day"] == 9

    def test_baseline_metrics(self, monitor):
        np.random.seed(42)
        result = monitor.set_morning_baseline(_make_signal())
        assert "metrics" in result

    def test_single_channel(self, monitor):
        np.random.seed(42)
        sig = 10 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        result = monitor.set_morning_baseline(sig)
        assert result["baseline_set"] is True


class TestAssess:
    def test_output_keys(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        expected = {"alertness_score", "alertness_level", "sleep_pressure",
                    "recommendations", "has_baseline"}
        assert expected.issubset(set(result.keys()))

    def test_alertness_range(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        assert 0 <= result["alertness_score"] <= 100

    def test_level_valid(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        assert result["alertness_level"] in (
            "peak", "alert", "moderate", "drowsy", "very_drowsy"
        )

    def test_sleep_pressure_range(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        assert 0 <= result["sleep_pressure"] <= 1

    def test_alpha_peak_freq(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        assert 7 <= result["alpha_peak_freq"] <= 14


class TestAlertness:
    def test_high_alertness_high_alpha(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal(alpha_amp=20, theta_amp=2, beta_amp=8))
        assert result["alertness_score"] > 50

    def test_drowsy_high_theta(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal(alpha_amp=2, theta_amp=20, beta_amp=2))
        assert result["alertness_score"] < 50


class TestOptimalWindow:
    def test_with_hour(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal(), hour_of_day=10)
        assert result["optimal_window"] is not None
        assert "recommended_task" in result["optimal_window"]

    def test_without_hour(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        assert result["optimal_window"] is None


class TestRecommendations:
    def test_recommendations_list(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        assert isinstance(result["recommendations"], list)


class TestSessionStats:
    def test_empty_stats(self, monitor):
        stats = monitor.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_after_data(self, monitor):
        np.random.seed(42)
        for _ in range(5):
            monitor.assess(_make_signal())
        stats = monitor.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "mean_alertness" in stats
        assert "dominant_level" in stats


class TestHistory:
    def test_empty_history(self, monitor):
        assert monitor.get_history() == []

    def test_history_grows(self, monitor):
        np.random.seed(42)
        monitor.assess(_make_signal())
        monitor.assess(_make_signal())
        assert len(monitor.get_history()) == 2

    def test_history_last_n(self, monitor):
        np.random.seed(42)
        for _ in range(10):
            monitor.assess(_make_signal())
        assert len(monitor.get_history(last_n=3)) == 3


class TestMultiUser:
    def test_independent(self, monitor):
        np.random.seed(42)
        monitor.assess(_make_signal(), user_id="a")
        monitor.assess(_make_signal(), user_id="b")
        assert len(monitor.get_history("a")) == 1
        assert len(monitor.get_history("b")) == 1


class TestReset:
    def test_reset_clears(self, monitor):
        np.random.seed(42)
        monitor.set_morning_baseline(_make_signal())
        monitor.assess(_make_signal())
        monitor.reset()
        assert monitor.get_history() == []
        stats = monitor.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False
