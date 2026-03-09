"""Tests for adaptive workload controller."""
import numpy as np
import pytest

from models.workload_adapter import WorkloadAdapter


@pytest.fixture
def adapter():
    return WorkloadAdapter(fs=256.0, adaptation_rate=0.1)


def _make_signal(fs=256, duration=4, n_channels=4, theta_amp=5.0, alpha_amp=10.0,
                 beta_amp=5.0, noise_amp=2.0):
    """EEG with controllable band amplitudes."""
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + ch * 0.3)
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.5)
        beta = beta_amp * np.sin(2 * np.pi * 20 * t + ch * 0.7)
        noise = noise_amp * np.random.randn(len(t))
        signals.append(theta + alpha + beta + noise)
    return np.array(signals)


class TestBaseline:
    def test_set_baseline(self, adapter):
        np.random.seed(42)
        result = adapter.set_baseline(_make_signal())
        assert result["baseline_set"] is True
        assert result["initial_difficulty"] == 0.5

    def test_baseline_powers(self, adapter):
        np.random.seed(42)
        result = adapter.set_baseline(_make_signal())
        assert "band_powers" in result
        for band in ("theta", "alpha", "beta"):
            assert band in result["band_powers"]

    def test_single_channel(self, adapter):
        np.random.seed(42)
        sig = 10 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        result = adapter.set_baseline(sig)
        assert result["baseline_set"] is True


class TestAssess:
    def test_output_keys(self, adapter):
        np.random.seed(42)
        result = adapter.assess(_make_signal())
        expected = {"workload_score", "workload_zone", "fatigue_index",
                    "current_difficulty", "adaptation_command", "has_baseline"}
        assert expected.issubset(set(result.keys()))

    def test_workload_range(self, adapter):
        np.random.seed(42)
        result = adapter.assess(_make_signal())
        assert 0 <= result["workload_score"] <= 1

    def test_zone_valid(self, adapter):
        np.random.seed(42)
        result = adapter.assess(_make_signal())
        assert result["workload_zone"] in (
            "underload", "optimal", "overload", "fatigue"
        )

    def test_command_valid(self, adapter):
        np.random.seed(42)
        result = adapter.assess(_make_signal())
        assert result["adaptation_command"] in (
            "maintain", "increase_difficulty", "reduce_difficulty", "reduce_and_break"
        )

    def test_difficulty_range(self, adapter):
        np.random.seed(42)
        adapter.set_baseline(_make_signal())
        result = adapter.assess(_make_signal())
        assert 0 <= result["current_difficulty"] <= 1


class TestWorkloadZones:
    def test_high_workload_with_high_theta(self, adapter):
        np.random.seed(42)
        adapter.set_baseline(_make_signal())
        result = adapter.assess(_make_signal(theta_amp=20, alpha_amp=2, beta_amp=10))
        assert result["workload_score"] > 0.5

    def test_low_workload_with_high_alpha(self, adapter):
        np.random.seed(42)
        adapter.set_baseline(_make_signal())
        result = adapter.assess(_make_signal(theta_amp=2, alpha_amp=20, beta_amp=2))
        assert result["workload_score"] < 0.5


class TestDifficultyAdaptation:
    def test_difficulty_increases_on_underload(self, adapter):
        np.random.seed(42)
        adapter.set_baseline(_make_signal())
        # Low theta, high alpha = underload
        for _ in range(5):
            result = adapter.assess(_make_signal(theta_amp=2, alpha_amp=20, beta_amp=2))
        assert result["current_difficulty"] > 0.5

    def test_difficulty_decreases_on_overload(self, adapter):
        np.random.seed(42)
        adapter.set_baseline(_make_signal())
        # High theta, low alpha = overload
        for _ in range(5):
            result = adapter.assess(_make_signal(theta_amp=20, alpha_amp=2, beta_amp=15))
        assert result["current_difficulty"] < 0.5

    def test_difficulty_stable_in_optimal(self, adapter):
        np.random.seed(42)
        adapter.set_baseline(_make_signal())
        # Balanced signal
        result = adapter.assess(_make_signal(theta_amp=5, alpha_amp=8, beta_amp=5))
        if result["workload_zone"] == "optimal":
            assert result["difficulty_adjustment"] == 0.0


class TestFatigue:
    def test_no_fatigue_initially(self, adapter):
        np.random.seed(42)
        result = adapter.assess(_make_signal())
        assert result["fatigue_index"] == 0.0

    def test_fatigue_range(self, adapter):
        np.random.seed(42)
        adapter.set_baseline(_make_signal())
        for _ in range(10):
            result = adapter.assess(_make_signal())
        assert 0 <= result["fatigue_index"] <= 1


class TestSessionStats:
    def test_empty_stats(self, adapter):
        stats = adapter.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_after_data(self, adapter):
        np.random.seed(42)
        adapter.set_baseline(_make_signal())
        for _ in range(5):
            adapter.assess(_make_signal())
        stats = adapter.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "mean_workload" in stats
        assert "zone_distribution" in stats
        assert "optimal_percentage" in stats


class TestHistory:
    def test_empty_history(self, adapter):
        assert adapter.get_history() == []

    def test_history_grows(self, adapter):
        np.random.seed(42)
        adapter.assess(_make_signal())
        adapter.assess(_make_signal())
        assert len(adapter.get_history()) == 2

    def test_history_last_n(self, adapter):
        np.random.seed(42)
        for _ in range(10):
            adapter.assess(_make_signal())
        assert len(adapter.get_history(last_n=3)) == 3


class TestMultiUser:
    def test_independent(self, adapter):
        np.random.seed(42)
        adapter.set_baseline(_make_signal(), user_id="a")
        adapter.assess(_make_signal(), user_id="a")
        adapter.set_baseline(_make_signal(), user_id="b")
        adapter.assess(_make_signal(), user_id="b")
        a = adapter.get_session_stats("a")
        b = adapter.get_session_stats("b")
        assert a["n_epochs"] == 1
        assert b["n_epochs"] == 1


class TestReset:
    def test_reset_clears(self, adapter):
        np.random.seed(42)
        adapter.set_baseline(_make_signal())
        adapter.assess(_make_signal())
        adapter.reset()
        assert adapter.get_history() == []
        stats = adapter.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False
