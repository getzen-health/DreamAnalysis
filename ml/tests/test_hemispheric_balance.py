"""Tests for hemispheric balance monitor."""
import numpy as np
import pytest

from models.hemispheric_balance import HemisphericBalanceMonitor


@pytest.fixture
def monitor():
    return HemisphericBalanceMonitor(fs=256.0)


def _make_signal(fs=256, duration=4, left_alpha=10.0, right_alpha=10.0,
                 noise_amp=2.0):
    """4-channel EEG with controllable left/right alpha amplitudes."""
    t = np.arange(int(fs * duration)) / fs
    signals = []
    # ch0=TP9 (left), ch1=AF7 (left), ch2=AF8 (right), ch3=TP10 (right)
    for ch in range(4):
        if ch in (0, 1):
            amp = left_alpha
        else:
            amp = right_alpha
        alpha = amp * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        theta = 5 * np.sin(2 * np.pi * 6 * t + ch * 0.5)
        noise = noise_amp * np.random.randn(len(t))
        signals.append(alpha + theta + noise)
    return np.array(signals)


class TestBaseline:
    def test_set_baseline(self, monitor):
        np.random.seed(42)
        result = monitor.set_baseline(_make_signal())
        assert result["baseline_set"] is True
        assert "baseline_asymmetries" in result

    def test_baseline_has_all_bands(self, monitor):
        np.random.seed(42)
        result = monitor.set_baseline(_make_signal())
        for band in ("delta", "theta", "alpha", "beta"):
            assert band in result["baseline_asymmetries"]


class TestAssess:
    def test_output_keys(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        expected = {"asymmetries", "balance_score", "dominance",
                    "valence_proxy", "has_baseline"}
        assert expected.issubset(set(result.keys()))

    def test_balance_range(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        assert 0 <= result["balance_score"] <= 100

    def test_dominance_valid(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        assert result["dominance"] in ("left_dominant", "right_dominant", "balanced")

    def test_valence_range(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        assert -1 <= result["valence_proxy"] <= 1

    def test_asymmetries_all_bands(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        for band in ("delta", "theta", "alpha", "beta"):
            assert band in result["asymmetries"]


class TestHemisphericDominance:
    def test_left_dominant_high_right_alpha(self, monitor):
        """More right alpha = left hemisphere more active."""
        np.random.seed(42)
        result = monitor.assess(_make_signal(left_alpha=5, right_alpha=20))
        assert result["dominance"] == "left_dominant"
        assert result["valence_proxy"] > 0

    def test_right_dominant_high_left_alpha(self, monitor):
        """More left alpha = right hemisphere more active."""
        np.random.seed(42)
        result = monitor.assess(_make_signal(left_alpha=20, right_alpha=5))
        assert result["dominance"] == "right_dominant"
        assert result["valence_proxy"] < 0

    def test_balanced_equal_alpha(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal(left_alpha=10, right_alpha=10))
        assert result["dominance"] == "balanced"


class TestDeviations:
    def test_deviations_with_baseline(self, monitor):
        np.random.seed(42)
        monitor.set_baseline(_make_signal(left_alpha=10, right_alpha=10))
        result = monitor.assess(_make_signal(left_alpha=5, right_alpha=15))
        assert result["deviations_from_baseline"] is not None
        assert "alpha" in result["deviations_from_baseline"]

    def test_no_deviations_without_baseline(self, monitor):
        np.random.seed(42)
        result = monitor.assess(_make_signal())
        assert result["deviations_from_baseline"] is None


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
        assert "mean_balance" in stats
        assert "dominant_pattern" in stats


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
        monitor.set_baseline(_make_signal())
        monitor.assess(_make_signal())
        monitor.reset()
        assert monitor.get_history() == []
        stats = monitor.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False
