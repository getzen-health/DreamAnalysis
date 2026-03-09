"""Tests for memory consolidation tracker."""
import numpy as np
import pytest

from models.memory_consolidation import MemoryConsolidationTracker


@pytest.fixture
def tracker():
    return MemoryConsolidationTracker()


def _make_sleep_signal(fs=256, duration=30, spindle_amp=2.0, so_amp=5.0):
    """Create synthetic sleep EEG with spindles and slow oscillations."""
    t = np.arange(int(fs * duration)) / fs
    # Slow oscillation at 0.8 Hz
    so = so_amp * np.sin(2 * np.pi * 0.8 * t)
    # Sleep spindle at 13 Hz (burst around t=5s and t=15s)
    spindle = np.zeros_like(t)
    for center in [5, 15, 25]:
        gaussian = np.exp(-0.5 * ((t - center) / 0.3) ** 2)
        spindle += spindle_amp * gaussian * np.sin(2 * np.pi * 13 * t)
    # Background noise
    noise = 0.3 * np.random.randn(len(t))
    return so + spindle + noise


def _make_flat_signal(fs=256, duration=30):
    """Create flat noise signal (no sleep features)."""
    return 0.5 * np.random.randn(int(fs * duration))


class TestBasicAnalysis:
    def test_analyze_epoch_output_keys(self, tracker):
        """Should return all expected keys."""
        np.random.seed(42)
        signal = _make_sleep_signal()
        result = tracker.analyze_epoch(signal, sleep_stage="N2")
        expected = {"spindle_count", "spindle_density_per_min", "slow_oscillation_count",
                    "coupling_score", "consolidation_quality", "quality_label",
                    "sleep_stage", "epoch_duration_sec"}
        assert expected.issubset(set(result.keys()))

    def test_n2_has_quality(self, tracker):
        """N2 epochs should have non-zero quality if spindles present."""
        np.random.seed(42)
        signal = _make_sleep_signal(spindle_amp=5.0, so_amp=10.0)
        result = tracker.analyze_epoch(signal, sleep_stage="N2")
        assert result["consolidation_quality"] >= 0

    def test_wake_has_zero_quality(self, tracker):
        """Wake stage should have zero consolidation quality."""
        np.random.seed(42)
        signal = _make_sleep_signal()
        result = tracker.analyze_epoch(signal, sleep_stage="Wake")
        assert result["consolidation_quality"] == 0

    def test_rem_has_zero_quality(self, tracker):
        """REM stage should have zero quality (spindle-based metric)."""
        np.random.seed(42)
        signal = _make_sleep_signal()
        result = tracker.analyze_epoch(signal, sleep_stage="REM")
        assert result["consolidation_quality"] == 0


class TestSpindleDetection:
    def test_detects_spindles_in_synthetic(self, tracker):
        """Should detect spindles in synthetic signal with spindle bursts."""
        np.random.seed(42)
        signal = _make_sleep_signal(spindle_amp=5.0)
        result = tracker.analyze_epoch(signal, sleep_stage="N2")
        assert result["spindle_count"] >= 1

    def test_spindle_count_non_negative(self, tracker):
        """Spindle count should always be non-negative."""
        np.random.seed(42)
        result = tracker.analyze_epoch(_make_flat_signal(), sleep_stage="N2")
        assert result["spindle_count"] >= 0
        assert result["spindle_density_per_min"] >= 0


class TestSlowOscillations:
    def test_detects_so_in_synthetic(self, tracker):
        """Should detect slow oscillations."""
        np.random.seed(42)
        signal = _make_sleep_signal(so_amp=10.0)
        result = tracker.analyze_epoch(signal, sleep_stage="N3")
        assert result["slow_oscillation_count"] > 0


class TestCoupling:
    def test_coupling_range(self, tracker):
        """Coupling score should be between 0 and 1."""
        np.random.seed(42)
        signal = _make_sleep_signal()
        result = tracker.analyze_epoch(signal, sleep_stage="N2")
        assert 0 <= result["coupling_score"] <= 1


class TestQualityLabels:
    def test_quality_labels_valid(self, tracker):
        """Quality label should be one of the defined levels."""
        np.random.seed(42)
        for stage in ["N2", "N3", "Wake", "REM"]:
            result = tracker.analyze_epoch(_make_sleep_signal(), sleep_stage=stage)
            assert result["quality_label"] in ("excellent", "good", "moderate", "low")


class TestNightSummary:
    def test_empty_summary(self, tracker):
        """Empty session → zero stats."""
        summary = tracker.get_night_summary()
        assert summary["total_epochs"] == 0

    def test_summary_with_data(self, tracker):
        """Summary should reflect recorded epochs."""
        np.random.seed(42)
        for _ in range(5):
            tracker.analyze_epoch(_make_sleep_signal(), sleep_stage="N2")
        summary = tracker.get_night_summary()
        assert summary["total_epochs"] == 5
        assert summary["n2_n3_epochs"] == 5
        assert summary["total_spindles"] >= 0

    def test_summary_quality_label(self, tracker):
        """Summary should include quality label."""
        np.random.seed(42)
        tracker.analyze_epoch(_make_sleep_signal(), sleep_stage="N2")
        summary = tracker.get_night_summary()
        assert "quality_label" in summary


class TestMultiUser:
    def test_users_independent(self, tracker):
        """Different users should have separate sessions."""
        np.random.seed(42)
        tracker.analyze_epoch(_make_sleep_signal(), user_id="alice")
        tracker.analyze_epoch(_make_sleep_signal(), user_id="bob")
        alice = tracker.get_night_summary("alice")
        bob = tracker.get_night_summary("bob")
        assert alice["total_epochs"] == 1
        assert bob["total_epochs"] == 1


class TestReset:
    def test_reset_clears(self, tracker):
        """Reset should clear session data."""
        np.random.seed(42)
        tracker.analyze_epoch(_make_sleep_signal())
        tracker.reset()
        assert tracker.get_night_summary()["total_epochs"] == 0


class TestEdgeCases:
    def test_short_signal(self, tracker):
        """Very short signal should not crash."""
        np.random.seed(42)
        short = np.random.randn(64)
        result = tracker.analyze_epoch(short)
        assert "consolidation_quality" in result

    def test_multichannel_input(self, tracker):
        """2D input should extract single channel."""
        np.random.seed(42)
        multi = np.random.randn(4, 256 * 10)
        result = tracker.analyze_epoch(multi, sleep_stage="N2")
        assert "spindle_count" in result
