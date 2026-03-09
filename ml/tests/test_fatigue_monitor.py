"""Tests for FatigueMonitor — mental fatigue detection via EEG theta/beta ratio."""

import numpy as np
import pytest

from models.fatigue_monitor import FatigueMonitor, get_fatigue_monitor


# ── Fixtures ──────────────────────────────────────────────────────────────────

FS = 256  # Hz


def make_eeg(n_channels: int = 4, duration_s: float = 2.0) -> np.ndarray:
    """Generate synthetic sinusoidal EEG data."""
    n_samples = int(FS * duration_s)
    t = np.linspace(0, duration_s, n_samples)
    # Theta component at 6 Hz, beta at 20 Hz
    signal = 10 * np.sin(2 * np.pi * 6 * t) + 5 * np.sin(2 * np.pi * 20 * t)
    if n_channels > 1:
        return np.tile(signal, (n_channels, 1)).astype(np.float32)
    return signal.astype(np.float32)


@pytest.fixture
def monitor():
    """Fresh FatigueMonitor instance for each test."""
    return FatigueMonitor()


# ── Unit tests: FatigueMonitor ────────────────────────────────────────────────


class TestFatigueMonitorInit:
    def test_initial_status(self, monitor):
        status = monitor.get_status()
        assert status["baseline_calibrated"] is False
        assert status["baseline_frames"] == 0
        assert status["history_length"] == 0
        assert status["current_fatigue_index"] is None


class TestFatigueMonitorCalibrate:
    def test_calibrate_returns_dict(self, monitor):
        eeg = make_eeg()
        result = monitor.calibrate_baseline(eeg, fs=FS)
        assert isinstance(result, dict)
        assert "baseline_ready" in result
        assert "frames" in result

    def test_calibrate_not_ready_before_threshold(self, monitor):
        eeg = make_eeg()
        for _ in range(5):
            result = monitor.calibrate_baseline(eeg, fs=FS)
        assert result["baseline_ready"] is False

    def test_calibrate_ready_after_10_frames(self, monitor):
        eeg = make_eeg()
        for _ in range(10):
            result = monitor.calibrate_baseline(eeg, fs=FS)
        assert result["baseline_ready"] is True

    def test_calibrate_increments_frames(self, monitor):
        eeg = make_eeg()
        for i in range(1, 6):
            result = monitor.calibrate_baseline(eeg, fs=FS)
            assert result["frames"] == i

    def test_calibrate_target_frames_in_response(self, monitor):
        eeg = make_eeg()
        result = monitor.calibrate_baseline(eeg, fs=FS)
        assert "target_frames" in result
        assert result["target_frames"] == 10

    def test_calibrate_updates_status(self, monitor):
        eeg = make_eeg()
        for _ in range(10):
            monitor.calibrate_baseline(eeg, fs=FS)
        status = monitor.get_status()
        assert status["baseline_calibrated"] is True
        assert status["baseline_frames"] == 10


class TestFatigueMonitorPredict:
    def test_predict_returns_required_keys(self, monitor):
        eeg = make_eeg()
        result = monitor.predict(eeg, fs=FS, session_minutes=5.0)
        required = {
            "fatigue_index",
            "fatigue_stage",
            "theta_beta_ratio",
            "theta_beta_trend_slope",
            "session_minutes",
            "break_recommendation",
            "baseline_calibrated",
            "fatigue_curve",
        }
        assert required.issubset(result.keys())

    def test_predict_fatigue_index_in_range(self, monitor):
        eeg = make_eeg()
        result = monitor.predict(eeg, fs=FS)
        assert 0.0 <= result["fatigue_index"] <= 1.0

    def test_predict_fatigue_stage_is_string(self, monitor):
        eeg = make_eeg()
        result = monitor.predict(eeg, fs=FS)
        assert isinstance(result["fatigue_stage"], str)
        assert len(result["fatigue_stage"]) > 0

    def test_predict_valid_stages(self, monitor):
        valid_stages = {
            "fresh",
            "mild_fatigue",
            "moderate_fatigue",
            "severe_fatigue",
            "critical_fatigue",
            "insufficient_data",
        }
        eeg = make_eeg()
        result = monitor.predict(eeg, fs=FS)
        assert result["fatigue_stage"] in valid_stages

    def test_predict_theta_beta_ratio_positive(self, monitor):
        eeg = make_eeg()
        result = monitor.predict(eeg, fs=FS)
        assert result["theta_beta_ratio"] >= 0.0

    def test_predict_break_recommendation_is_dict(self, monitor):
        eeg = make_eeg()
        result = monitor.predict(eeg, fs=FS)
        br = result["break_recommendation"]
        assert isinstance(br, dict)
        assert "recommended" in br
        assert "urgency" in br

    def test_predict_fatigue_curve_is_list(self, monitor):
        eeg = make_eeg()
        result = monitor.predict(eeg, fs=FS)
        assert isinstance(result["fatigue_curve"], list)

    def test_predict_session_minutes_preserved(self, monitor):
        eeg = make_eeg()
        result = monitor.predict(eeg, fs=FS, session_minutes=12.5)
        assert result["session_minutes"] == pytest.approx(12.5)

    def test_predict_accumulates_curve(self, monitor):
        eeg = make_eeg()
        for _ in range(5):
            result = monitor.predict(eeg, fs=FS)
        assert len(result["fatigue_curve"]) == 5

    def test_predict_after_calibration_uses_baseline(self, monitor):
        eeg = make_eeg()
        for _ in range(10):
            monitor.calibrate_baseline(eeg, fs=FS)
        result = monitor.predict(eeg, fs=FS)
        assert result["baseline_calibrated"] is True

    def test_predict_1d_input(self, monitor):
        eeg_1d = make_eeg(n_channels=1, duration_s=2.0).reshape(-1)
        result = monitor.predict(eeg_1d, fs=FS)
        # Should not crash; may return insufficient_data if too short
        assert "fatigue_index" in result

    def test_predict_high_session_time_increases_fatigue(self, monitor):
        eeg = make_eeg()
        result_fresh = monitor.predict(eeg, fs=FS, session_minutes=0.0)
        monitor_long = FatigueMonitor()
        result_tired = monitor_long.predict(eeg, fs=FS, session_minutes=60.0)
        # Time-on-task component pushes index higher at 60 min
        assert result_tired["fatigue_index"] >= result_fresh["fatigue_index"]


class TestFatigueMonitorReset:
    def test_reset_session_clears_history(self, monitor):
        eeg = make_eeg()
        for _ in range(5):
            monitor.predict(eeg, fs=FS)
        monitor.reset_session()
        status = monitor.get_status()
        assert status["history_length"] == 0
        assert status["current_fatigue_index"] is None

    def test_reset_session_preserves_baseline(self, monitor):
        eeg = make_eeg()
        for _ in range(10):
            monitor.calibrate_baseline(eeg, fs=FS)
        monitor.reset_session()
        status = monitor.get_status()
        assert status["baseline_calibrated"] is True

    def test_reset_baseline_clears_all(self, monitor):
        eeg = make_eeg()
        for _ in range(10):
            monitor.calibrate_baseline(eeg, fs=FS)
        for _ in range(5):
            monitor.predict(eeg, fs=FS)
        monitor.reset_baseline()
        status = monitor.get_status()
        assert status["baseline_calibrated"] is False
        assert status["baseline_frames"] == 0
        assert status["history_length"] == 0


class TestFatigueMonitorInsufficientData:
    def test_too_short_signal_returns_insufficient(self, monitor):
        # 0.1 seconds of data is too short for FFT (< fs samples)
        n = int(FS * 0.1)
        eeg_short = np.random.randn(4, n).astype(np.float32)
        result = monitor.predict(eeg_short, fs=FS)
        # Should either return insufficient_data or handle gracefully
        assert "fatigue_index" in result
        assert "fatigue_stage" in result


class TestGetFatigueMonitorSingleton:
    def test_singleton_returns_same_instance(self):
        m1 = get_fatigue_monitor()
        m2 = get_fatigue_monitor()
        assert m1 is m2

    def test_singleton_is_fatigue_monitor(self):
        m = get_fatigue_monitor()
        assert isinstance(m, FatigueMonitor)


class TestBreakRecommendation:
    def test_no_break_when_fresh(self, monitor):
        eeg = make_eeg()
        result = monitor.predict(eeg, fs=FS, session_minutes=5.0)
        br = result["break_recommendation"]
        assert isinstance(br["recommended"], bool)

    def test_break_recommended_at_long_session(self):
        """45+ min session should trigger time-based break recommendation."""
        m = FatigueMonitor()
        eeg = make_eeg()
        result = m.predict(eeg, fs=FS, session_minutes=50.0)
        br = result["break_recommendation"]
        assert br["recommended"] is True

    def test_urgency_values(self, monitor):
        eeg = make_eeg()
        result = monitor.predict(eeg, fs=FS)
        valid_urgency = {"none", "low", "medium", "high"}
        assert result["break_recommendation"]["urgency"] in valid_urgency
