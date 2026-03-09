"""Tests for eye state detector."""
import numpy as np
import pytest

from models.eye_state_detector import EyeStateDetector


@pytest.fixture
def detector():
    return EyeStateDetector(fs=256.0, blink_threshold=75.0)


def _make_signal(fs=256, duration=4, n_channels=4, alpha_amp=10.0, noise_amp=2.0):
    """EEG with controllable alpha amplitude."""
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        noise = noise_amp * np.random.randn(len(t))
        signals.append(alpha + noise)
    return np.array(signals)


def _make_blink_signal(fs=256, duration=4, n_channels=4, n_blinks=3):
    """EEG with embedded blink artifacts."""
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        sig = 5.0 * np.sin(2 * np.pi * 10 * t) + 2.0 * np.random.randn(len(t))
        # Add blinks at frontal channels
        if ch in (1, 2):
            for i in range(n_blinks):
                blink_center = int(fs * (0.5 + i * 1.0))
                if blink_center < len(sig) - 50:
                    blink = 150 * np.exp(-0.5 * ((np.arange(100) - 50) / 15) ** 2)
                    sig[blink_center:blink_center + 100] += blink
        signals.append(sig)
    return np.array(signals)


class TestCalibrate:
    def test_calibration(self, detector):
        np.random.seed(42)
        eo = _make_signal(alpha_amp=5)
        ec = _make_signal(alpha_amp=20)
        result = detector.calibrate(eo, ec)
        assert result["calibrated"] is True
        assert result["alpha_reactivity"] > 1.0

    def test_calibration_stores_threshold(self, detector):
        np.random.seed(42)
        eo = _make_signal(alpha_amp=5)
        ec = _make_signal(alpha_amp=20)
        detector.calibrate(eo, ec)
        result = detector.detect(_make_signal(alpha_amp=20))
        assert result["has_calibration"] is True

    def test_single_channel(self, detector):
        np.random.seed(42)
        eo = 5 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        ec = 20 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        result = detector.calibrate(eo, ec)
        assert result["calibrated"] is True


class TestDetect:
    def test_output_keys(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_signal())
        expected = {"eye_state", "alpha_power", "blink_detected", "blink_count",
                    "confidence", "has_calibration"}
        assert expected.issubset(set(result.keys()))

    def test_state_valid(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_signal())
        assert result["eye_state"] in ("open", "closed", "blink")

    def test_confidence_range(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_signal())
        assert 0 <= result["confidence"] <= 1

    def test_eyes_closed_high_alpha(self, detector):
        np.random.seed(42)
        detector.calibrate(_make_signal(alpha_amp=5), _make_signal(alpha_amp=25))
        result = detector.detect(_make_signal(alpha_amp=25))
        assert result["eye_state"] == "closed"

    def test_eyes_open_low_alpha(self, detector):
        np.random.seed(42)
        detector.calibrate(_make_signal(alpha_amp=5), _make_signal(alpha_amp=25))
        result = detector.detect(_make_signal(alpha_amp=5))
        assert result["eye_state"] == "open"


class TestBlinks:
    def test_blink_detected(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_blink_signal(n_blinks=3))
        assert result["blink_detected"] is True
        assert result["blink_count"] > 0

    def test_no_blinks_in_clean_signal(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_signal(alpha_amp=10, noise_amp=1))
        assert result["blink_count"] == 0

    def test_blink_state(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_blink_signal(n_blinks=5))
        if result["blink_count"] > 0:
            assert result["eye_state"] == "blink"


class TestAlphaReactivity:
    def test_high_reactivity(self, detector):
        np.random.seed(42)
        result = detector.calibrate(
            _make_signal(alpha_amp=3), _make_signal(alpha_amp=20)
        )
        assert result["alpha_reactivity"] > 2.0

    def test_low_reactivity(self, detector):
        np.random.seed(42)
        result = detector.calibrate(
            _make_signal(alpha_amp=10), _make_signal(alpha_amp=12)
        )
        assert result["alpha_reactivity"] < 2.0


class TestSessionStats:
    def test_empty_stats(self, detector):
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_after_data(self, detector):
        np.random.seed(42)
        for _ in range(5):
            detector.detect(_make_signal())
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "state_distribution" in stats
        assert "total_blinks" in stats


class TestHistory:
    def test_empty_history(self, detector):
        assert detector.get_history() == []

    def test_history_grows(self, detector):
        np.random.seed(42)
        detector.detect(_make_signal())
        detector.detect(_make_signal())
        assert len(detector.get_history()) == 2

    def test_history_last_n(self, detector):
        np.random.seed(42)
        for _ in range(10):
            detector.detect(_make_signal())
        assert len(detector.get_history(last_n=3)) == 3


class TestMultiUser:
    def test_independent(self, detector):
        np.random.seed(42)
        detector.detect(_make_signal(), user_id="a")
        detector.detect(_make_signal(), user_id="b")
        assert len(detector.get_history("a")) == 1
        assert len(detector.get_history("b")) == 1


class TestReset:
    def test_reset_clears(self, detector):
        np.random.seed(42)
        detector.calibrate(_make_signal(), _make_signal())
        detector.detect(_make_signal())
        detector.reset()
        assert detector.get_history() == []
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_calibration"] is False
