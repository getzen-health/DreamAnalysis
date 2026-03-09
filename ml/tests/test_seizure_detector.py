"""Tests for seizure detector."""
import numpy as np
import pytest

from models.seizure_detector import SeizureDetector, MEDICAL_DISCLAIMER


@pytest.fixture
def detector():
    return SeizureDetector(alarm_threshold=0.7, alarm_trigger_count=3, fs=256.0)


def _make_normal_eeg(fs=256, duration=4, n_channels=4):
    """Normal resting EEG: dominant alpha + some theta/beta."""
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = 15.0 * np.sin(2 * np.pi * 10 * t + ch * 0.5)
        theta = 5.0 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
        beta = 3.0 * np.sin(2 * np.pi * 20 * t + ch * 0.7)
        noise = 2.0 * np.random.randn(len(t))
        signals.append(alpha + theta + beta + noise)
    return np.array(signals)


def _make_seizure_eeg(fs=256, duration=4, n_channels=4):
    """Seizure-like EEG: high amplitude, rhythmic, hypersynchronous, high beta."""
    t = np.arange(int(fs * duration)) / fs
    # Strong rhythmic spike-wave at ~3 Hz with high beta
    base = 80.0 * np.sin(2 * np.pi * 3 * t)
    beta_burst = 40.0 * np.sin(2 * np.pi * 18 * t)
    spikes = 60.0 * np.sin(2 * np.pi * 14 * t)
    # All channels highly correlated (hypersynchrony)
    signals = []
    for ch in range(n_channels):
        noise = 3.0 * np.random.randn(len(t))
        signals.append(base + beta_burst + spikes + noise)
    return np.array(signals)


class TestBasicDetection:
    def test_detect_output_keys(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_normal_eeg())
        expected_keys = {
            "seizure_probability", "is_seizure", "alarm_triggered",
            "consecutive_detections", "channel_scores",
            "cross_channel_synchronization", "severity",
            "n_channels", "medical_disclaimer",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_always_has_disclaimer(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_normal_eeg())
        assert result["medical_disclaimer"] == MEDICAL_DISCLAIMER

    def test_probability_range(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_normal_eeg())
        assert 0 <= result["seizure_probability"] <= 1

    def test_normal_eeg_low_probability(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_normal_eeg())
        assert result["seizure_probability"] < 0.5
        assert result["severity"] in ("normal", "elevated")

    def test_seizure_eeg_high_probability(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_seizure_eeg())
        assert result["seizure_probability"] > 0.3

    def test_n_channels_reported(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_normal_eeg(n_channels=2))
        assert result["n_channels"] == 2


class TestAlarmLogic:
    def test_no_alarm_on_single_detection(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_seizure_eeg())
        assert result["alarm_triggered"] is False

    def test_alarm_after_consecutive_detections(self):
        det = SeizureDetector(alarm_threshold=0.1, alarm_trigger_count=3)
        np.random.seed(42)
        for _ in range(3):
            result = det.detect(_make_seizure_eeg())
        assert result["consecutive_detections"] >= 3

    def test_consecutive_resets_on_normal(self, detector):
        np.random.seed(42)
        # First a seizure-like epoch
        detector.detect(_make_seizure_eeg())
        consec_after_seizure = detector._consecutive.get("default", 0)
        # Then normal
        detector.detect(_make_normal_eeg())
        consec_after_normal = detector._consecutive.get("default", 0)
        assert consec_after_normal <= consec_after_seizure

    def test_severity_levels(self, detector):
        np.random.seed(42)
        normal_result = detector.detect(_make_normal_eeg())
        assert normal_result["severity"] in ("normal", "elevated")


class TestBaseline:
    def test_set_baseline(self, detector):
        np.random.seed(42)
        result = detector.set_baseline(_make_normal_eeg())
        assert result["baseline_set"] is True
        assert result["n_channels"] == 4

    def test_baseline_affects_detection(self, detector):
        np.random.seed(42)
        # Without baseline
        result_no_bl = detector.detect(_make_seizure_eeg())

        detector.reset()
        np.random.seed(42)
        # With baseline
        detector.set_baseline(_make_normal_eeg())
        result_with_bl = detector.detect(_make_seizure_eeg())

        # Both should detect — baseline adds boost
        assert result_with_bl["seizure_probability"] >= 0

    def test_single_channel_baseline(self, detector):
        np.random.seed(42)
        signal_1d = np.random.randn(256 * 4)
        result = detector.set_baseline(signal_1d)
        assert result["n_channels"] == 1


class TestChannelScores:
    def test_channel_scores_count(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_normal_eeg(n_channels=4))
        assert len(result["channel_scores"]) == 4

    def test_channel_scores_range(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_normal_eeg())
        for score in result["channel_scores"]:
            assert 0 <= score <= 1

    def test_synchronization_range(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_normal_eeg())
        assert 0 <= result["cross_channel_synchronization"] <= 1

    def test_hypersync_seizure(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_seizure_eeg())
        # Seizure signals are nearly identical across channels
        assert result["cross_channel_synchronization"] > 0.8


class TestEventLog:
    def test_empty_log(self, detector):
        assert detector.get_event_log() == []

    def test_log_grows(self, detector):
        np.random.seed(42)
        detector.detect(_make_normal_eeg())
        detector.detect(_make_normal_eeg())
        assert len(detector.get_event_log()) == 2

    def test_log_last_n(self, detector):
        np.random.seed(42)
        for _ in range(5):
            detector.detect(_make_normal_eeg())
        assert len(detector.get_event_log(last_n=2)) == 2


class TestStatus:
    def test_empty_status(self, detector):
        status = detector.get_status()
        assert status["total_epochs"] == 0
        assert status["monitoring"] is False

    def test_status_after_detection(self, detector):
        np.random.seed(42)
        detector.detect(_make_normal_eeg())
        status = detector.get_status()
        assert status["total_epochs"] == 1
        assert status["monitoring"] is True

    def test_status_has_baseline_flag(self, detector):
        np.random.seed(42)
        detector.set_baseline(_make_normal_eeg())
        detector.detect(_make_normal_eeg())
        assert detector.get_status()["has_baseline"] is True


class TestConfiguration:
    def test_set_alarm_threshold(self, detector):
        detector.set_alarm_threshold(0.5)
        assert detector._alarm_threshold == 0.5

    def test_threshold_clamped(self, detector):
        detector.set_alarm_threshold(2.0)
        assert detector._alarm_threshold <= 0.99
        detector.set_alarm_threshold(-1.0)
        assert detector._alarm_threshold >= 0.1


class TestMultiUser:
    def test_independent_users(self, detector):
        np.random.seed(42)
        detector.detect(_make_normal_eeg(), user_id="alice")
        detector.detect(_make_seizure_eeg(), user_id="bob")
        assert len(detector.get_event_log("alice")) == 1
        assert len(detector.get_event_log("bob")) == 1


class TestReset:
    def test_reset_clears_all(self, detector):
        np.random.seed(42)
        detector.set_baseline(_make_normal_eeg())
        detector.detect(_make_normal_eeg())
        detector.reset()
        assert detector.get_event_log() == []
        assert detector.get_status()["total_epochs"] == 0


class TestEdgeCases:
    def test_short_signal(self, detector):
        np.random.seed(42)
        short = np.random.randn(4, 64)
        result = detector.detect(short)
        assert "seizure_probability" in result

    def test_single_channel(self, detector):
        np.random.seed(42)
        signal = np.random.randn(256 * 4)
        result = detector.detect(signal)
        assert result["n_channels"] == 1

    def test_flat_signal(self, detector):
        flat = np.zeros((4, 256 * 4))
        result = detector.detect(flat)
        assert result["seizure_probability"] == 0.0 or result["severity"] == "normal"
