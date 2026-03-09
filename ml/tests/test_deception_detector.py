"""Tests for deception detector."""
import numpy as np
import pytest

from models.deception_detector import DeceptionDetector, DISCLAIMER


@pytest.fixture
def detector():
    return DeceptionDetector(fs=256.0)


def _make_probe_epoch(fs=256, duration_ms=800):
    """Probe epoch: larger P300 component (250-500ms)."""
    n = int(fs * duration_ms / 1000)
    t = np.arange(n) / fs
    signal = np.random.randn(n) * 2
    # Add large P300 at 300-400ms
    p300_start = int(0.3 * fs)
    p300_end = int(0.4 * fs)
    signal[p300_start:p300_end] += 10.0  # large positive deflection
    return signal


def _make_irrelevant_epoch(fs=256, duration_ms=800):
    """Irrelevant epoch: smaller P300."""
    n = int(fs * duration_ms / 1000)
    signal = np.random.randn(n) * 2
    # Small P300
    p300_start = int(0.3 * fs)
    p300_end = int(0.4 * fs)
    signal[p300_start:p300_end] += 1.0
    return signal


def _make_multichannel_epoch(fs=256, duration_ms=800, n_channels=4, p300_amp=5.0):
    """Multichannel epoch with P300 on channel 2 (AF8)."""
    n = int(fs * duration_ms / 1000)
    signals = np.random.randn(n_channels, n) * 2
    p300_start = int(0.3 * fs)
    p300_end = int(0.4 * fs)
    signals[2, p300_start:p300_end] += p300_amp
    return signals


class TestBasicDetection:
    def test_output_keys(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        expected = {
            "deception_score", "deception_detected", "p300_difference",
            "probe_p300_amplitude", "irrelevant_p300_amplitude",
            "confidence", "disclaimer",
        }
        assert expected.issubset(set(result.keys()))

    def test_always_has_disclaimer(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        assert result["disclaimer"] == DISCLAIMER

    def test_deception_score_range(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        assert 0 <= result["deception_score"] <= 1

    def test_probe_larger_than_irrelevant(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        assert result["probe_p300_amplitude"] > result["irrelevant_p300_amplitude"]

    def test_positive_p300_difference(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        assert result["p300_difference"] > 0


class TestDeceptionDetection:
    def test_large_p300_diff_detected(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        assert result["deception_detected"] is True

    def test_similar_epochs_not_detected(self, detector):
        np.random.seed(42)
        irr1 = _make_irrelevant_epoch()
        np.random.seed(43)
        irr2 = _make_irrelevant_epoch()
        result = detector.detect(irr1, irr2)
        # Similar epochs should have low deception score
        assert result["deception_score"] < 0.7

    def test_confidence_levels(self, detector):
        np.random.seed(42)
        result = detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        assert result["confidence"] in ("high", "medium", "low")


class TestMultichannelInput:
    def test_multichannel_probe(self, detector):
        np.random.seed(42)
        probe = _make_multichannel_epoch(p300_amp=10.0)
        irrel = _make_multichannel_epoch(p300_amp=1.0)
        result = detector.detect(probe, irrel, channel_idx=2)
        assert result["p300_difference"] > 0

    def test_channel_selection(self, detector):
        np.random.seed(42)
        probe = _make_multichannel_epoch(p300_amp=10.0)
        irrel = _make_multichannel_epoch(p300_amp=1.0)
        # P300 is on channel 2 — result should detect it
        result = detector.detect(probe, irrel, channel_idx=2)
        assert result["deception_score"] > 0.5


class TestAverageDetection:
    def test_average_output_keys(self, detector):
        np.random.seed(42)
        probes = [_make_probe_epoch() for _ in range(5)]
        irrels = [_make_irrelevant_epoch() for _ in range(5)]
        result = detector.detect_average(probes, irrels)
        assert "n_probe_trials" in result
        assert "n_irrelevant_trials" in result
        assert "snr_improvement" in result

    def test_average_improves_detection(self, detector):
        np.random.seed(42)
        probes = [_make_probe_epoch() for _ in range(10)]
        irrels = [_make_irrelevant_epoch() for _ in range(10)]
        result = detector.detect_average(probes, irrels)
        assert result["snr_improvement"] > 1.0

    def test_empty_epochs(self, detector):
        result = detector.detect_average([], [])
        assert result["deception_detected"] is False
        assert result["n_probe_trials"] == 0

    def test_trial_counts(self, detector):
        np.random.seed(42)
        probes = [_make_probe_epoch() for _ in range(7)]
        irrels = [_make_irrelevant_epoch() for _ in range(5)]
        result = detector.detect_average(probes, irrels)
        assert result["n_probe_trials"] == 7
        assert result["n_irrelevant_trials"] == 5


class TestHistory:
    def test_empty_history(self, detector):
        assert detector.get_history() == []

    def test_history_grows(self, detector):
        np.random.seed(42)
        detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        assert len(detector.get_history()) == 2

    def test_history_last_n(self, detector):
        np.random.seed(42)
        for _ in range(5):
            detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        assert len(detector.get_history(last_n=2)) == 2


class TestSummary:
    def test_empty_summary(self, detector):
        assert detector.get_summary()["n_trials"] == 0

    def test_summary_with_data(self, detector):
        np.random.seed(42)
        for _ in range(5):
            detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        s = detector.get_summary()
        assert s["n_trials"] == 5
        assert "mean_deception_score" in s
        assert "detection_rate" in s


class TestMultiUser:
    def test_independent_users(self, detector):
        np.random.seed(42)
        detector.detect(_make_probe_epoch(), _make_irrelevant_epoch(), user_id="a")
        detector.detect(_make_probe_epoch(), _make_irrelevant_epoch(), user_id="b")
        assert len(detector.get_history("a")) == 1
        assert len(detector.get_history("b")) == 1


class TestReset:
    def test_reset_clears(self, detector):
        np.random.seed(42)
        detector.detect(_make_probe_epoch(), _make_irrelevant_epoch())
        detector.reset()
        assert detector.get_summary()["n_trials"] == 0


class TestEdgeCases:
    def test_very_short_epoch(self, detector):
        np.random.seed(42)
        short = np.random.randn(32)
        result = detector.detect(short, short)
        assert "deception_score" in result

    def test_identical_epochs(self, detector):
        np.random.seed(42)
        epoch = _make_irrelevant_epoch()
        result = detector.detect(epoch, epoch.copy())
        assert abs(result["p300_difference"]) < 0.1
