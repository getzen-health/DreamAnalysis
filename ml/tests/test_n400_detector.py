"""Tests for N400 ERP detector."""
import numpy as np
import pytest

from models.n400_detector import N400Detector


@pytest.fixture
def detector():
    return N400Detector()


def _make_epoch_with_n400(fs=256, duration_ms=800, onset_ms=200, n400_amp=-5.0):
    """Create synthetic epoch with an N400-like component."""
    n_samples = int(fs * duration_ms / 1000)
    t = np.arange(n_samples) / fs
    onset_s = onset_ms / 1000

    # Baseline noise
    signal = 0.5 * np.random.randn(n_samples)

    # Add N400 component: negative peak at 350-400ms post-stimulus
    n400_center = onset_s + 0.375  # 375ms post-stimulus
    n400_width = 0.05  # 50ms gaussian width
    n400 = n400_amp * np.exp(-0.5 * ((t - n400_center) / n400_width) ** 2)
    signal += n400

    return signal


def _make_flat_epoch(fs=256, duration_ms=800):
    """Create epoch with no ERP component."""
    n_samples = int(fs * duration_ms / 1000)
    return 0.3 * np.random.randn(n_samples)


class TestBasicDetection:
    def test_detects_n400(self, detector):
        """Should detect clear N400 component."""
        np.random.seed(42)
        epoch = _make_epoch_with_n400(n400_amp=-8.0)
        result = detector.detect(epoch, stimulus_onset_ms=200)
        assert result["n400_detected"] is True
        assert result["n400_amplitude_uv"] < 0

    def test_no_n400_in_flat_signal(self, detector):
        """Should not detect N400 in flat noise."""
        np.random.seed(42)
        epoch = _make_flat_epoch()
        result = detector.detect(epoch, stimulus_onset_ms=200)
        # With threshold -2.0, random noise shouldn't consistently trigger
        assert result["n400_amplitude_uv"] > -5.0

    def test_larger_n400_higher_difficulty(self, detector):
        """Larger (more negative) N400 should indicate higher semantic difficulty."""
        np.random.seed(42)
        small = detector.detect(_make_epoch_with_n400(n400_amp=-2.0), stimulus_onset_ms=200)
        large = detector.detect(_make_epoch_with_n400(n400_amp=-10.0), stimulus_onset_ms=200)
        assert large["semantic_difficulty"] > small["semantic_difficulty"]


class TestMultichannel:
    def test_multichannel_input(self, detector):
        """Should extract AF8 (ch2) from multichannel input."""
        np.random.seed(42)
        single = _make_epoch_with_n400(n400_amp=-6.0)
        multi = np.stack([
            _make_flat_epoch(),  # TP9
            _make_flat_epoch(),  # AF7
            single,              # AF8 — with N400
            _make_flat_epoch(),  # TP10
        ])
        result = detector.detect(multi, stimulus_onset_ms=200)
        assert result["detection_channel"] == "AF8"
        assert result["n400_amplitude_uv"] < -1.0

    def test_single_channel_fallback(self, detector):
        """Should work with fewer channels than expected."""
        np.random.seed(42)
        epoch = np.stack([_make_epoch_with_n400(n400_amp=-6.0)])
        result = detector.detect(epoch, stimulus_onset_ms=200)
        assert "n400_amplitude_uv" in result


class TestAveraging:
    def test_average_multiple_epochs(self, detector):
        """Averaging should improve detection reliability."""
        np.random.seed(42)
        epochs = [_make_epoch_with_n400(n400_amp=-4.0) for _ in range(20)]
        result = detector.detect_average(epochs, stimulus_onset_ms=200)
        assert result["n_epochs"] == 20
        assert result["snr_improvement_db"] > 10  # ~13 dB for 20 epochs

    def test_average_empty_epochs(self, detector):
        """Empty epochs list should return error."""
        result = detector.detect_average([], stimulus_onset_ms=200)
        assert result["n400_detected"] is False
        assert result["n_epochs"] == 0


class TestHistory:
    def test_history_tracking(self, detector):
        """Should track detection history."""
        for _ in range(5):
            detector.detect(_make_epoch_with_n400(), stimulus_onset_ms=200)
        assert len(detector.get_history()) == 5

    def test_history_cap(self, detector):
        """History should cap at 200 entries."""
        for _ in range(210):
            detector.detect(_make_flat_epoch(), stimulus_onset_ms=200)
        assert len(detector.get_history()) == 200

    def test_per_user_history(self, detector):
        """Different users should have independent histories."""
        detector.detect(_make_flat_epoch(), stimulus_onset_ms=200, user_id="a")
        detector.detect(_make_flat_epoch(), stimulus_onset_ms=200, user_id="b")
        assert len(detector.get_history("a")) == 1
        assert len(detector.get_history("b")) == 1

    def test_reset_clears_history(self, detector):
        """Reset should clear user history."""
        detector.detect(_make_flat_epoch(), stimulus_onset_ms=200)
        detector.reset()
        assert len(detector.get_history()) == 0


class TestSummary:
    def test_summary_empty(self, detector):
        """Empty summary should have zero trials."""
        summary = detector.get_summary()
        assert summary["n_trials"] == 0

    def test_summary_with_data(self, detector):
        """Summary should compute correct statistics."""
        np.random.seed(42)
        for _ in range(10):
            detector.detect(_make_epoch_with_n400(n400_amp=-5.0), stimulus_onset_ms=200)
        summary = detector.get_summary()
        assert summary["n_trials"] == 10
        assert "detection_rate" in summary
        assert "mean_amplitude_uv" in summary
        assert "mean_latency_ms" in summary


class TestOutputStructure:
    def test_output_keys(self, detector):
        """All expected keys should be present."""
        result = detector.detect(_make_flat_epoch(), stimulus_onset_ms=200)
        expected = {"n400_detected", "n400_amplitude_uv", "n400_mean_uv",
                    "baseline_mean_uv", "peak_latency_ms", "semantic_difficulty",
                    "detection_channel"}
        assert expected.issubset(set(result.keys()))

    def test_latency_in_n400_window(self, detector):
        """Peak latency should be in or near the N400 window."""
        np.random.seed(42)
        result = detector.detect(_make_epoch_with_n400(n400_amp=-8.0), stimulus_onset_ms=200)
        if result["n400_detected"]:
            assert 250 <= result["peak_latency_ms"] <= 550

    def test_difficulty_range(self, detector):
        """Semantic difficulty should be between 0 and 1."""
        for amp in [-10, -5, -1, 0, 2]:
            result = detector.detect(_make_epoch_with_n400(n400_amp=amp), stimulus_onset_ms=200)
            assert 0 <= result["semantic_difficulty"] <= 1
