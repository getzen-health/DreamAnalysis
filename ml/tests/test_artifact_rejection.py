"""Tests for MultiMethodArtifactDetector (4-channel consumer EEG)."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure ml/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.artifact_rejection import ArtifactResult, MultiMethodArtifactDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector() -> MultiMethodArtifactDetector:
    """Default detector with standard Muse 2 settings."""
    return MultiMethodArtifactDetector(fs=256.0)


@pytest.fixture
def clean_epoch() -> np.ndarray:
    """4-channel, 1-second epoch of realistic clean EEG (~10 uV RMS).

    Uses correlated channels so correlation check passes.
    Low-pass filtered to simulate real bandpass-filtered EEG (no sharp
    sample-to-sample transitions), keeping gradients well under 10 uV/ms.
    """
    from scipy.signal import butter, filtfilt

    rng = np.random.RandomState(42)
    n_channels = 4
    n_samples = 256  # 1 second at 256 Hz
    fs = 256.0
    # Base signal shared across channels (ensures correlation)
    base = rng.randn(n_samples) * 8.0
    epoch = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        # Shared base + small per-channel noise
        epoch[ch] = base + rng.randn(n_samples) * 2.0
    # Low-pass at 45 Hz to mimic real preprocessed EEG
    b, a = butter(4, 45.0 / (fs / 2), btype="low")
    for ch in range(n_channels):
        epoch[ch] = filtfilt(b, a, epoch[ch])
    return epoch


@pytest.fixture
def high_amplitude_epoch(clean_epoch: np.ndarray) -> np.ndarray:
    """Epoch with amplitude spike >75 uV on channel 0."""
    epoch = clean_epoch.copy()
    epoch[0, 128] = 200.0  # large spike
    return epoch


@pytest.fixture
def high_gradient_epoch() -> np.ndarray:
    """Epoch with a sudden jump exceeding 10 uV/ms gradient."""
    rng = np.random.RandomState(99)
    epoch = rng.randn(4, 256) * 5.0
    # Insert a step function: jump of 100 uV in one sample
    # At 256 Hz, 1 sample = ~3.9 ms, so gradient = 100/3.9 ~ 25.6 uV/ms >> 10
    epoch[1, 100] = -50.0
    epoch[1, 101] = 50.0
    return epoch


@pytest.fixture
def flat_epoch() -> np.ndarray:
    """Epoch where channel 2 is flat (disconnected electrode)."""
    rng = np.random.RandomState(7)
    epoch = rng.randn(4, 256) * 10.0
    epoch[2, :] = 0.01  # nearly constant — std << 0.5 uV
    return epoch


@pytest.fixture
def high_kurtosis_epoch() -> np.ndarray:
    """Epoch with impulsive artifacts producing high kurtosis."""
    rng = np.random.RandomState(123)
    epoch = rng.randn(4, 256) * 5.0
    # Add several large spikes to channel 3 to increase kurtosis
    spike_indices = [20, 50, 80, 110, 140, 170, 200, 230]
    for idx in spike_indices:
        epoch[3, idx] = 60.0 * (1 if idx % 2 == 0 else -1)
    return epoch


@pytest.fixture
def spectral_contamination_epoch() -> np.ndarray:
    """Epoch contaminated with broadband 25-45 Hz muscle noise.

    Amplitude kept under 75 uV so only the spectral method fires (not amplitude).
    """
    rng = np.random.RandomState(55)
    t = np.arange(256) / 256.0
    epoch = rng.randn(4, 256) * 3.0
    # Add moderate broadband noise (30 Hz + 35 Hz + 40 Hz sinusoids)
    # Each at 15 uV amplitude: combined peak ~48 uV, well under 75 uV threshold
    for freq in [30.0, 35.0, 40.0]:
        for ch in range(4):
            epoch[ch] += 15.0 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return epoch


@pytest.fixture
def low_correlation_epoch() -> np.ndarray:
    """Epoch where one channel is uncorrelated with others."""
    rng = np.random.RandomState(77)
    # Channels 0-2 share a base signal
    base = rng.randn(256) * 10.0
    epoch = np.zeros((4, 256))
    for ch in range(3):
        epoch[ch] = base + rng.randn(256) * 2.0
    # Channel 3 is completely independent noise
    epoch[3] = rng.randn(256) * 10.0
    return epoch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanSignal:
    """Clean signal should pass all checks."""

    def test_clean_passes(self, detector: MultiMethodArtifactDetector, clean_epoch: np.ndarray) -> None:
        result = detector.detect(clean_epoch)
        assert result.is_clean is True
        assert len(result.rejected_reasons) == 0
        assert result.overall_quality > 0.5
        for q in result.channel_quality:
            assert q > 0.5

    def test_clean_channel_quality_near_one(self, detector: MultiMethodArtifactDetector, clean_epoch: np.ndarray) -> None:
        result = detector.detect(clean_epoch)
        for q in result.channel_quality:
            assert q >= 0.8, f"Channel quality {q} unexpectedly low for clean epoch"


class TestAmplitudeRejection:
    """High amplitude should trigger rejection."""

    def test_high_amplitude_rejected(self, detector: MultiMethodArtifactDetector, high_amplitude_epoch: np.ndarray) -> None:
        result = detector.detect(high_amplitude_epoch)
        assert result.is_clean is False
        assert "amplitude" in result.rejected_reasons

    def test_amplitude_channel_flagged(self, detector: MultiMethodArtifactDetector, high_amplitude_epoch: np.ndarray) -> None:
        result = detector.detect(high_amplitude_epoch)
        # Channel 0 had the spike; its quality should be lower than others
        assert result.channel_quality[0] < result.channel_quality[1]


class TestGradientRejection:
    """High gradient should trigger rejection."""

    def test_high_gradient_rejected(self, detector: MultiMethodArtifactDetector, high_gradient_epoch: np.ndarray) -> None:
        result = detector.detect(high_gradient_epoch)
        assert result.is_clean is False
        assert "gradient" in result.rejected_reasons

    def test_gradient_correct_channel(self, detector: MultiMethodArtifactDetector, high_gradient_epoch: np.ndarray) -> None:
        result = detector.detect(high_gradient_epoch)
        assert "gradient" in result.artifact_types
        # Channel 1 was the one with the jump
        assert result.channel_quality[1] < 1.0


class TestFlatChannelRejection:
    """Flat channel should trigger rejection."""

    def test_flat_channel_rejected(self, detector: MultiMethodArtifactDetector, flat_epoch: np.ndarray) -> None:
        result = detector.detect(flat_epoch)
        assert result.is_clean is False
        assert "flat" in result.rejected_reasons

    def test_flat_correct_channel(self, detector: MultiMethodArtifactDetector, flat_epoch: np.ndarray) -> None:
        result = detector.detect(flat_epoch)
        # Channel 2 was flat — should have lowest quality
        assert result.channel_quality[2] == min(result.channel_quality)


class TestKurtosisRejection:
    """High kurtosis should trigger rejection."""

    def test_high_kurtosis_rejected(self, detector: MultiMethodArtifactDetector, high_kurtosis_epoch: np.ndarray) -> None:
        result = detector.detect(high_kurtosis_epoch)
        assert result.is_clean is False
        assert "kurtosis" in result.rejected_reasons

    def test_kurtosis_artifact_type_recorded(self, detector: MultiMethodArtifactDetector, high_kurtosis_epoch: np.ndarray) -> None:
        result = detector.detect(high_kurtosis_epoch)
        assert "kurtosis" in result.artifact_types
        assert result.artifact_types["kurtosis"] >= 1


class TestSpectralContamination:
    """Spectral contamination should trigger rejection after baseline builds."""

    def test_spectral_contamination_rejected(self, detector: MultiMethodArtifactDetector, clean_epoch: np.ndarray, spectral_contamination_epoch: np.ndarray) -> None:
        # Feed 6 clean epochs to build spectral baseline
        for _ in range(6):
            detector.detect(clean_epoch)

        # Now the contaminated epoch should be caught by spectral method
        result = detector.detect(spectral_contamination_epoch)
        assert result.is_clean is False
        assert "spectral" in result.rejected_reasons

    def test_spectral_needs_baseline(self, detector: MultiMethodArtifactDetector) -> None:
        # With no baseline (< 5 epochs), spectral method should not fire
        # Use an epoch that has high broadband power but won't trip amplitude
        rng = np.random.RandomState(55)
        t = np.arange(256) / 256.0
        epoch = rng.randn(4, 256) * 3.0
        for freq in [30.0, 35.0, 40.0]:
            for ch in range(4):
                epoch[ch] += 15.0 * np.sin(2 * np.pi * freq * t)
        result = detector.detect(epoch)
        # Spectral specifically should not fire without baseline
        assert "spectral" not in result.rejected_reasons


class TestCorrelationRejection:
    """Low inter-channel correlation should trigger rejection."""

    def test_low_correlation_rejected(self, detector: MultiMethodArtifactDetector, low_correlation_epoch: np.ndarray) -> None:
        result = detector.detect(low_correlation_epoch)
        assert result.is_clean is False
        assert "correlation" in result.rejected_reasons

    def test_correlation_correct_channel(self, detector: MultiMethodArtifactDetector, low_correlation_epoch: np.ndarray) -> None:
        result = detector.detect(low_correlation_epoch)
        # Channel 3 was independent — should have lowest quality
        assert result.channel_quality[3] == min(result.channel_quality)


class TestMixedArtifacts:
    """One bad channel should still report per-channel quality."""

    def test_one_bad_channel_quality(self, detector: MultiMethodArtifactDetector, clean_epoch: np.ndarray) -> None:
        epoch = clean_epoch.copy()
        # Make only channel 0 flat
        epoch[0, :] = 0.01
        result = detector.detect(epoch)
        # Channel 0 should have worse quality than others
        assert result.channel_quality[0] < result.channel_quality[1]
        assert result.channel_quality[0] < result.channel_quality[2]
        assert result.channel_quality[0] < result.channel_quality[3]
        # Other channels should still be mostly clean
        assert result.channel_quality[1] > 0.5
        assert result.channel_quality[2] > 0.5
        assert result.channel_quality[3] > 0.5

    def test_overall_quality_reflects_bad_channel(self, detector: MultiMethodArtifactDetector, clean_epoch: np.ndarray) -> None:
        epoch = clean_epoch.copy()
        epoch[0, :] = 0.01  # flat channel 0
        result = detector.detect(epoch)
        # Overall quality should be less than 1.0 but not 0
        assert 0.0 < result.overall_quality < 1.0


class TestBatchFiltering:
    """Batch filtering should return only clean epochs."""

    def test_batch_returns_clean_only(self, detector: MultiMethodArtifactDetector, clean_epoch: np.ndarray, high_amplitude_epoch: np.ndarray) -> None:
        epochs = [clean_epoch, high_amplitude_epoch, clean_epoch]
        clean_epochs, results = detector.detect_batch(epochs)
        assert len(results) == 3
        assert len(clean_epochs) == 2
        # First and third should be clean
        assert results[0].is_clean is True
        assert results[1].is_clean is False
        assert results[2].is_clean is True

    def test_batch_all_clean(self, detector: MultiMethodArtifactDetector, clean_epoch: np.ndarray) -> None:
        epochs = [clean_epoch, clean_epoch, clean_epoch]
        clean_epochs, results = detector.detect_batch(epochs)
        assert len(clean_epochs) == 3
        assert all(r.is_clean for r in results)

    def test_batch_empty(self, detector: MultiMethodArtifactDetector) -> None:
        clean_epochs, results = detector.detect_batch([])
        assert len(clean_epochs) == 0
        assert len(results) == 0


class TestCleanRatio:
    """Clean ratio calculation."""

    def test_clean_ratio_all_clean(self, detector: MultiMethodArtifactDetector, clean_epoch: np.ndarray) -> None:
        _, results = detector.detect_batch([clean_epoch] * 5)
        ratio = detector.get_clean_ratio(results)
        assert ratio == 1.0

    def test_clean_ratio_mixed(self, detector: MultiMethodArtifactDetector, clean_epoch: np.ndarray, high_amplitude_epoch: np.ndarray) -> None:
        epochs = [clean_epoch, high_amplitude_epoch, clean_epoch, high_amplitude_epoch]
        _, results = detector.detect_batch(epochs)
        ratio = detector.get_clean_ratio(results)
        assert ratio == pytest.approx(0.5)

    def test_clean_ratio_empty(self, detector: MultiMethodArtifactDetector) -> None:
        ratio = detector.get_clean_ratio([])
        assert ratio == 0.0

    def test_clean_ratio_all_bad(self, detector: MultiMethodArtifactDetector, high_amplitude_epoch: np.ndarray) -> None:
        _, results = detector.detect_batch([high_amplitude_epoch] * 3)
        ratio = detector.get_clean_ratio(results)
        assert ratio == 0.0


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_single_channel_input(self, detector: MultiMethodArtifactDetector) -> None:
        """1D input should be reshaped to (1, n_samples)."""
        rng = np.random.RandomState(0)
        signal_1d = rng.randn(256) * 10.0
        result = detector.detect(signal_1d)
        assert isinstance(result, ArtifactResult)
        assert len(result.channel_quality) == 1

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should be respected."""
        detector = MultiMethodArtifactDetector(
            fs=256.0,
            amplitude_threshold=200.0,  # very lenient
            gradient_threshold=50.0,    # very lenient
            kurtosis_threshold=50.0,
        )
        rng = np.random.RandomState(42)
        base = rng.randn(256) * 10.0
        epoch = np.zeros((4, 256))
        for ch in range(4):
            epoch[ch] = base + rng.randn(256) * 3.0
        # With very lenient thresholds, should pass
        result = detector.detect(epoch)
        assert result.is_clean is True

    def test_artifact_result_dataclass(self) -> None:
        """ArtifactResult fields are accessible."""
        result = ArtifactResult(
            is_clean=False,
            rejected_reasons=["amplitude", "kurtosis"],
            channel_quality=[0.8, 0.5, 0.9, 0.7],
            overall_quality=0.725,
            artifact_types={"amplitude": 1, "kurtosis": 1},
        )
        assert result.is_clean is False
        assert len(result.rejected_reasons) == 2
        assert result.overall_quality == 0.725
        assert result.artifact_types["amplitude"] == 1
