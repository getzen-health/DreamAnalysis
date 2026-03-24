"""Tests for nonlinear complexity features in the enhanced emotion feature vector.

Verifies that Higuchi Fractal Dimension (HFD), Sample Entropy (SampEn), and
Lempel-Ziv Complexity (LZC) are correctly integrated into the 80-dim emotion
feature vector (previously 68-dim).

Feature layout for the 12 new features (indices 68-79):
    hfd_TP9, hfd_AF7, hfd_AF8, hfd_TP10      (4 HFD per channel)
    sampen_TP9, sampen_AF7, sampen_AF8, sampen_TP10  (4 SampEn per channel)
    lzc_TP9, lzc_AF7, lzc_AF8, lzc_TP10      (4 LZC per channel)

References:
    Ahmadlou et al. (2012): HFD for EEG emotion classification
    Richman & Moorman (2000): Sample entropy
    Lempel & Ziv (1976): LZ complexity
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.emotion_features_enhanced import (
    ENHANCED_FEATURE_DIM,
    FEATURE_NAMES,
    extract_enhanced_emotion_features,
    extract_temporal_features,
    get_feature_names,
)


@pytest.fixture
def four_channel_eeg():
    """4 channels x 4 seconds of synthetic EEG at 256 Hz."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 1024)) * 20.0


@pytest.fixture
def simple_sine_eeg():
    """4-channel EEG with a clean 10 Hz sine wave -- low complexity signal."""
    t = np.arange(1024) / 256.0
    sine = 30.0 * np.sin(2 * np.pi * 10.0 * t)
    return np.tile(sine[np.newaxis, :], (4, 1))


@pytest.fixture
def random_noise_eeg():
    """4-channel EEG with pure random noise -- high complexity signal."""
    rng = np.random.default_rng(999)
    return rng.standard_normal((4, 1024)) * 20.0


class TestNonlinearFeaturesIntegrated:
    """Tests for nonlinear complexity features in the emotion feature vector."""

    def test_feature_dim_is_80(self):
        """Feature vector should now be 80 dimensions (68 + 12 nonlinear)."""
        assert ENHANCED_FEATURE_DIM == 80

    def test_returns_correct_shape(self, four_channel_eeg):
        """Feature vector must have exactly 80 dimensions."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        assert features.shape == (80,)

    def test_all_features_finite(self, four_channel_eeg):
        """No NaN or inf values in the expanded feature vector."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        assert np.all(np.isfinite(features)), (
            f"Non-finite values at indices: {np.where(~np.isfinite(features))[0]}"
        )

    def test_feature_names_include_nonlinear(self):
        """Feature names should include hfd_, sampen_, and lzc_ entries."""
        hfd_names = [n for n in FEATURE_NAMES if n.startswith("hfd_")]
        sampen_names = [n for n in FEATURE_NAMES if n.startswith("sampen_")]
        lzc_names = [n for n in FEATURE_NAMES if n.startswith("lzc_")]
        assert len(hfd_names) == 4, f"Expected 4 HFD features, got {len(hfd_names)}"
        assert len(sampen_names) == 4, f"Expected 4 SampEn features, got {len(sampen_names)}"
        assert len(lzc_names) == 4, f"Expected 4 LZC features, got {len(lzc_names)}"

    def test_feature_names_match_dim(self):
        """Feature name list must match the 80-dim feature vector."""
        assert len(FEATURE_NAMES) == 80

    def test_hfd_range_valid(self, four_channel_eeg):
        """HFD values should be in [1.0, 2.0] range."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        hfd_indices = [FEATURE_NAMES.index(f"hfd_{ch}") for ch in ("TP9", "AF7", "AF8", "TP10")]
        for idx in hfd_indices:
            assert 1.0 <= features[idx] <= 2.0, (
                f"HFD at index {idx} ({FEATURE_NAMES[idx]}) = {features[idx]}, expected [1.0, 2.0]"
            )

    def test_sampen_non_negative(self, four_channel_eeg):
        """Sample entropy should be non-negative."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        sampen_indices = [FEATURE_NAMES.index(f"sampen_{ch}") for ch in ("TP9", "AF7", "AF8", "TP10")]
        for idx in sampen_indices:
            assert features[idx] >= 0.0, (
                f"SampEn at index {idx} ({FEATURE_NAMES[idx]}) = {features[idx]}, expected >= 0"
            )

    def test_lzc_in_unit_range(self, four_channel_eeg):
        """Lempel-Ziv complexity should be in [0, 1] range."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        lzc_indices = [FEATURE_NAMES.index(f"lzc_{ch}") for ch in ("TP9", "AF7", "AF8", "TP10")]
        for idx in lzc_indices:
            assert 0.0 <= features[idx] <= 1.0, (
                f"LZC at index {idx} ({FEATURE_NAMES[idx]}) = {features[idx]}, expected [0, 1]"
            )

    def test_sine_has_lower_complexity_than_noise(self, simple_sine_eeg, random_noise_eeg):
        """A pure sine wave should have lower HFD and LZC than random noise."""
        feat_sine = extract_enhanced_emotion_features(simple_sine_eeg, fs=256)
        feat_noise = extract_enhanced_emotion_features(random_noise_eeg, fs=256)

        # Check HFD for channel 0 (TP9)
        hfd_idx = FEATURE_NAMES.index("hfd_TP9")
        assert feat_sine[hfd_idx] < feat_noise[hfd_idx], (
            f"Sine HFD ({feat_sine[hfd_idx]:.4f}) should be < noise HFD ({feat_noise[hfd_idx]:.4f})"
        )

        # Check LZC for channel 0 (TP9)
        lzc_idx = FEATURE_NAMES.index("lzc_TP9")
        assert feat_sine[lzc_idx] < feat_noise[lzc_idx], (
            f"Sine LZC ({feat_sine[lzc_idx]:.4f}) should be < noise LZC ({feat_noise[lzc_idx]:.4f})"
        )

    def test_flat_signal_nonlinear_features_finite(self):
        """Flat (DC) signal should not produce NaN/inf for nonlinear features."""
        flat = np.ones((4, 1024)) * 0.001
        features = extract_enhanced_emotion_features(flat, fs=256)
        # Check the nonlinear features specifically (last 12)
        nonlinear = features[68:]
        assert np.all(np.isfinite(nonlinear)), (
            f"Non-finite nonlinear features at: {np.where(~np.isfinite(nonlinear))[0]}"
        )

    def test_short_signal_nonlinear_features_finite(self):
        """Very short signals (64 samples) should still produce finite nonlinear features."""
        rng = np.random.default_rng(77)
        short = rng.standard_normal((4, 64)) * 20.0
        features = extract_enhanced_emotion_features(short, fs=256)
        assert features.shape == (80,)
        assert np.all(np.isfinite(features))

    def test_single_channel_produces_80_features(self):
        """Single-channel input padded to 4 channels should still return 80 features."""
        rng = np.random.default_rng(99)
        signal_1d = rng.standard_normal(1024) * 20.0
        features = extract_enhanced_emotion_features(signal_1d, fs=256)
        assert features.shape == (80,)
        assert np.all(np.isfinite(features))


class TestTemporalFeaturesWith80Dim:
    """Tests that temporal features work with the expanded 80-dim vector."""

    def test_temporal_output_shape_is_160(self, four_channel_eeg):
        """Temporal features should be 160 dims (80 + 80 deltas)."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        temporal = extract_temporal_features(features, history=None)
        assert temporal.shape == (160,)

    def test_first_epoch_deltas_are_zero(self, four_channel_eeg):
        """Without history, delta features should all be zero."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        temporal = extract_temporal_features(features, history=None)
        np.testing.assert_array_equal(temporal[:80], features)
        np.testing.assert_array_equal(temporal[80:], 0.0)

    def test_subsequent_epoch_has_nonzero_deltas(self, four_channel_eeg):
        """With a different previous epoch, deltas should be non-zero."""
        rng = np.random.default_rng(55)
        signals2 = rng.standard_normal((4, 1024)) * 30.0

        f1 = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        f2 = extract_enhanced_emotion_features(signals2, fs=256)

        temporal = extract_temporal_features(f2, history=[f1])
        deltas = temporal[80:]
        assert np.any(np.abs(deltas) > 1e-10), "Expected non-zero deltas"


class TestGetFeatureNamesExpanded:
    """Tests that feature name retrieval includes nonlinear names."""

    def test_base_names_length_80(self):
        names = get_feature_names(include_temporal=False)
        assert len(names) == 80

    def test_temporal_names_length_160(self):
        names = get_feature_names(include_temporal=True)
        assert len(names) == 160

    def test_nonlinear_names_present(self):
        names = get_feature_names(include_temporal=False)
        assert "hfd_TP9" in names
        assert "sampen_AF7" in names
        assert "lzc_TP10" in names
