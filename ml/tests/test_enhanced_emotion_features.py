"""Tests for enhanced 53-dim emotion feature extraction.

Verifies:
1. extract_enhanced_emotion_features returns 53 features for 4-channel input
2. All features are finite (no NaN/inf)
3. DASM features change sign when channels are swapped
4. DE features are finite (non-negative for Gaussian assumption)
5. Temporal features are zero for first epoch (no history)
6. Temporal features are non-zero for subsequent epochs
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
    return rng.standard_normal((4, 1024)) * 20.0  # ~20 uV RMS


@pytest.fixture
def asymmetric_eeg():
    """4-channel EEG with known left-right asymmetry.

    AF7 (ch1) has strong alpha; AF8 (ch2) has weak alpha.
    This should produce negative DASM_alpha (right < left).
    """
    rng = np.random.default_rng(123)
    t = np.arange(1024) / 256.0
    signals = rng.standard_normal((4, 1024)) * 5.0

    # Inject strong 10 Hz alpha into AF7 (ch1)
    signals[1] += 40.0 * np.sin(2 * np.pi * 10.0 * t)

    # AF8 (ch2) gets weak alpha
    signals[2] += 5.0 * np.sin(2 * np.pi * 10.0 * t)

    return signals


class TestExtractEnhancedEmotionFeatures:
    """Tests for the main feature extractor."""

    def test_returns_correct_shape(self, four_channel_eeg):
        """Feature vector must have exactly 53 dimensions."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        assert features.shape == (ENHANCED_FEATURE_DIM,)
        assert features.shape == (53,)

    def test_all_features_finite(self, four_channel_eeg):
        """No NaN or inf values allowed."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        assert np.all(np.isfinite(features)), (
            f"Non-finite values at indices: {np.where(~np.isfinite(features))[0]}"
        )

    def test_feature_names_match_dim(self):
        """Feature name list must match the feature dimension."""
        assert len(FEATURE_NAMES) == ENHANCED_FEATURE_DIM

    def test_de_features_present(self, four_channel_eeg):
        """DE features (first 20) should be finite real numbers.

        DE = 0.5 * ln(2*pi*e*sigma^2) can be negative when sigma^2 < 1/(2*pi*e),
        but should always be finite.
        """
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        de_features = features[:20]
        assert np.all(np.isfinite(de_features))

    def test_dasm_changes_sign_on_channel_swap(self, asymmetric_eeg):
        """DASM should flip sign when AF7/AF8 channels are swapped."""
        features_normal = extract_enhanced_emotion_features(asymmetric_eeg, fs=256)

        # Swap AF7 (ch1) and AF8 (ch2)
        swapped = asymmetric_eeg.copy()
        swapped[1], swapped[2] = asymmetric_eeg[2].copy(), asymmetric_eeg[1].copy()
        features_swapped = extract_enhanced_emotion_features(swapped, fs=256)

        # DASM features are at indices 20-24
        dasm_normal = features_normal[20:25]
        dasm_swapped = features_swapped[20:25]

        # Signs should be opposite (or both zero)
        for i in range(5):
            if abs(dasm_normal[i]) > 1e-6 and abs(dasm_swapped[i]) > 1e-6:
                assert np.sign(dasm_normal[i]) != np.sign(dasm_swapped[i]), (
                    f"DASM band {i}: normal={dasm_normal[i]:.4f}, "
                    f"swapped={dasm_swapped[i]:.4f} — expected opposite signs"
                )

    def test_single_channel_input_padded(self):
        """Single-channel input should be padded to 4 channels and still work."""
        rng = np.random.default_rng(99)
        signal_1d = rng.standard_normal(1024) * 20.0
        features = extract_enhanced_emotion_features(signal_1d, fs=256)
        assert features.shape == (53,)
        assert np.all(np.isfinite(features))

    def test_short_signal_does_not_crash(self):
        """Very short signals should produce features without errors."""
        rng = np.random.default_rng(77)
        short = rng.standard_normal((4, 64)) * 20.0
        features = extract_enhanced_emotion_features(short, fs=256)
        assert features.shape == (53,)
        assert np.all(np.isfinite(features))

    def test_flat_signal_produces_finite_features(self):
        """Flat (DC) signal should not produce NaN/inf."""
        flat = np.ones((4, 1024)) * 0.001
        features = extract_enhanced_emotion_features(flat, fs=256)
        assert features.shape == (53,)
        assert np.all(np.isfinite(features))


class TestExtractTemporalFeatures:
    """Tests for temporal delta features."""

    def test_first_epoch_deltas_are_zero(self, four_channel_eeg):
        """Without history, delta features should all be zero."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        temporal = extract_temporal_features(features, history=None)

        assert temporal.shape == (ENHANCED_FEATURE_DIM * 2,)
        # First half = instantaneous features
        np.testing.assert_array_equal(temporal[:ENHANCED_FEATURE_DIM], features)
        # Second half = deltas (all zero since no history)
        np.testing.assert_array_equal(temporal[ENHANCED_FEATURE_DIM:], 0.0)

    def test_empty_history_deltas_are_zero(self, four_channel_eeg):
        """Empty history list should produce zero deltas."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        temporal = extract_temporal_features(features, history=[])

        deltas = temporal[ENHANCED_FEATURE_DIM:]
        np.testing.assert_array_equal(deltas, 0.0)

    def test_subsequent_epoch_has_nonzero_deltas(self, four_channel_eeg):
        """With a different previous epoch, deltas should be non-zero."""
        rng = np.random.default_rng(55)
        signals2 = rng.standard_normal((4, 1024)) * 30.0  # different amplitude

        features1 = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        features2 = extract_enhanced_emotion_features(signals2, fs=256)

        temporal = extract_temporal_features(features2, history=[features1])

        deltas = temporal[ENHANCED_FEATURE_DIM:]
        # At least some deltas should be non-zero (different inputs)
        assert np.any(np.abs(deltas) > 1e-10), "Expected non-zero deltas between different epochs"

    def test_temporal_output_shape(self, four_channel_eeg):
        """Output should be exactly 106 dimensions (53 + 53)."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        temporal = extract_temporal_features(features, history=None)
        assert temporal.shape == (106,)

    def test_temporal_features_are_finite(self, four_channel_eeg):
        """All temporal features must be finite."""
        rng = np.random.default_rng(88)
        signals2 = rng.standard_normal((4, 1024)) * 20.0

        f1 = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        f2 = extract_enhanced_emotion_features(signals2, fs=256)

        temporal = extract_temporal_features(f2, history=[f1])
        assert np.all(np.isfinite(temporal))


class TestGetFeatureNames:
    """Tests for feature name retrieval."""

    def test_base_names_length(self):
        names = get_feature_names(include_temporal=False)
        assert len(names) == 53

    def test_temporal_names_length(self):
        names = get_feature_names(include_temporal=True)
        assert len(names) == 106

    def test_temporal_names_prefixed(self):
        names = get_feature_names(include_temporal=True)
        for name in names[53:]:
            assert name.startswith("delta_"), f"Expected 'delta_' prefix, got: {name}"
