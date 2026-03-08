"""Tests for EEG channel reflection data augmentation."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure training module is importable
_ML_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ML_ROOT), str(_ML_ROOT / "training")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.augmentation import (
    MUSE2_SYMMETRIC_PAIRS,
    channel_reflect,
    augment_with_reflection,
    reflect_features_85dim,
    augment_features_with_reflection,
    add_gaussian_noise,
    temporal_jitter,
)


class TestChannelReflect:
    """Tests for raw EEG channel reflection."""

    def test_basic_swap(self):
        """Channels swap correctly: TP9<->TP10, AF7<->AF8."""
        eeg = np.array([
            [1.0, 1.0, 1.0],   # ch0 = TP9
            [2.0, 2.0, 2.0],   # ch1 = AF7
            [3.0, 3.0, 3.0],   # ch2 = AF8
            [4.0, 4.0, 4.0],   # ch3 = TP10
        ])
        reflected = channel_reflect(eeg)

        # TP9 (ch0) should now contain TP10 data (ch3)
        np.testing.assert_array_equal(reflected[0], eeg[3])
        # AF7 (ch1) should now contain AF8 data (ch2)
        np.testing.assert_array_equal(reflected[1], eeg[2])
        # AF8 (ch2) should now contain AF7 data (ch1)
        np.testing.assert_array_equal(reflected[2], eeg[1])
        # TP10 (ch3) should now contain TP9 data (ch0)
        np.testing.assert_array_equal(reflected[3], eeg[0])

    def test_double_reflect_is_identity(self):
        """Reflecting twice returns original data."""
        eeg = np.random.randn(4, 256)
        reflected_twice = channel_reflect(channel_reflect(eeg))
        np.testing.assert_array_almost_equal(reflected_twice, eeg)

    def test_no_mutation(self):
        """Original array is not modified."""
        eeg = np.random.randn(4, 100)
        original = eeg.copy()
        channel_reflect(eeg)
        np.testing.assert_array_equal(eeg, original)

    def test_wrong_shape_raises(self):
        """Non-4-channel input raises ValueError."""
        with pytest.raises(ValueError):
            channel_reflect(np.random.randn(3, 100))
        with pytest.raises(ValueError):
            channel_reflect(np.random.randn(100))

    def test_preserves_dtype(self):
        """Output dtype matches input dtype."""
        eeg_f32 = np.random.randn(4, 100).astype(np.float32)
        reflected = channel_reflect(eeg_f32)
        assert reflected.dtype == np.float32

    def test_preserves_shape(self):
        """Output shape matches input shape."""
        eeg = np.random.randn(4, 512)
        reflected = channel_reflect(eeg)
        assert reflected.shape == eeg.shape


class TestAugmentWithReflection:
    """Tests for raw EEG batch augmentation."""

    def test_doubles_data(self):
        """Output is 2x the input size."""
        X = np.random.randn(10, 4, 256)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        X_aug, y_aug, _ = augment_with_reflection(X, y)
        assert len(X_aug) == 20
        assert len(y_aug) == 20

    def test_labels_preserved(self):
        """Class labels are the same for original and reflected."""
        X = np.random.randn(6, 4, 128)
        y = np.array([0, 1, 2, 0, 1, 2])
        _, y_aug, _ = augment_with_reflection(X, y)
        # First half is original, second half is reflected -- same labels
        np.testing.assert_array_equal(y_aug[:6], y)
        np.testing.assert_array_equal(y_aug[6:], y)

    def test_valence_flipped(self):
        """Valence labels are negated for reflected samples."""
        X = np.random.randn(4, 4, 128)
        y = np.array([0, 1, 2, 0])
        valence = np.array([0.5, -0.3, 0.0, 0.8])
        _, _, v_aug = augment_with_reflection(X, y, valence_labels=valence)
        # Original valence preserved
        np.testing.assert_array_almost_equal(v_aug[:4], valence)
        # Reflected valence negated
        np.testing.assert_array_almost_equal(v_aug[4:], -valence)

    def test_no_valence_returns_none(self):
        """Without valence_labels, valence_aug is None."""
        X = np.random.randn(3, 4, 128)
        y = np.array([0, 1, 2])
        _, _, v_aug = augment_with_reflection(X, y)
        assert v_aug is None


class TestReflectFeatures85dim:
    """Tests for feature-space channel reflection (85-dim)."""

    def test_channel_swap_in_features(self):
        """Per-channel features swap correctly in the 80-feature block."""
        features = np.zeros(85)
        # Set ch0 (TP9) features for band 0 to 1.0
        # Index: band*16 + ch*4 + stat
        for stat in range(4):
            features[0 * 16 + 0 * 4 + stat] = 1.0  # TP9, band 0
            features[0 * 16 + 3 * 4 + stat] = 4.0  # TP10, band 0
            features[0 * 16 + 1 * 4 + stat] = 2.0  # AF7, band 0
            features[0 * 16 + 2 * 4 + stat] = 3.0  # AF8, band 0

        reflected = reflect_features_85dim(features)

        for stat in range(4):
            # TP9 should now have TP10 values
            assert reflected[0 * 16 + 0 * 4 + stat] == 4.0
            # TP10 should now have TP9 values
            assert reflected[0 * 16 + 3 * 4 + stat] == 1.0
            # AF7 should now have AF8 values
            assert reflected[0 * 16 + 1 * 4 + stat] == 3.0
            # AF8 should now have AF7 values
            assert reflected[0 * 16 + 2 * 4 + stat] == 2.0

    def test_dasm_negated(self):
        """DASM features (indices 80-84) are negated."""
        features = np.zeros(85)
        features[80:85] = [0.1, -0.2, 0.3, -0.4, 0.5]
        reflected = reflect_features_85dim(features)
        np.testing.assert_array_almost_equal(
            reflected[80:85], [-0.1, 0.2, -0.3, 0.4, -0.5]
        )

    def test_double_reflect_is_identity(self):
        """Reflecting features twice returns original."""
        features = np.random.randn(85)
        reflected_twice = reflect_features_85dim(reflect_features_85dim(features))
        np.testing.assert_array_almost_equal(reflected_twice, features)

    def test_batch_mode(self):
        """Works on (n_samples, 85) arrays."""
        features = np.random.randn(10, 85)
        reflected = reflect_features_85dim(features)
        assert reflected.shape == (10, 85)

        # Verify batch matches individual
        for i in range(10):
            individual = reflect_features_85dim(features[i])
            np.testing.assert_array_almost_equal(reflected[i], individual)

    def test_all_bands_swapped(self):
        """Channel swap happens across all 5 bands, not just band 0."""
        features = np.random.randn(85)
        reflected = reflect_features_85dim(features)

        for band in range(5):
            for stat in range(4):
                # AF7 and AF8 should be swapped
                idx_af7 = band * 16 + 1 * 4 + stat
                idx_af8 = band * 16 + 2 * 4 + stat
                assert reflected[idx_af7] == features[idx_af8]
                assert reflected[idx_af8] == features[idx_af7]

    def test_no_mutation(self):
        """Original array is not modified."""
        features = np.random.randn(85)
        original = features.copy()
        reflect_features_85dim(features)
        np.testing.assert_array_equal(features, original)


class TestAugmentFeaturesWithReflection:
    """Tests for feature-space batch augmentation."""

    def test_doubles_data(self):
        """Output is 2x the input."""
        X = np.random.randn(50, 85)
        y = np.random.randint(0, 3, 50)
        X_aug, y_aug = augment_features_with_reflection(X, y)
        assert len(X_aug) == 100
        assert len(y_aug) == 100

    def test_labels_preserved(self):
        """Labels are identical for original and reflected."""
        X = np.random.randn(20, 85)
        y = np.array([0, 1, 2] * 6 + [0, 1])
        X_aug, y_aug = augment_features_with_reflection(X, y)
        np.testing.assert_array_equal(y_aug[:20], y)
        np.testing.assert_array_equal(y_aug[20:], y)

    def test_reflected_half_differs(self):
        """Reflected features are not identical to originals (due to DASM)."""
        X = np.random.randn(10, 85)
        # Ensure DASM values are non-zero
        X[:, 80:85] = np.random.randn(10, 5) * 0.5
        X_aug, _ = augment_features_with_reflection(X, np.zeros(10))
        # Reflected features should differ from originals
        assert not np.allclose(X_aug[:10], X_aug[10:])


class TestGaussianNoise:
    """Tests for Gaussian noise augmentation."""

    def test_output_shape(self):
        """Output shape matches input."""
        eeg = np.random.randn(4, 256)
        noisy = add_gaussian_noise(eeg)
        assert noisy.shape == eeg.shape

    def test_noise_added(self):
        """Output differs from input."""
        eeg = np.random.randn(4, 256)
        noisy = add_gaussian_noise(eeg, std=0.1)
        assert not np.allclose(noisy, eeg)

    def test_noise_magnitude(self):
        """Noise is roughly the right magnitude."""
        eeg = np.ones((4, 1000))
        noisy = add_gaussian_noise(eeg, std=0.05)
        diff = noisy - eeg
        # Noise should be small relative to signal
        assert np.std(diff) < 0.5


class TestTemporalJitter:
    """Tests for temporal jitter augmentation."""

    def test_output_shape(self):
        """Output shape matches input."""
        eeg = np.random.randn(4, 256)
        jittered = temporal_jitter(eeg)
        assert jittered.shape == eeg.shape

    def test_shifts_data(self):
        """With max_shift > 0, data can be shifted."""
        np.random.seed(42)
        eeg = np.arange(20).reshape(4, 5).astype(float)
        # Run multiple times -- at least one should differ
        any_different = False
        for _ in range(20):
            jittered = temporal_jitter(eeg, max_shift=2)
            if not np.allclose(jittered, eeg):
                any_different = True
                break
        assert any_different
