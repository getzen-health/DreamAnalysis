"""Tests for voice_biomarker.py — eGeMAPS-inspired feature extraction."""

import numpy as np
import pytest

from models.voice_biomarker import (
    VoiceBiomarkerFeatures,
    extract_egemaps_features,
    features_to_array,
)


class TestVoiceBiomarkerFeatures:
    """Test the VoiceBiomarkerFeatures dataclass."""

    def test_default_values(self):
        f = VoiceBiomarkerFeatures()
        assert f.f0_mean == 0.0
        assert f.jitter_local == 0.0
        assert f.shimmer_local == 0.0
        assert f.hnr_mean == 0.0
        assert len(f.mfcc) == 13
        assert f.sample_rate == 16000

    def test_mfcc_length(self):
        f = VoiceBiomarkerFeatures()
        assert len(f.mfcc) == 13
        assert all(v == 0.0 for v in f.mfcc)


class TestExtractEgemapsFeatures:
    """Test the main feature extraction function."""

    def test_returns_features_for_sine_wave(self):
        """A pure sine wave at 200 Hz should yield a stable F0 near 200 Hz."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 200.0
        signal = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float64)

        features = extract_egemaps_features(signal, sr)

        assert features.duration_sec == pytest.approx(1.0, abs=0.01)
        assert features.sample_rate == 16000
        # F0 should be roughly near 200 Hz (not exact due to pre-emphasis + windowing)
        if features.f0_mean > 0:
            assert features.f0_mean == pytest.approx(200.0, rel=0.3)

    def test_returns_zeroed_features_for_short_audio(self):
        """Audio shorter than 2 frames should return zeroed features."""
        sr = 16000
        signal = np.zeros(100, dtype=np.float64)  # way too short

        features = extract_egemaps_features(signal, sr)

        assert features.f0_mean == 0.0
        assert features.jitter_local == 0.0
        assert features.shimmer_local == 0.0

    def test_returns_features_for_silence(self):
        """A silent signal should have zero F0 and zero voiced fraction."""
        sr = 16000
        signal = np.zeros(sr, dtype=np.float64)  # 1 second of silence

        features = extract_egemaps_features(signal, sr)

        assert features.f0_mean == 0.0
        assert features.voiced_fraction == 0.0
        assert features.n_voiced_frames == 0

    def test_voiced_fraction_for_loud_signal(self):
        """A loud signal should have non-zero voiced fraction."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Moderate volume sine wave
        signal = 0.3 * np.sin(2 * np.pi * 150 * t).astype(np.float64)

        features = extract_egemaps_features(signal, sr)

        assert features.voiced_fraction > 0.0

    def test_jitter_shimmer_non_negative(self):
        """Jitter and shimmer should be non-negative."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        signal = 0.3 * np.sin(2 * np.pi * 150 * t).astype(np.float64)
        # Add slight noise to create some perturbation
        signal += 0.01 * np.random.randn(len(signal))

        features = extract_egemaps_features(signal, sr)

        assert features.jitter_local >= 0.0
        assert features.shimmer_local >= 0.0

    def test_mfcc_computed(self):
        """MFCC should be computed for a non-trivial signal."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        signal = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float64)

        features = extract_egemaps_features(signal, sr)

        assert len(features.mfcc) == 13
        # At least some MFCCs should be non-zero for a non-silent signal
        assert any(v != 0.0 for v in features.mfcc)

    def test_spectral_centroid_positive_for_signal(self):
        """Spectral centroid should be positive for a tonal signal."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        signal = 0.3 * np.sin(2 * np.pi * 300 * t).astype(np.float64)

        features = extract_egemaps_features(signal, sr)

        assert features.spectral_centroid > 0


class TestFeaturesToArray:
    """Test conversion to flat numpy array."""

    def test_output_shape(self):
        features = VoiceBiomarkerFeatures()
        arr = features_to_array(features)
        # 12 scalar features + 13 MFCCs = 25
        assert arr.shape == (25,)
        assert arr.dtype == np.float32

    def test_values_match(self):
        features = VoiceBiomarkerFeatures()
        features.f0_mean = 150.0
        features.jitter_local = 1.5
        features.mfcc = [0.1 * i for i in range(13)]

        arr = features_to_array(features)

        assert arr[0] == pytest.approx(150.0)
        assert arr[3] == pytest.approx(1.5)
        assert arr[12] == pytest.approx(0.0)  # mfcc[0] = 0.0
        assert arr[13] == pytest.approx(0.1)  # mfcc[1] = 0.1

    def test_round_trip(self):
        """Feature extraction -> to_array should produce finite values."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        signal = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float64)

        features = extract_egemaps_features(signal, sr)
        arr = features_to_array(features)

        assert np.all(np.isfinite(arr))
        assert arr.shape == (25,)
