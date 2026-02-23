"""Tests for EEG processing pipeline."""

import numpy as np


class TestFeatureExtraction:
    def test_extract_band_powers(self, sample_eeg, fs):
        from processing.eeg_processor import extract_band_powers
        bands = extract_band_powers(sample_eeg, fs)
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            assert band in bands
            assert bands[band] >= 0

    def test_extract_features_returns_17(self, sample_eeg, fs):
        from processing.eeg_processor import extract_features
        features = extract_features(sample_eeg, fs)
        assert len(features) == 31

    def test_preprocess_preserves_length(self, sample_eeg, fs):
        from processing.eeg_processor import preprocess
        processed = preprocess(sample_eeg, fs)
        assert len(processed) == len(sample_eeg)

    def test_hjorth_parameters(self, sample_eeg):
        from processing.eeg_processor import compute_hjorth_parameters
        hjorth = compute_hjorth_parameters(sample_eeg)
        assert "activity" in hjorth
        assert "mobility" in hjorth
        assert "complexity" in hjorth
        assert hjorth["activity"] > 0

    def test_spectral_entropy(self, sample_eeg, fs):
        from processing.eeg_processor import spectral_entropy
        se = spectral_entropy(sample_eeg, fs)
        assert 0 <= se <= 1

    def test_differential_entropy(self, sample_eeg, fs):
        from processing.eeg_processor import differential_entropy
        de = differential_entropy(sample_eeg, fs)
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            assert band in de

    def test_features_are_finite(self, sample_eeg, fs):
        from processing.eeg_processor import extract_features
        features = extract_features(sample_eeg, fs)
        for key, val in features.items():
            assert np.isfinite(val), f"Non-finite value for {key}: {val}"


class TestEEGSimulator:
    def test_simulate_default(self):
        from simulation.eeg_simulator import simulate_eeg
        data = simulate_eeg()
        assert "signals" in data
        assert len(data["signals"]) > 0

    def test_simulate_states(self):
        from simulation.eeg_simulator import simulate_eeg
        for state in ["rest", "focus", "meditation", "rem", "deep_sleep"]:
            data = simulate_eeg(state=state, duration=2.0, fs=256, n_channels=1)
            assert len(data["signals"][0]) == 512

    def test_simulate_multichannel(self):
        from simulation.eeg_simulator import simulate_eeg
        data = simulate_eeg(n_channels=4, duration=2.0, fs=256)
        assert len(data["signals"]) == 4


class TestNoiseAugmentation:
    def test_augment_eeg(self, sample_eeg):
        from processing.noise_augmentation import augment_eeg
        augmented = augment_eeg(sample_eeg, fs=256, difficulty="medium")
        assert len(augmented) == len(sample_eeg)
        # Should be different from original
        assert not np.array_equal(augmented, sample_eeg)

    def test_add_gaussian_noise(self, sample_eeg):
        from processing.noise_augmentation import add_gaussian_noise
        noisy = add_gaussian_noise(sample_eeg, snr_db=10)
        assert len(noisy) == len(sample_eeg)
        assert not np.array_equal(noisy, sample_eeg)
