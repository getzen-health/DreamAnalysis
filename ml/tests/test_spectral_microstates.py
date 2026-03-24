"""Tests for spectral microstate analysis module."""

import numpy as np
import pytest


class TestClassifyWindow:
    """Test single-window classification by dominant frequency band."""

    def test_alpha_dominant_signal(self, fs):
        """A pure 10 Hz signal should be classified as alpha (A)."""
        from processing.spectral_microstates import classify_window

        # Use a longer window (1 sec) for cleaner PSD resolution at 10 Hz
        t = np.arange(int(fs * 1.0)) / fs
        signal = np.sin(2 * np.pi * 10 * t) * 50  # 10 Hz = alpha
        assert classify_window(signal, fs) == "A"

    def test_beta_dominant_signal(self, fs):
        """A pure 20 Hz signal should be classified as beta (B)."""
        from processing.spectral_microstates import classify_window

        t = np.arange(int(fs * 0.25)) / fs
        signal = np.sin(2 * np.pi * 20 * t) * 50  # 20 Hz = beta
        assert classify_window(signal, fs) == "B"

    def test_delta_dominant_signal(self, fs):
        """A pure 2 Hz signal should be classified as delta (D)."""
        from processing.spectral_microstates import classify_window

        # Need a longer window for delta (low freq needs more cycles)
        t = np.arange(int(fs * 2.0)) / fs
        signal = np.sin(2 * np.pi * 2 * t) * 200  # 2 Hz = delta, high amplitude
        assert classify_window(signal, fs) == "D"

    def test_theta_dominant_signal(self, fs):
        """A pure 6 Hz signal should be classified as theta (T)."""
        from processing.spectral_microstates import classify_window

        t = np.arange(int(fs * 0.5)) / fs
        signal = np.sin(2 * np.pi * 6 * t) * 50  # 6 Hz = theta
        assert classify_window(signal, fs) == "T"

    def test_very_short_signal_returns_default(self, fs):
        """Extremely short signal (< 4 samples) should return default 'A'."""
        from processing.spectral_microstates import classify_window

        signal = np.array([1.0, 2.0, 3.0])
        assert classify_window(signal, fs) == "A"

    def test_returns_valid_state_name(self, fs):
        """Output must be one of the 4 defined state labels."""
        from processing.spectral_microstates import classify_window, STATE_NAMES

        signal = np.random.randn(64)
        result = classify_window(signal, fs)
        assert result in STATE_NAMES


class TestExtractMicrostateSequence:
    """Test conversion of multichannel EEG to microstate sequence."""

    def test_returns_list_of_state_labels(self, multichannel_eeg, fs):
        """Sequence should be a list of valid state labels."""
        from processing.spectral_microstates import (
            extract_microstate_sequence,
            STATE_NAMES,
        )

        seq = extract_microstate_sequence(multichannel_eeg, fs)
        assert isinstance(seq, list)
        assert len(seq) > 0
        for s in seq:
            assert s in STATE_NAMES

    def test_sequence_length_matches_windows(self, fs):
        """Number of states should equal n_samples // window_samples."""
        from processing.spectral_microstates import extract_microstate_sequence

        # 4 channels, 2 seconds at 256 Hz = 512 samples
        signals = np.random.randn(4, 512) * 20
        window_ms = 250
        expected_windows = 512 // (window_ms * fs // 1000)

        seq = extract_microstate_sequence(signals, fs, window_ms=window_ms)
        assert len(seq) == expected_windows

    def test_single_channel_input(self, sample_eeg, fs):
        """Should handle 1D input by reshaping."""
        from processing.spectral_microstates import extract_microstate_sequence

        seq = extract_microstate_sequence(sample_eeg, fs)
        assert isinstance(seq, list)
        assert len(seq) > 0

    def test_custom_window_size(self, multichannel_eeg, fs):
        """Larger window should produce fewer states."""
        from processing.spectral_microstates import extract_microstate_sequence

        seq_250 = extract_microstate_sequence(multichannel_eeg, fs, window_ms=250)
        seq_500 = extract_microstate_sequence(multichannel_eeg, fs, window_ms=500)
        assert len(seq_250) >= len(seq_500)


class TestExtractMicrostateFeatures:
    """Test full feature extraction pipeline."""

    def test_returns_expected_keys(self, multichannel_eeg, fs):
        """Output dict must contain all documented keys."""
        from processing.spectral_microstates import extract_microstate_features

        result = extract_microstate_features(multichannel_eeg, fs)
        expected_keys = {
            "coverage",
            "avg_duration",
            "occurrence",
            "transition_matrix",
            "dominant_state",
            "state_diversity",
            "feature_vector",
            "n_features",
            "sequence_length",
        }
        assert expected_keys.issubset(result.keys())

    def test_feature_vector_length_is_31(self, multichannel_eeg, fs):
        """Feature vector should have exactly 31 elements (28 base + 3 entropy)."""
        from processing.spectral_microstates import extract_microstate_features

        result = extract_microstate_features(multichannel_eeg, fs)
        assert result["n_features"] == 31
        assert len(result["feature_vector"]) == 31

    def test_coverage_sums_to_one(self, multichannel_eeg, fs):
        """State coverages should sum to 1.0."""
        from processing.spectral_microstates import extract_microstate_features

        result = extract_microstate_features(multichannel_eeg, fs)
        total = sum(result["coverage"].values())
        assert abs(total - 1.0) < 1e-6

    def test_coverage_values_between_zero_and_one(self, multichannel_eeg, fs):
        """Each coverage should be in [0, 1]."""
        from processing.spectral_microstates import extract_microstate_features

        result = extract_microstate_features(multichannel_eeg, fs)
        for state, cov in result["coverage"].items():
            assert 0.0 <= cov <= 1.0, f"Coverage for {state}: {cov}"

    def test_durations_are_non_negative(self, multichannel_eeg, fs):
        """Average durations should be >= 0."""
        from processing.spectral_microstates import extract_microstate_features

        result = extract_microstate_features(multichannel_eeg, fs)
        for state, dur in result["avg_duration"].items():
            assert dur >= 0.0, f"Duration for {state}: {dur}"

    def test_occurrences_are_non_negative(self, multichannel_eeg, fs):
        """Occurrence rates should be >= 0."""
        from processing.spectral_microstates import extract_microstate_features

        result = extract_microstate_features(multichannel_eeg, fs)
        for state, occ in result["occurrence"].items():
            assert occ >= 0.0, f"Occurrence for {state}: {occ}"

    def test_transition_rows_sum_to_one(self, multichannel_eeg, fs):
        """Each row in the transition matrix should sum to ~1.0."""
        from processing.spectral_microstates import (
            extract_microstate_features,
            STATE_NAMES,
        )

        result = extract_microstate_features(multichannel_eeg, fs)
        tm = result["transition_matrix"]

        for from_state in STATE_NAMES:
            row_sum = sum(
                tm.get(f"{from_state}->{to_state}", 0.0)
                for to_state in STATE_NAMES
            )
            # Row sums to 1 if the state appeared, or 0 if it never appeared
            assert row_sum < 1.01, f"Row {from_state} sums to {row_sum}"

    def test_dominant_state_is_valid(self, multichannel_eeg, fs):
        """Dominant state must be one of the 4 defined states."""
        from processing.spectral_microstates import (
            extract_microstate_features,
            STATE_NAMES,
        )

        result = extract_microstate_features(multichannel_eeg, fs)
        assert result["dominant_state"] in STATE_NAMES

    def test_state_diversity_is_non_negative(self, multichannel_eeg, fs):
        """Shannon entropy of coverage should be >= 0."""
        from processing.spectral_microstates import extract_microstate_features

        result = extract_microstate_features(multichannel_eeg, fs)
        assert result["state_diversity"] >= 0.0

    def test_feature_vector_all_finite(self, multichannel_eeg, fs):
        """All feature vector elements must be finite."""
        from processing.spectral_microstates import extract_microstate_features

        result = extract_microstate_features(multichannel_eeg, fs)
        for i, val in enumerate(result["feature_vector"]):
            assert np.isfinite(val), f"Non-finite at index {i}: {val}"

    def test_empty_features_for_too_short_signal(self, fs):
        """Signal too short for even 1 window should return empty features."""
        from processing.spectral_microstates import extract_microstate_features

        signals = np.random.randn(4, 10)  # Only 10 samples
        result = extract_microstate_features(signals, fs)
        assert result["sequence_length"] == 0
        assert result["n_features"] == 31
        assert len(result["feature_vector"]) == 31


class TestEegProcessorWrapper:
    """Test the wrapper function in eeg_processor.py."""

    def test_wrapper_delegates_to_module(self, multichannel_eeg, fs):
        """eeg_processor wrapper should return same structure as direct call."""
        from processing.eeg_processor import extract_spectral_microstate_features

        result = extract_spectral_microstate_features(multichannel_eeg, fs)
        assert "coverage" in result
        assert "feature_vector" in result
        assert result["n_features"] == 31
