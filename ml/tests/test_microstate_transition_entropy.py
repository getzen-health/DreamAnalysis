"""Tests for microstate transition entropy features.

Improvement #40: Adds entropy rate, excess entropy, and Lempel-Ziv
complexity from the microstate transition matrix.

Evidence:
- Weng et al. (2025, Sleep Med) -- entropy rate + excess entropy
  distinguish insomnia vs controls at 82.8% accuracy
- Bai et al. (2024) -- unique transition pairs distinguish positive vs
  negative emotions in VR EEG microstate analysis
- von Wegner et al. (2018, Front Comput Neurosci) -- information-theoretic
  framework for microstate sequence analysis
"""

import numpy as np
import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def fs():
    return 256


@pytest.fixture
def multichannel_eeg():
    """4 channels x 4 seconds of synthetic EEG at 256 Hz."""
    np.random.seed(42)
    return np.random.randn(4, 1024) * 20


# ── Unit tests for compute_transition_entropy ───────────────────────────────

class TestComputeTransitionEntropy:
    """Test the standalone transition entropy computation."""

    def test_returns_expected_keys(self):
        """Output dict must contain entropy_rate, excess_entropy, lz_complexity."""
        from processing.spectral_microstates import compute_transition_entropy

        # Simple deterministic sequence
        sequence = ["A", "B", "A", "B", "A", "B", "A", "B"]
        result = compute_transition_entropy(sequence)
        assert "entropy_rate" in result
        assert "excess_entropy" in result
        assert "lz_complexity" in result

    def test_constant_sequence_has_zero_entropy_rate(self):
        """A sequence stuck in one state has zero transition entropy."""
        from processing.spectral_microstates import compute_transition_entropy

        sequence = ["A"] * 20
        result = compute_transition_entropy(sequence)
        assert result["entropy_rate"] == pytest.approx(0.0, abs=1e-6)

    def test_constant_sequence_has_zero_excess_entropy(self):
        """A constant sequence also has zero excess entropy."""
        from processing.spectral_microstates import compute_transition_entropy

        sequence = ["A"] * 20
        result = compute_transition_entropy(sequence)
        assert result["excess_entropy"] == pytest.approx(0.0, abs=1e-6)

    def test_constant_sequence_has_low_lz_complexity(self):
        """A constant sequence should have minimal LZ complexity."""
        from processing.spectral_microstates import compute_transition_entropy

        sequence = ["A"] * 50
        result = compute_transition_entropy(sequence)
        # Normalized LZ for a constant sequence should be very low
        assert result["lz_complexity"] < 0.2

    def test_alternating_sequence_has_known_entropy_rate(self):
        """A->B->A->B is perfectly predictable: entropy_rate should be 0."""
        from processing.spectral_microstates import compute_transition_entropy

        # Perfectly deterministic alternation: given current state,
        # next state is certain. Entropy rate = 0 for deterministic Markov chain.
        sequence = ["A", "B"] * 20
        result = compute_transition_entropy(sequence)
        assert result["entropy_rate"] == pytest.approx(0.0, abs=1e-6)

    def test_random_sequence_has_high_entropy_rate(self):
        """A maximally random sequence should have high entropy rate."""
        from processing.spectral_microstates import compute_transition_entropy

        np.random.seed(123)
        states = ["D", "T", "A", "B"]
        sequence = [states[i] for i in np.random.randint(0, 4, size=200)]
        result = compute_transition_entropy(sequence)
        # Max entropy rate for 4 states = log2(4) = 2.0
        # Random sequence should be close to maximum
        assert result["entropy_rate"] > 1.5

    def test_random_sequence_has_near_zero_excess_entropy(self):
        """For iid random, excess_entropy should be near zero
        (no temporal structure beyond single-symbol entropy)."""
        from processing.spectral_microstates import compute_transition_entropy

        np.random.seed(123)
        states = ["D", "T", "A", "B"]
        sequence = [states[i] for i in np.random.randint(0, 4, size=500)]
        result = compute_transition_entropy(sequence)
        # Excess entropy = H1 - H_rate; for iid, these are equal
        assert abs(result["excess_entropy"]) < 0.3

    def test_random_sequence_has_high_lz_complexity(self):
        """A random sequence should have high LZ complexity."""
        from processing.spectral_microstates import compute_transition_entropy

        np.random.seed(123)
        states = ["D", "T", "A", "B"]
        sequence = [states[i] for i in np.random.randint(0, 4, size=200)]
        result = compute_transition_entropy(sequence)
        assert result["lz_complexity"] > 0.5

    def test_all_values_finite(self):
        """All entropy values must be finite numbers."""
        from processing.spectral_microstates import compute_transition_entropy

        sequence = ["A", "B", "T", "D", "A", "B", "A", "T"]
        result = compute_transition_entropy(sequence)
        assert np.isfinite(result["entropy_rate"])
        assert np.isfinite(result["excess_entropy"])
        assert np.isfinite(result["lz_complexity"])

    def test_all_values_non_negative(self):
        """Entropy rate, LZ complexity must be >= 0.
        Excess entropy can be negative in theory for very short sequences
        but should be >= 0 for reasonably long ones."""
        from processing.spectral_microstates import compute_transition_entropy

        np.random.seed(42)
        states = ["D", "T", "A", "B"]
        sequence = [states[i] for i in np.random.randint(0, 4, size=100)]
        result = compute_transition_entropy(sequence)
        assert result["entropy_rate"] >= 0.0
        assert result["lz_complexity"] >= 0.0

    def test_short_sequence_returns_zeros(self):
        """Sequence of length < 2 should return zero entropies."""
        from processing.spectral_microstates import compute_transition_entropy

        result = compute_transition_entropy(["A"])
        assert result["entropy_rate"] == 0.0
        assert result["excess_entropy"] == 0.0
        assert result["lz_complexity"] == 0.0


# ── Integration: extract_microstate_features includes entropy ───────────────

class TestMicrostateEntropyIntegration:
    """Test that entropy features are included in the main extraction."""

    def test_features_dict_has_entropy_keys(self, multichannel_eeg, fs):
        """extract_microstate_features must return transition_entropy dict."""
        from processing.spectral_microstates import extract_microstate_features

        result = extract_microstate_features(multichannel_eeg, fs)
        assert "transition_entropy" in result
        te = result["transition_entropy"]
        assert "entropy_rate" in te
        assert "excess_entropy" in te
        assert "lz_complexity" in te

    def test_feature_vector_length_is_31(self, multichannel_eeg, fs):
        """Feature vector grows from 28 to 31 with 3 entropy features."""
        from processing.spectral_microstates import extract_microstate_features

        result = extract_microstate_features(multichannel_eeg, fs)
        assert result["n_features"] == 31
        assert len(result["feature_vector"]) == 31

    def test_entropy_features_in_feature_vector(self, multichannel_eeg, fs):
        """Last 3 elements of feature vector should be the entropy features."""
        from processing.spectral_microstates import extract_microstate_features

        result = extract_microstate_features(multichannel_eeg, fs)
        te = result["transition_entropy"]
        fv = result["feature_vector"]
        # Last 3 elements: entropy_rate, excess_entropy, lz_complexity
        assert fv[-3] == pytest.approx(te["entropy_rate"], abs=1e-3)
        assert fv[-2] == pytest.approx(te["excess_entropy"], abs=1e-3)
        assert fv[-1] == pytest.approx(te["lz_complexity"], abs=1e-3)

    def test_empty_features_has_entropy_zeros(self, fs):
        """Too-short signal should return zero entropy features."""
        from processing.spectral_microstates import extract_microstate_features

        signals = np.random.randn(4, 10)  # Too short
        result = extract_microstate_features(signals, fs)
        assert result["transition_entropy"]["entropy_rate"] == 0.0
        assert result["transition_entropy"]["excess_entropy"] == 0.0
        assert result["transition_entropy"]["lz_complexity"] == 0.0
        assert result["n_features"] == 31
        assert len(result["feature_vector"]) == 31

    def test_wrapper_returns_31_features(self, multichannel_eeg, fs):
        """eeg_processor wrapper should also return 31-element vector."""
        from processing.eeg_processor import extract_spectral_microstate_features

        result = extract_spectral_microstate_features(multichannel_eeg, fs)
        assert result["n_features"] == 31
        assert "transition_entropy" in result
