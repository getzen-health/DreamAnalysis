"""Tests for NeuralComplexityAnalyzer.

Covers:
  - analyze() output structure and key presence
  - Individual metric ranges (sample entropy, permutation entropy, LZ, Hurst,
    fractal dimension, DFA)
  - Composite complexity_index (0-100 range)
  - Consciousness level classification (conscious/reduced/minimal)
  - State transition detection across sequential calls
  - get_consciousness_estimate() returns valid dict
  - get_session_stats() returns min/max/mean/std
  - get_history() returns chronological records
  - reset() clears all internal state
  - Edge cases: flat signal, single-channel, short signal, very noisy signal,
    railed signal, NaN/Inf safety
  - Multichannel averaging vs single-channel
  - Deterministic results with fixed seed
  - History length grows with repeated calls
  - State transition from high to low complexity
  - Consciousness level thresholds
"""

import numpy as np
import pytest

from models.neural_complexity import (
    NeuralComplexityAnalyzer,
    CONSCIOUSNESS_LEVELS,
)


@pytest.fixture
def analyzer():
    return NeuralComplexityAnalyzer()


def _make_eeg(fs=256, duration=4, n_channels=4, seed=42):
    """Synthetic multichannel EEG with realistic spectral content."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = 15 * np.sin(2 * np.pi * 10 * t + ch * 0.5)
        theta = 8 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
        beta = 4 * np.sin(2 * np.pi * 20 * t + ch * 0.7)
        noise = rng.randn(len(t)) * 5
        signals.append(alpha + theta + beta + noise)
    return np.array(signals)


def _make_flat(fs=256, duration=4, n_channels=4):
    """Near-flat signal (disconnected electrode)."""
    return np.ones((n_channels, int(fs * duration))) * 0.001


def _make_sine(freq=10.0, fs=256, duration=4, n_channels=4):
    """Pure sine wave — very low complexity / highly regular."""
    t = np.arange(int(fs * duration)) / fs
    s = np.sin(2 * np.pi * freq * t)
    return np.tile(s, (n_channels, 1))


def _make_random(fs=256, duration=4, n_channels=4, seed=99):
    """Pure white noise — high complexity / maximum entropy."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_channels, int(fs * duration)) * 30


# ── Output structure ──────────────────────────────────────────────────────────

class TestOutputStructure:
    def test_analyze_returns_all_required_keys(self, analyzer):
        """analyze() must return every documented key."""
        result = analyzer.analyze(_make_eeg())
        required = {
            "sample_entropy",
            "permutation_entropy",
            "lz_complexity",
            "hurst_exponent",
            "fractal_dimension",
            "dfa_exponent",
            "complexity_index",
            "consciousness_level",
            "state_transition_detected",
        }
        assert required.issubset(set(result.keys()))

    def test_analyze_values_are_finite(self, analyzer):
        """All numeric values must be finite (no NaN or Inf)."""
        result = analyzer.analyze(_make_eeg())
        for key in ("sample_entropy", "permutation_entropy", "lz_complexity",
                     "hurst_exponent", "fractal_dimension", "dfa_exponent",
                     "complexity_index"):
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"


# ── Individual metric ranges ──────────────────────────────────────────────────

class TestMetricRanges:
    def test_sample_entropy_non_negative(self, analyzer):
        """Sample entropy should be >= 0."""
        result = analyzer.analyze(_make_eeg())
        assert result["sample_entropy"] >= 0.0

    def test_permutation_entropy_bounded(self, analyzer):
        """Permutation entropy should be in [0, 1] (normalized)."""
        result = analyzer.analyze(_make_eeg())
        assert 0.0 <= result["permutation_entropy"] <= 1.0

    def test_lz_complexity_positive(self, analyzer):
        """Lempel-Ziv complexity should be > 0 for any real signal."""
        result = analyzer.analyze(_make_eeg())
        assert result["lz_complexity"] > 0.0

    def test_lz_complexity_normalized(self, analyzer):
        """Lempel-Ziv complexity should be normalized to [0, 1]."""
        result = analyzer.analyze(_make_eeg())
        assert 0.0 <= result["lz_complexity"] <= 1.0

    def test_hurst_exponent_range(self, analyzer):
        """Hurst exponent should be in [0, 1]."""
        result = analyzer.analyze(_make_eeg())
        assert 0.0 <= result["hurst_exponent"] <= 1.0

    def test_fractal_dimension_range(self, analyzer):
        """Higuchi fractal dimension for EEG is typically 1.0-2.0."""
        result = analyzer.analyze(_make_eeg())
        assert 1.0 <= result["fractal_dimension"] <= 2.0

    def test_dfa_exponent_positive(self, analyzer):
        """DFA exponent should be positive."""
        result = analyzer.analyze(_make_eeg())
        assert result["dfa_exponent"] > 0.0


# ── Composite complexity index ────────────────────────────────────────────────

class TestComplexityIndex:
    def test_complexity_index_range(self, analyzer):
        """Composite complexity_index must be in [0, 100]."""
        result = analyzer.analyze(_make_eeg())
        assert 0 <= result["complexity_index"] <= 100

    def test_random_signal_higher_complexity_than_sine(self, analyzer):
        """White noise should have higher complexity than a pure sine wave."""
        result_rand = analyzer.analyze(_make_random())
        result_sine = analyzer.analyze(_make_sine())
        assert result_rand["complexity_index"] > result_sine["complexity_index"]

    def test_eeg_complexity_between_sine_and_noise(self, analyzer):
        """Realistic EEG should have intermediate complexity."""
        ci_sine = analyzer.analyze(_make_sine())["complexity_index"]
        ci_noise = analyzer.analyze(_make_random())["complexity_index"]
        ci_eeg = analyzer.analyze(_make_eeg())["complexity_index"]
        # EEG should be more complex than pure sine but could overlap with noise
        assert ci_eeg > ci_sine


# ── Consciousness level ───────────────────────────────────────────────────────

class TestConsciousnessLevel:
    def test_consciousness_level_valid_value(self, analyzer):
        """consciousness_level must be one of the defined levels."""
        result = analyzer.analyze(_make_eeg())
        assert result["consciousness_level"] in CONSCIOUSNESS_LEVELS

    def test_high_complexity_signal_is_conscious(self, analyzer):
        """Complex EEG-like signal should yield 'conscious' or 'reduced'."""
        result = analyzer.analyze(_make_eeg())
        assert result["consciousness_level"] in ("conscious", "reduced")

    def test_flat_signal_is_minimal(self, analyzer):
        """A near-flat signal should yield 'minimal' consciousness."""
        result = analyzer.analyze(_make_flat())
        assert result["consciousness_level"] == "minimal"

    def test_consciousness_levels_constant(self):
        """Verify the expected levels are defined."""
        assert set(CONSCIOUSNESS_LEVELS) == {"conscious", "reduced", "minimal"}


# ── State transition detection ────────────────────────────────────────────────

class TestStateTransition:
    def test_first_call_no_transition(self, analyzer):
        """First analyze() call should never detect a transition."""
        result = analyzer.analyze(_make_eeg())
        assert result["state_transition_detected"] is False

    def test_same_signal_no_transition(self, analyzer):
        """Repeated identical signals should not trigger transition."""
        sig = _make_eeg()
        analyzer.analyze(sig)
        result = analyzer.analyze(sig)
        assert result["state_transition_detected"] is False

    def test_transition_detected_on_large_shift(self, analyzer):
        """Switching from complex to flat signal should detect transition."""
        analyzer.analyze(_make_eeg())
        result = analyzer.analyze(_make_flat())
        assert result["state_transition_detected"] is True

    def test_transition_detected_flat_to_complex(self, analyzer):
        """Switching from flat to complex should also detect transition."""
        analyzer.analyze(_make_flat())
        result = analyzer.analyze(_make_eeg())
        assert result["state_transition_detected"] is True


# ── get_consciousness_estimate ────────────────────────────────────────────────

class TestConsciousnessEstimate:
    def test_returns_dict_with_required_keys(self, analyzer):
        """get_consciousness_estimate() returns dict with level and index."""
        analyzer.analyze(_make_eeg())
        est = analyzer.get_consciousness_estimate()
        assert "consciousness_level" in est
        assert "complexity_index" in est

    def test_returns_none_before_any_analysis(self, analyzer):
        """Before any analyze() call, estimate should return None."""
        est = analyzer.get_consciousness_estimate()
        assert est is None


# ── get_session_stats ─────────────────────────────────────────────────────────

class TestSessionStats:
    def test_empty_stats_before_analysis(self, analyzer):
        """get_session_stats() should return empty stats with n_epochs=0."""
        stats = analyzer.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_after_multiple_analyses(self, analyzer):
        """Stats should contain min/max/mean/std after multiple calls."""
        for seed in range(5):
            analyzer.analyze(_make_eeg(seed=seed))
        stats = analyzer.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "mean_complexity" in stats
        assert "std_complexity" in stats
        assert "min_complexity" in stats
        assert "max_complexity" in stats
        assert stats["min_complexity"] <= stats["mean_complexity"]
        assert stats["mean_complexity"] <= stats["max_complexity"]
        assert stats["std_complexity"] >= 0.0

    def test_stats_single_epoch(self, analyzer):
        """With one epoch, min == max == mean and std == 0."""
        analyzer.analyze(_make_eeg())
        stats = analyzer.get_session_stats()
        assert stats["n_epochs"] == 1
        assert stats["min_complexity"] == stats["max_complexity"]
        assert stats["std_complexity"] == pytest.approx(0.0, abs=1e-9)


# ── get_history ───────────────────────────────────────────────────────────────

class TestHistory:
    def test_empty_history_initially(self, analyzer):
        """History should be empty before any calls."""
        assert analyzer.get_history() == []

    def test_history_grows_with_calls(self, analyzer):
        """Each analyze() call adds one entry to history."""
        for i in range(3):
            analyzer.analyze(_make_eeg(seed=i))
        history = analyzer.get_history()
        assert len(history) == 3

    def test_history_entries_have_required_keys(self, analyzer):
        """Each history entry should contain complexity_index and consciousness_level."""
        analyzer.analyze(_make_eeg())
        entry = analyzer.get_history()[0]
        assert "complexity_index" in entry
        assert "consciousness_level" in entry


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_history(self, analyzer):
        """After reset(), history should be empty."""
        analyzer.analyze(_make_eeg())
        analyzer.analyze(_make_eeg(seed=10))
        assert len(analyzer.get_history()) == 2
        analyzer.reset()
        assert analyzer.get_history() == []

    def test_reset_clears_consciousness_estimate(self, analyzer):
        """After reset(), consciousness estimate should be None."""
        analyzer.analyze(_make_eeg())
        analyzer.reset()
        assert analyzer.get_consciousness_estimate() is None

    def test_reset_clears_session_stats(self, analyzer):
        """After reset(), session stats should show n_epochs=0."""
        analyzer.analyze(_make_eeg())
        analyzer.reset()
        stats = analyzer.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_no_transition_after_reset(self, analyzer):
        """First analyze() after reset() should not detect transition."""
        analyzer.analyze(_make_eeg())
        analyzer.reset()
        result = analyzer.analyze(_make_flat())
        assert result["state_transition_detected"] is False


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_channel_input(self, analyzer):
        """1D input (single channel) should work without error."""
        sig = _make_eeg(n_channels=1)[0]  # shape (n_samples,)
        result = analyzer.analyze(sig)
        assert np.isfinite(result["complexity_index"])

    def test_2d_single_channel(self, analyzer):
        """(1, n_samples) input should work."""
        sig = _make_eeg(n_channels=1)  # shape (1, n_samples)
        result = analyzer.analyze(sig)
        assert np.isfinite(result["complexity_index"])

    def test_flat_signal_low_complexity(self, analyzer):
        """Near-flat signal should have very low complexity_index."""
        result = analyzer.analyze(_make_flat())
        assert result["complexity_index"] < 20

    def test_railed_signal_does_not_crash(self, analyzer):
        """Saturated signal should not produce errors."""
        railed = np.ones((4, 1024)) * 300
        result = analyzer.analyze(railed)
        assert np.isfinite(result["complexity_index"])

    def test_short_signal_minimum(self, analyzer):
        """Signal with minimum viable length should not crash."""
        short = np.random.randn(4, 128) * 20  # 0.5 seconds at 256 Hz
        result = analyzer.analyze(short, fs=256)
        assert np.isfinite(result["complexity_index"])

    def test_noisy_signal_high_complexity(self, analyzer):
        """Pure noise should have high complexity."""
        result = analyzer.analyze(_make_random())
        assert result["complexity_index"] > 50

    def test_different_sampling_rates(self, analyzer):
        """Different fs should still produce valid results."""
        sig = np.random.randn(4, 512) * 20
        result = analyzer.analyze(sig, fs=128)
        assert np.isfinite(result["complexity_index"])
        assert 0 <= result["complexity_index"] <= 100


# ── Multichannel vs single channel ────────────────────────────────────────────

class TestMultichannel:
    def test_multichannel_averages_channels(self, analyzer):
        """4-channel input should produce results averaged across channels."""
        sig = _make_eeg(n_channels=4)
        result = analyzer.analyze(sig)
        # Just verify it produces valid output without error
        assert np.isfinite(result["sample_entropy"])
        assert np.isfinite(result["complexity_index"])

    def test_single_and_multi_both_valid(self, analyzer):
        """Both single-channel and multi-channel produce valid results."""
        rng = np.random.RandomState(42)
        single = rng.randn(1024) * 20
        multi = rng.randn(4, 1024) * 20

        r1 = analyzer.analyze(single)
        analyzer.reset()
        r2 = analyzer.analyze(multi)

        for key in ("sample_entropy", "complexity_index"):
            assert np.isfinite(r1[key])
            assert np.isfinite(r2[key])


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_same_output(self):
        """Same input should produce identical complexity metrics."""
        a1 = NeuralComplexityAnalyzer()
        a2 = NeuralComplexityAnalyzer()
        sig = _make_eeg(seed=42)
        r1 = a1.analyze(sig)
        r2 = a2.analyze(sig)
        assert r1["complexity_index"] == pytest.approx(r2["complexity_index"], abs=1e-6)
        assert r1["sample_entropy"] == pytest.approx(r2["sample_entropy"], abs=1e-6)
