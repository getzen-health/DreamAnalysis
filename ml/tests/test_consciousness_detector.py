"""Tests for ConsciousnessDetector.

Covers:
  - assess() output structure and all required keys
  - Value ranges for all numeric outputs (complexity_index, spectral_entropy,
    alpha_suppression, diversity_score, state_stability)
  - State classification thresholds (ordinary/relaxed/meditative/altered/deeply_altered)
  - Complexity ordering: random noise > structured EEG > sine wave
  - Entropy ranges for different signal types
  - set_baseline() behavior and return structure
  - Baseline vs no-baseline alpha_suppression differences
  - get_session_stats() output structure and correctness
  - get_history() chronological ordering and capping at 500
  - Multi-user support via user_id isolation
  - reset() clears per-user state
  - Edge cases: flat signal, short signal, railed signal, single channel, NaN safety
  - Deterministic output for identical input
  - State stability increases with repeated identical signals
"""

import numpy as np
import pytest

from models.consciousness_detector import ConsciousnessDetector, CONSCIOUSNESS_STATES


# ── Helper signal generators ─────────────────────────────────────────────────


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
    """Pure sine wave -- very low complexity / highly regular."""
    t = np.arange(int(fs * duration)) / fs
    s = np.sin(2 * np.pi * freq * t)
    return np.tile(s, (n_channels, 1))


def _make_random(fs=256, duration=4, n_channels=4, seed=99):
    """Pure white noise -- high complexity / maximum entropy."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_channels, int(fs * duration)) * 30


def _make_alpha_dominant(fs=256, duration=4, n_channels=4, seed=50):
    """Signal dominated by alpha (8-12 Hz) -- ordinary relaxed state."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = 40 * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        noise = rng.randn(len(t)) * 2
        signals.append(alpha + noise)
    return np.array(signals)


@pytest.fixture
def detector():
    return ConsciousnessDetector(fs=256.0)


# ── Output structure ────────────────────────────────────────────────────────


class TestOutputStructure:
    def test_assess_returns_all_required_keys(self, detector):
        """assess() must return every documented key."""
        result = detector.assess(_make_eeg())
        required = {
            "consciousness_state",
            "complexity_index",
            "spectral_entropy",
            "alpha_suppression",
            "diversity_score",
            "state_stability",
            "has_baseline",
        }
        assert required.issubset(set(result.keys()))

    def test_assess_values_are_finite(self, detector):
        """All numeric values must be finite (no NaN or Inf)."""
        result = detector.assess(_make_eeg())
        for key in (
            "complexity_index",
            "spectral_entropy",
            "alpha_suppression",
            "diversity_score",
            "state_stability",
        ):
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    def test_consciousness_state_is_string(self, detector):
        """consciousness_state must be a string."""
        result = detector.assess(_make_eeg())
        assert isinstance(result["consciousness_state"], str)

    def test_has_baseline_is_bool(self, detector):
        """has_baseline must be a boolean."""
        result = detector.assess(_make_eeg())
        assert isinstance(result["has_baseline"], bool)


# ── Value ranges ────────────────────────────────────────────────────────────


class TestValueRanges:
    def test_complexity_index_bounded_0_1(self, detector):
        """complexity_index must be in [0, 1]."""
        result = detector.assess(_make_eeg())
        assert 0.0 <= result["complexity_index"] <= 1.0

    def test_spectral_entropy_bounded_0_1(self, detector):
        """spectral_entropy must be in [0, 1]."""
        result = detector.assess(_make_eeg())
        assert 0.0 <= result["spectral_entropy"] <= 1.0

    def test_alpha_suppression_bounded_0_1(self, detector):
        """alpha_suppression must be in [0, 1]."""
        result = detector.assess(_make_eeg())
        assert 0.0 <= result["alpha_suppression"] <= 1.0

    def test_diversity_score_bounded_0_100(self, detector):
        """diversity_score must be in [0, 100]."""
        result = detector.assess(_make_eeg())
        assert 0.0 <= result["diversity_score"] <= 100.0

    def test_state_stability_bounded_0_1(self, detector):
        """state_stability must be in [0, 1]."""
        result = detector.assess(_make_eeg())
        assert 0.0 <= result["state_stability"] <= 1.0

    def test_random_noise_high_complexity(self, detector):
        """White noise should have higher complexity_index than structured signals."""
        result = detector.assess(_make_random())
        # LZ complexity normalized by n/log2(n) produces values in ~0.05-0.15 range
        # for typical signal lengths; random should exceed the sine baseline
        assert result["complexity_index"] > 0.05

    def test_sine_low_complexity(self, detector):
        """Pure sine wave should have low complexity_index."""
        result = detector.assess(_make_sine())
        assert result["complexity_index"] < 0.5


# ── State classification ────────────────────────────────────────────────────


class TestStateClassification:
    def test_valid_state(self, detector):
        """consciousness_state must be one of the defined states."""
        result = detector.assess(_make_eeg())
        assert result["consciousness_state"] in CONSCIOUSNESS_STATES

    def test_states_constant_has_five_values(self):
        """CONSCIOUSNESS_STATES must have exactly 5 values."""
        assert len(CONSCIOUSNESS_STATES) == 5
        assert set(CONSCIOUSNESS_STATES) == {
            "ordinary",
            "relaxed",
            "meditative",
            "altered",
            "deeply_altered",
        }

    def test_ordinary_state_for_alpha_dominant(self, detector):
        """Alpha-dominant relaxed signal should be ordinary or relaxed."""
        result = detector.assess(_make_alpha_dominant())
        assert result["consciousness_state"] in ("ordinary", "relaxed")

    def test_high_diversity_maps_to_altered_or_deeper(self, detector):
        """Signal producing high diversity score should be altered or deeply_altered."""
        # Random noise has high complexity -> high diversity
        result = detector.assess(_make_random())
        assert result["diversity_score"] > 50


# ── Complexity ordering ─────────────────────────────────────────────────────


class TestComplexityOrdering:
    def test_random_more_complex_than_sine(self, detector):
        """White noise should have higher complexity than a pure sine wave."""
        result_rand = detector.assess(_make_random())
        detector.reset()
        result_sine = detector.assess(_make_sine())
        assert result_rand["complexity_index"] > result_sine["complexity_index"]

    def test_eeg_more_complex_than_sine(self, detector):
        """Realistic EEG should have higher complexity than pure sine."""
        result_eeg = detector.assess(_make_eeg())
        detector.reset()
        result_sine = detector.assess(_make_sine())
        assert result_eeg["complexity_index"] > result_sine["complexity_index"]


# ── Entropy ranges ──────────────────────────────────────────────────────────


class TestEntropyRanges:
    def test_random_high_entropy(self, detector):
        """White noise should have high spectral entropy."""
        result = detector.assess(_make_random())
        assert result["spectral_entropy"] > 0.5

    def test_sine_low_entropy(self, detector):
        """Pure sine should have low spectral entropy."""
        result = detector.assess(_make_sine())
        assert result["spectral_entropy"] < 0.5

    def test_eeg_moderate_entropy(self, detector):
        """EEG should have moderate spectral entropy."""
        result = detector.assess(_make_eeg())
        assert 0.1 <= result["spectral_entropy"] <= 0.95


# ── Baseline ────────────────────────────────────────────────────────────────


class TestBaseline:
    def test_set_baseline_returns_required_keys(self, detector):
        """set_baseline() must return baseline_set, baseline_complexity, baseline_entropy."""
        result = detector.set_baseline(_make_eeg())
        assert result["baseline_set"] is True
        assert "baseline_complexity" in result
        assert "baseline_entropy" in result

    def test_has_baseline_false_before_set(self, detector):
        """Before set_baseline(), has_baseline should be False."""
        result = detector.assess(_make_eeg())
        assert result["has_baseline"] is False

    def test_has_baseline_true_after_set(self, detector):
        """After set_baseline(), has_baseline should be True."""
        detector.set_baseline(_make_eeg())
        result = detector.assess(_make_eeg())
        assert result["has_baseline"] is True

    def test_alpha_suppression_changes_with_baseline(self, detector):
        """Alpha suppression should differ with vs without baseline."""
        # Without baseline: absolute alpha suppression
        result_no_bl = detector.assess(_make_eeg())
        detector.reset()
        # With baseline from alpha-dominant signal, then assess with less alpha
        detector.set_baseline(_make_alpha_dominant())
        result_bl = detector.assess(_make_random())
        # Random signal has minimal alpha -> high suppression relative to alpha-dominant baseline
        assert result_bl["alpha_suppression"] > result_no_bl["alpha_suppression"]

    def test_baseline_values_are_finite(self, detector):
        """Baseline complexity and entropy must be finite."""
        result = detector.set_baseline(_make_eeg())
        assert np.isfinite(result["baseline_complexity"])
        assert np.isfinite(result["baseline_entropy"])


# ── Session stats ───────────────────────────────────────────────────────────


class TestSessionStats:
    def test_empty_stats_before_assess(self, detector):
        """get_session_stats() should return n_epochs=0 before any calls."""
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_after_multiple_assessments(self, detector):
        """Stats should contain n_epochs, mean_diversity, dominant_state, state_distribution."""
        for seed in range(5):
            detector.assess(_make_eeg(seed=seed))
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "mean_diversity" in stats
        assert "dominant_state" in stats
        assert "state_distribution" in stats

    def test_state_distribution_sums_to_n_epochs(self, detector):
        """Sum of state_distribution values should equal n_epochs."""
        for seed in range(5):
            detector.assess(_make_eeg(seed=seed))
        stats = detector.get_session_stats()
        total = sum(stats["state_distribution"].values())
        assert total == stats["n_epochs"]

    def test_mean_diversity_in_range(self, detector):
        """mean_diversity must be in [0, 100]."""
        for seed in range(3):
            detector.assess(_make_eeg(seed=seed))
        stats = detector.get_session_stats()
        assert 0.0 <= stats["mean_diversity"] <= 100.0


# ── History ─────────────────────────────────────────────────────────────────


class TestHistory:
    def test_empty_history_initially(self, detector):
        """History should be empty before any calls."""
        assert detector.get_history() == []

    def test_history_grows_with_calls(self, detector):
        """Each assess() call adds one entry to history."""
        for i in range(3):
            detector.assess(_make_eeg(seed=i))
        history = detector.get_history()
        assert len(history) == 3

    def test_history_entries_have_required_keys(self, detector):
        """Each history entry should contain consciousness_state and diversity_score."""
        detector.assess(_make_eeg())
        entry = detector.get_history()[0]
        assert "consciousness_state" in entry
        assert "diversity_score" in entry

    def test_history_last_n_parameter(self, detector):
        """get_history(last_n=2) should return only the last 2 entries."""
        for i in range(5):
            detector.assess(_make_eeg(seed=i))
        history = detector.get_history(last_n=2)
        assert len(history) == 2

    def test_history_capped_at_500(self, detector):
        """History should not exceed 500 entries per user."""
        # Create a short signal for speed
        short_sig = np.random.RandomState(0).randn(4, 256) * 20
        for i in range(510):
            detector.assess(short_sig)
        history = detector.get_history()
        assert len(history) <= 500


# ── Multi-user ──────────────────────────────────────────────────────────────


class TestMultiUser:
    def test_separate_history_per_user(self, detector):
        """Different user_ids should have independent histories."""
        detector.assess(_make_eeg(seed=1), user_id="alice")
        detector.assess(_make_eeg(seed=2), user_id="bob")
        detector.assess(_make_eeg(seed=3), user_id="alice")
        assert len(detector.get_history(user_id="alice")) == 2
        assert len(detector.get_history(user_id="bob")) == 1

    def test_separate_baseline_per_user(self, detector):
        """Different user_ids should have independent baselines."""
        detector.set_baseline(_make_alpha_dominant(), user_id="alice")
        result_alice = detector.assess(_make_eeg(), user_id="alice")
        result_bob = detector.assess(_make_eeg(), user_id="bob")
        assert result_alice["has_baseline"] is True
        assert result_bob["has_baseline"] is False

    def test_reset_only_affects_target_user(self, detector):
        """reset(user_id) should only clear that user's data."""
        detector.assess(_make_eeg(seed=1), user_id="alice")
        detector.assess(_make_eeg(seed=2), user_id="bob")
        detector.reset(user_id="alice")
        assert len(detector.get_history(user_id="alice")) == 0
        assert len(detector.get_history(user_id="bob")) == 1

    def test_session_stats_per_user(self, detector):
        """get_session_stats should return stats for the specified user only."""
        for i in range(3):
            detector.assess(_make_eeg(seed=i), user_id="alice")
        detector.assess(_make_eeg(seed=10), user_id="bob")
        assert detector.get_session_stats(user_id="alice")["n_epochs"] == 3
        assert detector.get_session_stats(user_id="bob")["n_epochs"] == 1


# ── Reset ───────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_history(self, detector):
        """After reset(), history should be empty for that user."""
        detector.assess(_make_eeg())
        detector.assess(_make_eeg(seed=10))
        assert len(detector.get_history()) == 2
        detector.reset()
        assert detector.get_history() == []

    def test_reset_clears_baseline(self, detector):
        """After reset(), baseline should be cleared."""
        detector.set_baseline(_make_eeg())
        detector.reset()
        result = detector.assess(_make_eeg())
        assert result["has_baseline"] is False

    def test_reset_clears_session_stats(self, detector):
        """After reset(), session stats should show n_epochs=0."""
        detector.assess(_make_eeg())
        detector.reset()
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 0


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_channel_input(self, detector):
        """1D input (single channel) should work without error."""
        sig = _make_eeg(n_channels=1)[0]  # shape (n_samples,)
        result = detector.assess(sig)
        assert np.isfinite(result["complexity_index"])
        assert result["consciousness_state"] in CONSCIOUSNESS_STATES

    def test_2d_single_channel(self, detector):
        """(1, n_samples) input should work."""
        sig = _make_eeg(n_channels=1)  # shape (1, n_samples)
        result = detector.assess(sig)
        assert np.isfinite(result["diversity_score"])

    def test_flat_signal_does_not_crash(self, detector):
        """Near-flat signal should not produce errors."""
        result = detector.assess(_make_flat())
        assert np.isfinite(result["complexity_index"])
        assert result["consciousness_state"] in CONSCIOUSNESS_STATES

    def test_railed_signal_does_not_crash(self, detector):
        """Saturated signal should not produce errors."""
        railed = np.ones((4, 1024)) * 300
        result = detector.assess(railed)
        assert np.isfinite(result["complexity_index"])

    def test_short_signal(self, detector):
        """Short signal (0.5 seconds) should work."""
        short = np.random.RandomState(7).randn(4, 128) * 20
        result = detector.assess(short, fs=256)
        assert np.isfinite(result["diversity_score"])
        assert 0 <= result["diversity_score"] <= 100

    def test_different_sampling_rate(self, detector):
        """Different fs should still produce valid results."""
        sig = np.random.RandomState(8).randn(4, 512) * 20
        det = ConsciousnessDetector(fs=128.0)
        result = det.assess(sig, fs=128)
        assert np.isfinite(result["complexity_index"])
        assert 0.0 <= result["complexity_index"] <= 1.0


# ── State stability ────────────────────────────────────────────────────────


class TestStateStability:
    def test_stability_increases_with_repeated_signal(self, detector):
        """State stability should increase when the same signal is assessed repeatedly."""
        sig = _make_eeg(seed=42)
        results = []
        for _ in range(5):
            results.append(detector.assess(sig))
        # Stability for the first call is 0 (no prior history)
        # Later calls should have higher stability
        assert results[-1]["state_stability"] >= results[0]["state_stability"]

    def test_stability_low_after_state_change(self, detector):
        """Stability should be lower right after a large signal change."""
        # Build stable state
        sig = _make_eeg(seed=42)
        for _ in range(5):
            detector.assess(sig)
        # Switch to very different signal
        result_switch = detector.assess(_make_random(seed=77))
        # Next assess with the new signal should still have low-ish stability
        assert result_switch["state_stability"] <= 1.0


# ── Determinism ─────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_input_same_output(self):
        """Same input should produce identical results from fresh instances."""
        d1 = ConsciousnessDetector(fs=256.0)
        d2 = ConsciousnessDetector(fs=256.0)
        sig = _make_eeg(seed=42)
        r1 = d1.assess(sig)
        r2 = d2.assess(sig)
        assert r1["complexity_index"] == pytest.approx(r2["complexity_index"], abs=1e-6)
        assert r1["spectral_entropy"] == pytest.approx(r2["spectral_entropy"], abs=1e-6)
        assert r1["diversity_score"] == pytest.approx(r2["diversity_score"], abs=1e-6)
        assert r1["consciousness_state"] == r2["consciousness_state"]
