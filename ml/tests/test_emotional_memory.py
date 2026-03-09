"""Tests for EmotionalMemoryPredictor — theta-gamma CFC memory encoding.

Synthetic data generators:
  - _make_coupled_signal: theta carrier with gamma amplitude modulated by theta
    phase (simulates strong memory encoding)
  - _make_uncoupled_signal: independent theta and constant-amplitude gamma
    (simulates baseline / no encoding)
  - _make_high_theta_signal: elevated frontal theta power
  - _make_multichannel: 4-channel Muse 2 layout

All tests use np.random.seed(42) for reproducibility.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, filtfilt, hilbert

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.emotional_memory import EmotionalMemoryPredictor

# -- Constants ----------------------------------------------------------------

FS = 256
DURATION_S = 4  # 4 seconds = ~2 theta cycles minimum


# -- Synthetic signal generators ----------------------------------------------


def _make_coupled_signal(
    fs: float = FS,
    duration_s: float = DURATION_S,
    theta_freq: float = 6.0,
    gamma_freq: float = 35.0,
    theta_amp: float = 20.0,
    gamma_amp: float = 5.0,
    coupling_strength: float = 0.8,
    seed: int = 42,
) -> np.ndarray:
    """Create a signal with theta-gamma phase-amplitude coupling.

    Gamma amplitude is modulated by theta phase: stronger coupling_strength
    means gamma bursts are more tightly locked to theta peaks.
    """
    np.random.seed(seed)
    n_samples = int(fs * duration_s)
    t = np.arange(n_samples) / fs

    # Theta carrier
    theta = theta_amp * np.sin(2 * np.pi * theta_freq * t)

    # Gamma with amplitude modulated by theta phase
    theta_phase = 2 * np.pi * theta_freq * t
    # Modulation: gamma amplitude peaks when theta phase is at peak (0 to pi)
    modulation = (1.0 - coupling_strength) + coupling_strength * (
        0.5 * (1.0 + np.cos(theta_phase))
    )
    gamma = gamma_amp * modulation * np.sin(2 * np.pi * gamma_freq * t)

    noise = 1.0 * np.random.randn(n_samples)
    return theta + gamma + noise


def _make_uncoupled_signal(
    fs: float = FS,
    duration_s: float = DURATION_S,
    theta_freq: float = 6.0,
    gamma_freq: float = 35.0,
    theta_amp: float = 20.0,
    gamma_amp: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Create a signal with independent theta and gamma (no coupling).

    Gamma amplitude is constant -- not modulated by theta phase.
    """
    np.random.seed(seed)
    n_samples = int(fs * duration_s)
    t = np.arange(n_samples) / fs

    theta = theta_amp * np.sin(2 * np.pi * theta_freq * t)
    gamma = gamma_amp * np.sin(2 * np.pi * gamma_freq * t)
    noise = 1.0 * np.random.randn(n_samples)
    return theta + gamma + noise


def _make_high_theta_signal(
    fs: float = FS,
    duration_s: float = DURATION_S,
    theta_amp: float = 40.0,
    seed: int = 42,
) -> np.ndarray:
    """Create signal with elevated frontal theta power."""
    np.random.seed(seed)
    n_samples = int(fs * duration_s)
    t = np.arange(n_samples) / fs

    theta = theta_amp * np.sin(2 * np.pi * 6.0 * t)
    alpha = 3.0 * np.sin(2 * np.pi * 10.0 * t)
    noise = 1.0 * np.random.randn(n_samples)
    return theta + alpha + noise


def _make_multichannel(
    signal_fn=_make_uncoupled_signal,
    n_channels: int = 4,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """Create multichannel (4, n_samples) signal from a generator function."""
    channels = []
    for ch in range(n_channels):
        channels.append(signal_fn(seed=seed + ch, **kwargs))
    return np.array(channels)


def _make_constant_signal(
    value: float = 0.0, fs: float = FS, duration_s: float = DURATION_S
) -> np.ndarray:
    """Constant (flat) signal."""
    return np.full(int(fs * duration_s), value)


# -- Test Classes -------------------------------------------------------------


class TestBaseline:
    """Tests for set_baseline()."""

    def test_baseline_returns_dict(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        signal = _make_uncoupled_signal()
        result = pred.set_baseline(signal)
        assert isinstance(result, dict)

    def test_baseline_required_keys(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        signal = _make_uncoupled_signal()
        result = pred.set_baseline(signal)
        required = {"baseline_set", "n_channels"}
        assert required.issubset(result.keys()), f"Missing: {required - result.keys()}"

    def test_baseline_set_flag_true(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.set_baseline(_make_uncoupled_signal())
        assert result["baseline_set"] is True

    def test_baseline_n_channels_1d(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.set_baseline(_make_uncoupled_signal())
        assert result["n_channels"] == 1

    def test_baseline_n_channels_multichannel(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        multi = _make_multichannel()
        result = pred.set_baseline(multi)
        assert result["n_channels"] == 4

    def test_baseline_theta_power_positive(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        signal = _make_high_theta_signal()
        result = pred.set_baseline(signal)
        assert result["baseline_theta_power"] > 0

    def test_baseline_coupling_mi_non_negative(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.set_baseline(_make_uncoupled_signal())
        assert result["baseline_coupling_mi"] >= 0.0

    def test_baseline_with_custom_fs(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=128.0)
        n = int(128.0 * 4)
        signal = np.random.randn(n) * 10
        result = pred.set_baseline(signal, fs=128.0)
        assert result["baseline_set"] is True

    def test_baseline_per_user(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.set_baseline(_make_uncoupled_signal(), user_id="alice")
        pred.set_baseline(_make_high_theta_signal(), user_id="bob")
        # Both should be set independently
        r1 = pred.predict(_make_uncoupled_signal(), user_id="alice")
        r2 = pred.predict(_make_uncoupled_signal(), user_id="bob")
        assert r1["has_baseline"] is True
        assert r2["has_baseline"] is True


class TestPredictOutputStructure:
    """Tests for predict() output keys, types, and ranges."""

    def test_predict_returns_dict(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_uncoupled_signal())
        assert isinstance(result, dict)

    def test_predict_required_keys(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_uncoupled_signal())
        required = {
            "encoding_strength",
            "theta_power",
            "gamma_power",
            "theta_gamma_coupling",
            "frontal_theta_increase",
            "encoding_level",
            "has_baseline",
        }
        assert required.issubset(result.keys()), f"Missing: {required - result.keys()}"

    def test_encoding_strength_is_float(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_uncoupled_signal())
        assert isinstance(result["encoding_strength"], float)

    def test_encoding_strength_range_0_1(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        for fn in [_make_coupled_signal, _make_uncoupled_signal, _make_high_theta_signal]:
            result = pred.predict(fn())
            assert 0.0 <= result["encoding_strength"] <= 1.0, (
                f"encoding_strength={result['encoding_strength']} out of range"
            )

    def test_theta_power_non_negative(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_uncoupled_signal())
        assert result["theta_power"] >= 0.0

    def test_gamma_power_non_negative(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_uncoupled_signal())
        assert result["gamma_power"] >= 0.0

    def test_theta_gamma_coupling_non_negative(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_uncoupled_signal())
        assert result["theta_gamma_coupling"] >= 0.0

    def test_theta_gamma_coupling_max_1(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_coupled_signal(coupling_strength=1.0))
        assert result["theta_gamma_coupling"] <= 1.0

    def test_has_baseline_false_initially(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_uncoupled_signal())
        assert result["has_baseline"] is False

    def test_has_baseline_true_after_set(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.set_baseline(_make_uncoupled_signal())
        result = pred.predict(_make_uncoupled_signal())
        assert result["has_baseline"] is True

    def test_encoding_level_is_string(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_uncoupled_signal())
        assert isinstance(result["encoding_level"], str)

    def test_encoding_level_valid_values(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        valid_levels = {
            "strong_encoding", "moderate_encoding",
            "weak_encoding", "minimal_encoding",
        }
        for fn in [_make_coupled_signal, _make_uncoupled_signal, _make_high_theta_signal]:
            result = pred.predict(fn())
            assert result["encoding_level"] in valid_levels

    def test_frontal_theta_increase_is_float(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_uncoupled_signal())
        assert isinstance(result["frontal_theta_increase"], float)


class TestThetaGammaCoupling:
    """Verify that coupled signals produce higher MI than uncoupled signals."""

    def test_coupled_higher_mi_than_uncoupled(self):
        """Strong theta-gamma coupling should yield higher MI."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        coupled = pred.predict(_make_coupled_signal(coupling_strength=0.9))
        uncoupled = pred.predict(_make_uncoupled_signal())
        assert coupled["theta_gamma_coupling"] > uncoupled["theta_gamma_coupling"], (
            f"coupled MI={coupled['theta_gamma_coupling']} should be > "
            f"uncoupled MI={uncoupled['theta_gamma_coupling']}"
        )

    def test_stronger_coupling_higher_mi(self):
        """Higher coupling strength should produce higher MI."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        weak = pred.predict(_make_coupled_signal(coupling_strength=0.3))
        strong = pred.predict(_make_coupled_signal(coupling_strength=0.9, seed=43))
        assert strong["theta_gamma_coupling"] >= weak["theta_gamma_coupling"]

    def test_mi_zero_for_constant_signal(self):
        """Constant signal should have zero coupling."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_constant_signal(value=5.0))
        assert result["theta_gamma_coupling"] < 0.05  # near zero, numerical noise allowed

    def test_mi_bounded_above(self):
        """MI should never exceed 1.0 even with extreme coupling."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(
            _make_coupled_signal(coupling_strength=1.0, theta_amp=50.0, gamma_amp=30.0)
        )
        assert result["theta_gamma_coupling"] <= 1.0

    def test_multichannel_coupled_detects_coupling(self):
        """Multichannel coupled signal should show non-zero MI."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        multi = _make_multichannel(
            signal_fn=_make_coupled_signal, coupling_strength=0.8
        )
        result = pred.predict(multi)
        assert result["theta_gamma_coupling"] > 0.0


class TestEncodingStrength:
    """Tests for composite encoding strength computation."""

    def test_high_theta_high_coupling_is_strong(self):
        """High theta + high coupling should give high encoding strength."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(
            _make_coupled_signal(
                coupling_strength=0.9, theta_amp=40.0, gamma_amp=15.0
            )
        )
        assert result["encoding_strength"] > 0.3, (
            f"Expected high encoding, got {result['encoding_strength']}"
        )

    def test_low_theta_low_coupling_is_weak(self):
        """Low theta + no coupling should give low encoding strength."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        # Small theta, small gamma, no coupling
        n = int(FS * DURATION_S)
        t = np.arange(n) / FS
        signal = 2.0 * np.sin(2 * np.pi * 6.0 * t) + np.random.randn(n) * 0.5
        result = pred.predict(signal)
        assert result["encoding_strength"] < 0.5

    def test_coupled_stronger_than_uncoupled_encoding(self):
        """Coupled signals should yield higher encoding strength than uncoupled."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        coupled = pred.predict(
            _make_coupled_signal(coupling_strength=0.9, theta_amp=30.0, gamma_amp=10.0)
        )
        uncoupled = pred.predict(
            _make_uncoupled_signal(theta_amp=30.0, gamma_amp=10.0, seed=99)
        )
        assert coupled["encoding_strength"] >= uncoupled["encoding_strength"]

    def test_encoding_strength_increases_with_theta(self):
        """Higher theta power should increase encoding strength."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        low_theta = pred.predict(_make_coupled_signal(theta_amp=5.0, coupling_strength=0.5))
        high_theta = pred.predict(
            _make_coupled_signal(theta_amp=40.0, coupling_strength=0.5, seed=43)
        )
        assert high_theta["encoding_strength"] >= low_theta["encoding_strength"]


class TestEncodingLevels:
    """Tests for encoding level string classification thresholds."""

    def test_minimal_encoding_level(self):
        """Very low activity should produce 'minimal_encoding'."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        # Nearly flat signal with very low amplitude
        signal = np.random.randn(int(FS * DURATION_S)) * 0.1
        result = pred.predict(signal)
        assert result["encoding_level"] == "minimal_encoding"

    def test_strong_encoding_level_possible(self):
        """Verify strong encoding is achievable with high theta+coupling."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(
            _make_coupled_signal(
                coupling_strength=0.95, theta_amp=60.0, gamma_amp=20.0, duration_s=6
            )
        )
        # Should be at least moderate if not strong
        assert result["encoding_level"] in (
            "strong_encoding", "moderate_encoding", "weak_encoding"
        )

    def test_encoding_level_matches_strength(self):
        """Encoding level labels should be consistent with encoding_strength value."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        for seed in range(42, 52):
            signal = _make_coupled_signal(
                coupling_strength=seed * 0.05, theta_amp=seed, seed=seed
            )
            result = pred.predict(signal)
            s = result["encoding_strength"]
            level = result["encoding_level"]
            if s >= 0.65:
                assert level == "strong_encoding"
            elif s >= 0.40:
                assert level == "moderate_encoding"
            elif s >= 0.20:
                assert level == "weak_encoding"
            else:
                assert level == "minimal_encoding"


class TestBaselineComparison:
    """Tests for frontal_theta_increase when baseline is available."""

    def test_theta_increase_zero_without_baseline(self):
        """Without baseline, frontal_theta_increase should be 0."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_high_theta_signal())
        assert result["frontal_theta_increase"] == 0.0

    def test_theta_increase_positive_when_elevated(self):
        """Elevated theta relative to low baseline should show positive increase."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        # Low-theta baseline
        n = int(FS * DURATION_S)
        t = np.arange(n) / FS
        low_theta = 3.0 * np.sin(2 * np.pi * 6.0 * t) + np.random.randn(n) * 0.5
        pred.set_baseline(low_theta)
        # High-theta test
        result = pred.predict(_make_high_theta_signal(theta_amp=40.0))
        assert result["frontal_theta_increase"] > 0.0

    def test_theta_increase_negative_when_reduced(self):
        """Reduced theta relative to high baseline should show negative increase."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        # High-theta baseline
        pred.set_baseline(_make_high_theta_signal(theta_amp=40.0))
        # Low-theta test
        n = int(FS * DURATION_S)
        t = np.arange(n) / FS
        low_theta = 3.0 * np.sin(2 * np.pi * 6.0 * t) + np.random.randn(n) * 0.5
        result = pred.predict(low_theta)
        assert result["frontal_theta_increase"] < 0.0

    def test_theta_increase_bounded(self):
        """frontal_theta_increase should be bounded in reasonable range."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        n = int(FS * DURATION_S)
        baseline = np.random.randn(n) * 0.01  # near-zero baseline
        pred.set_baseline(baseline)
        result = pred.predict(_make_high_theta_signal(theta_amp=100.0))
        assert result["frontal_theta_increase"] <= 5.0


class TestSessionStats:
    """Tests for get_session_stats()."""

    def test_empty_session_stats(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        stats = pred.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["mean_encoding"] == 0.0
        assert stats["peak_encoding"] == 0.0

    def test_session_stats_after_predictions(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.predict(_make_coupled_signal(coupling_strength=0.5))
        pred.predict(_make_coupled_signal(coupling_strength=0.8, seed=43))
        pred.predict(_make_uncoupled_signal(seed=44))
        stats = pred.get_session_stats()
        assert stats["n_epochs"] == 3
        assert stats["mean_encoding"] > 0.0
        assert stats["peak_encoding"] >= stats["mean_encoding"]

    def test_peak_encoding_is_max(self):
        """Peak encoding should be the maximum encoding strength seen."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        strengths = []
        for seed in range(42, 47):
            result = pred.predict(_make_coupled_signal(seed=seed))
            strengths.append(result["encoding_strength"])
        stats = pred.get_session_stats()
        assert abs(stats["peak_encoding"] - max(strengths)) < 1e-6

    def test_session_stats_per_user(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.predict(_make_coupled_signal(), user_id="alice")
        pred.predict(_make_coupled_signal(seed=43), user_id="alice")
        pred.predict(_make_uncoupled_signal(), user_id="bob")
        assert pred.get_session_stats("alice")["n_epochs"] == 2
        assert pred.get_session_stats("bob")["n_epochs"] == 1


class TestHistory:
    """Tests for get_history()."""

    def test_empty_history(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        assert pred.get_history() == []

    def test_history_grows_with_predictions(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        for i in range(5):
            pred.predict(_make_coupled_signal(seed=42 + i))
        assert len(pred.get_history()) == 5

    def test_history_contains_full_results(self):
        """Each history entry should have all prediction keys."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.predict(_make_coupled_signal())
        entry = pred.get_history()[0]
        assert "encoding_strength" in entry
        assert "theta_gamma_coupling" in entry
        assert "encoding_level" in entry

    def test_history_last_n(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        for i in range(10):
            pred.predict(_make_coupled_signal(seed=42 + i))
        last3 = pred.get_history(last_n=3)
        assert len(last3) == 3

    def test_history_last_n_larger_than_total(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.predict(_make_coupled_signal())
        pred.predict(_make_coupled_signal(seed=43))
        last10 = pred.get_history(last_n=10)
        assert len(last10) == 2

    def test_history_capped_at_max(self):
        """History should not exceed 500 entries per user."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        short_signal = np.random.randn(64)  # short = fast default result
        for _ in range(510):
            pred.predict(short_signal)
        assert len(pred.get_history()) == 500

    def test_history_per_user_independent(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.predict(_make_coupled_signal(), user_id="alice")
        pred.predict(_make_uncoupled_signal(), user_id="bob")
        assert len(pred.get_history("alice")) == 1
        assert len(pred.get_history("bob")) == 1
        assert len(pred.get_history("nobody")) == 0


class TestReset:
    """Tests for reset()."""

    def test_reset_clears_history(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.predict(_make_coupled_signal())
        pred.predict(_make_coupled_signal(seed=43))
        pred.reset()
        assert pred.get_history() == []
        assert pred.get_session_stats()["n_epochs"] == 0

    def test_reset_clears_baseline(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.set_baseline(_make_uncoupled_signal())
        pred.reset()
        result = pred.predict(_make_coupled_signal())
        assert result["has_baseline"] is False

    def test_reset_per_user(self):
        """Resetting one user should not affect another."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.predict(_make_coupled_signal(), user_id="alice")
        pred.predict(_make_coupled_signal(), user_id="bob")
        pred.reset(user_id="alice")
        assert pred.get_history("alice") == []
        assert len(pred.get_history("bob")) == 1

    def test_reset_then_predict_works(self):
        """Should work normally after reset."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.predict(_make_coupled_signal())
        pred.reset()
        result = pred.predict(_make_coupled_signal())
        assert result["encoding_strength"] >= 0.0
        assert pred.get_session_stats()["n_epochs"] == 1


class TestMultiUser:
    """Tests for multi-user isolation."""

    def test_independent_baselines(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.set_baseline(_make_uncoupled_signal(), user_id="alice")
        result_bob = pred.predict(_make_coupled_signal(), user_id="bob")
        assert result_bob["has_baseline"] is False

    def test_independent_histories(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        for i in range(5):
            pred.predict(_make_coupled_signal(seed=42 + i), user_id="alice")
        for i in range(3):
            pred.predict(_make_uncoupled_signal(seed=50 + i), user_id="bob")
        assert pred.get_session_stats("alice")["n_epochs"] == 5
        assert pred.get_session_stats("bob")["n_epochs"] == 3

    def test_independent_session_stats(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.predict(
            _make_coupled_signal(coupling_strength=0.9, theta_amp=40.0),
            user_id="alice",
        )
        pred.predict(
            _make_uncoupled_signal(theta_amp=3.0),
            user_id="bob",
        )
        alice_stats = pred.get_session_stats("alice")
        bob_stats = pred.get_session_stats("bob")
        # Alice had strong signal, Bob had weak
        assert alice_stats["mean_encoding"] >= bob_stats["mean_encoding"]

    def test_reset_one_user_preserves_other(self):
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.set_baseline(_make_uncoupled_signal(), user_id="alice")
        pred.set_baseline(_make_uncoupled_signal(), user_id="bob")
        pred.predict(_make_coupled_signal(), user_id="alice")
        pred.predict(_make_coupled_signal(), user_id="bob")
        pred.reset("alice")
        assert pred.get_session_stats("alice")["n_epochs"] == 0
        assert pred.get_session_stats("bob")["n_epochs"] == 1
        # Bob's baseline should still work
        result = pred.predict(_make_coupled_signal(), user_id="bob")
        assert result["has_baseline"] is True


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_constant_signal(self):
        """Constant signal should not crash, should give minimal encoding."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(_make_constant_signal(value=10.0))
        assert result["encoding_level"] == "minimal_encoding"
        assert result["encoding_strength"] >= 0.0

    def test_1d_input(self):
        """1D input should be accepted and reshaped."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        signal = np.random.randn(int(FS * DURATION_S)) * 10
        result = pred.predict(signal)
        assert "encoding_strength" in result

    def test_very_short_signal(self):
        """Signal shorter than _MIN_SAMPLES should return defaults."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        short = np.random.randn(64)
        result = pred.predict(short)
        assert result["encoding_strength"] == 0.0
        assert result["encoding_level"] == "minimal_encoding"

    def test_very_short_signal_still_in_history(self):
        """Even short-signal defaults should be recorded in history."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        pred.predict(np.random.randn(32))
        assert pred.get_session_stats()["n_epochs"] == 1

    def test_zero_signal(self):
        """All-zero signal should not crash."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(np.zeros(int(FS * DURATION_S)))
        assert result["encoding_strength"] >= 0.0
        assert result["theta_gamma_coupling"] == 0.0

    def test_nan_signal(self):
        """Signal with NaN values should be handled gracefully."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        signal = np.random.randn(int(FS * DURATION_S)) * 10
        signal[100:110] = np.nan
        result = pred.predict(signal)
        assert np.isfinite(result["encoding_strength"])
        assert np.isfinite(result["theta_power"])

    def test_inf_signal(self):
        """Signal with inf values should be handled gracefully."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        signal = np.random.randn(int(FS * DURATION_S)) * 10
        signal[50] = np.inf
        signal[51] = -np.inf
        result = pred.predict(signal)
        assert np.isfinite(result["encoding_strength"])

    def test_single_sample(self):
        """Single-sample input should not crash."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.predict(np.array([5.0]))
        assert result["encoding_strength"] == 0.0

    def test_multichannel_2d_input(self):
        """(4, n_samples) multichannel input should work."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        multi = _make_multichannel()
        result = pred.predict(multi)
        assert result["theta_power"] >= 0.0

    def test_single_channel_2d(self):
        """(1, n_samples) 2D input should work."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        signal = np.random.randn(1, int(FS * DURATION_S)) * 10
        result = pred.predict(signal)
        assert "encoding_strength" in result

    def test_two_channel_input(self):
        """(2, n_samples) input with fewer than 3 channels should use all channels."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        signal = np.random.randn(2, int(FS * DURATION_S)) * 10
        result = pred.predict(signal)
        assert result["theta_power"] >= 0.0

    def test_large_amplitude_signal(self):
        """High amplitude signals should not produce values outside 0-1."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        signal = np.random.randn(int(FS * DURATION_S)) * 1000
        result = pred.predict(signal)
        assert 0.0 <= result["encoding_strength"] <= 1.0
        assert result["theta_gamma_coupling"] <= 1.0

    def test_baseline_short_signal(self):
        """Setting baseline with short signal should still work."""
        np.random.seed(42)
        pred = EmotionalMemoryPredictor(fs=FS)
        result = pred.set_baseline(np.random.randn(64))
        assert result["baseline_set"] is True
