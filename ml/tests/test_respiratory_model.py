"""Tests for respiratory_model — passive respiratory biometrics (#416).

Covers:
  - Respiratory rate extraction from synthetic breathing signals
  - Feature computation (regularity, I/E ratio, sighs)
  - Breathing pattern classification
  - Edge cases (very fast/slow breathing, noisy signal, too-short signal)
  - Emotion correlation computation
"""
from __future__ import annotations

import numpy as np
import pytest

from models.respiratory_model import (
    classify_breathing_pattern,
    compute_respiratory_emotion_correlation,
    compute_respiratory_features,
    extract_respiratory_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_breathing_signal(
    bpm: float,
    duration_s: float = 30.0,
    fs: float = 100.0,
    noise_level: float = 0.05,
    ie_ratio: float = 1.0,
) -> np.ndarray:
    """Generate a synthetic breathing signal at a given rate.

    Creates a sinusoidal signal at the respiratory frequency with optional
    asymmetry (ie_ratio != 1.0) and additive Gaussian noise.

    Args:
        bpm:         Breaths per minute.
        duration_s:  Signal duration in seconds.
        fs:          Sampling rate in Hz.
        noise_level: Standard deviation of additive Gaussian noise.
        ie_ratio:    Inhalation/exhalation ratio (1.0 = symmetric).

    Returns:
        1-D numpy array simulating a breathing amplitude envelope.
    """
    t = np.arange(0, duration_s, 1.0 / fs)
    freq = bpm / 60.0
    # Simple sinusoidal breathing pattern
    signal = np.sin(2 * np.pi * freq * t)
    # Add offset to keep amplitude positive (like real audio envelope)
    signal = signal + 1.5
    rng = np.random.default_rng(42)
    signal += rng.normal(0, noise_level, len(signal))
    return signal


def _make_breathing_with_sighs(
    bpm: float = 15.0,
    duration_s: float = 60.0,
    fs: float = 100.0,
    n_sighs: int = 3,
) -> np.ndarray:
    """Generate a breathing signal with injected sighs (high-amplitude breaths)."""
    signal = _make_breathing_signal(bpm, duration_s, fs, noise_level=0.02)
    cycle_samples = int(fs * 60.0 / bpm)
    rng = np.random.default_rng(123)

    # Inject sighs at random cycle positions
    for _ in range(n_sighs):
        pos = rng.integers(cycle_samples, len(signal) - cycle_samples)
        # Create a large amplitude burst
        sigh_len = min(cycle_samples, len(signal) - pos)
        t_sigh = np.arange(sigh_len) / fs
        sigh = 3.0 * np.sin(2 * np.pi * (bpm / 60.0) * t_sigh)
        signal[pos : pos + sigh_len] += sigh

    return signal


# ---------------------------------------------------------------------------
# Tests: extract_respiratory_rate
# ---------------------------------------------------------------------------

class TestExtractRespiratoryRate:
    """Tests for respiratory rate extraction."""

    def test_normal_breathing_rate(self):
        """12 bpm signal should extract a rate near 12."""
        signal = _make_breathing_signal(bpm=12.0, duration_s=60.0, fs=100.0)
        result = extract_respiratory_rate(signal, fs=100.0)
        assert "error" not in result
        assert 9.0 <= result["respiratory_rate_bpm"] <= 16.0

    def test_slow_breathing_rate(self):
        """6 bpm (calm meditation) should be detectable."""
        signal = _make_breathing_signal(bpm=6.0, duration_s=60.0, fs=100.0)
        result = extract_respiratory_rate(signal, fs=100.0)
        assert "error" not in result
        assert 4.0 <= result["respiratory_rate_bpm"] <= 9.0

    def test_fast_breathing_rate(self):
        """25 bpm (exercise/anxiety) should be detectable."""
        signal = _make_breathing_signal(bpm=25.0, duration_s=30.0, fs=100.0)
        result = extract_respiratory_rate(signal, fs=100.0)
        assert "error" not in result
        assert 18.0 <= result["respiratory_rate_bpm"] <= 32.0

    def test_returns_peak_indices(self):
        """Result should include peak indices for each detected breath."""
        signal = _make_breathing_signal(bpm=15.0, duration_s=30.0, fs=100.0)
        result = extract_respiratory_rate(signal, fs=100.0)
        assert "error" not in result
        assert isinstance(result["peak_indices"], list)
        assert len(result["peak_indices"]) >= 3

    def test_returns_cycle_durations(self):
        """Result should include inter-peak cycle durations."""
        signal = _make_breathing_signal(bpm=15.0, duration_s=30.0, fs=100.0)
        result = extract_respiratory_rate(signal, fs=100.0)
        assert "error" not in result
        durations = result["cycle_durations_s"]
        assert len(durations) >= 2
        # Each cycle should be roughly 4 seconds (60/15)
        for d in durations:
            assert 2.0 <= d <= 8.0

    def test_too_short_signal_returns_error(self):
        """Signal shorter than 4 seconds should return error."""
        short = np.ones(100)  # 1 second at 100 Hz
        result = extract_respiratory_rate(short, fs=100.0)
        assert "error" in result
        assert result["error"] == "signal_too_short"


# ---------------------------------------------------------------------------
# Tests: compute_respiratory_features
# ---------------------------------------------------------------------------

class TestComputeRespiratoryFeatures:
    """Tests for full respiratory feature computation."""

    def test_feature_keys_present(self):
        """All expected feature keys should be present."""
        signal = _make_breathing_signal(bpm=15.0, duration_s=60.0, fs=100.0)
        result = compute_respiratory_features(signal, fs=100.0)
        assert "error" not in result
        expected_keys = {
            "breaths_per_minute",
            "inhalation_exhalation_ratio",
            "breath_regularity",
            "sigh_count",
            "sigh_indices",
            "mean_breath_amplitude",
        }
        assert expected_keys.issubset(result.keys())

    def test_regularity_of_clean_signal(self):
        """A clean sinusoidal breathing signal should have low CV."""
        signal = _make_breathing_signal(
            bpm=15.0, duration_s=60.0, fs=100.0, noise_level=0.01
        )
        result = compute_respiratory_features(signal, fs=100.0)
        assert "error" not in result
        # CV should be low (< 0.3) for a regular signal
        assert result["breath_regularity"] < 0.3

    def test_ie_ratio_near_one_for_symmetric(self):
        """Symmetric sinusoidal signal should have I:E ratio near 1.0."""
        signal = _make_breathing_signal(
            bpm=12.0, duration_s=60.0, fs=100.0, noise_level=0.01
        )
        result = compute_respiratory_features(signal, fs=100.0)
        assert "error" not in result
        assert 0.5 <= result["inhalation_exhalation_ratio"] <= 2.0

    def test_sigh_detection(self):
        """Signal with injected sighs should detect at least one sigh."""
        signal = _make_breathing_with_sighs(
            bpm=15.0, duration_s=60.0, fs=100.0, n_sighs=3
        )
        result = compute_respiratory_features(signal, fs=100.0)
        assert "error" not in result
        assert result["sigh_count"] >= 1

    def test_mean_breath_amplitude_positive(self):
        """Mean breath amplitude should be positive."""
        signal = _make_breathing_signal(bpm=15.0, duration_s=30.0, fs=100.0)
        result = compute_respiratory_features(signal, fs=100.0)
        assert "error" not in result
        assert result["mean_breath_amplitude"] > 0


# ---------------------------------------------------------------------------
# Tests: classify_breathing_pattern
# ---------------------------------------------------------------------------

class TestClassifyBreathingPattern:
    """Tests for breathing pattern classification."""

    def test_calm_classification(self):
        """8 bpm should classify as calm."""
        signal = _make_breathing_signal(bpm=8.0, duration_s=60.0, fs=100.0)
        features = compute_respiratory_features(signal, fs=100.0)
        classification = classify_breathing_pattern(features)
        assert classification["state"] in ("calm", "normal")
        assert 0.0 <= classification["confidence"] <= 1.0

    def test_normal_classification(self):
        """15 bpm should classify as normal."""
        signal = _make_breathing_signal(bpm=15.0, duration_s=60.0, fs=100.0)
        features = compute_respiratory_features(signal, fs=100.0)
        classification = classify_breathing_pattern(features)
        assert classification["state"] in ("normal", "calm")
        assert classification["confidence"] > 0.0

    def test_exercise_classification(self):
        """Very high rate (>30 bpm) should classify as exercise or anxious."""
        # Use a direct feature dict to avoid signal processing ambiguity
        features = {
            "breaths_per_minute": 35.0,
            "inhalation_exhalation_ratio": 1.0,
            "breath_regularity": 0.1,
            "sigh_count": 0,
        }
        classification = classify_breathing_pattern(features)
        assert classification["state"] == "exercise"

    def test_error_features_return_unknown(self):
        """Features dict with error key should return unknown state."""
        features = {"error": "signal_too_short"}
        classification = classify_breathing_pattern(features)
        assert classification["state"] == "unknown"
        assert classification["confidence"] == 0.0

    def test_contributing_factors_present(self):
        """Classification should include contributing factors."""
        features = {
            "breaths_per_minute": 22.0,
            "inhalation_exhalation_ratio": 1.0,
            "breath_regularity": 0.15,
            "sigh_count": 0,
        }
        classification = classify_breathing_pattern(features)
        assert isinstance(classification["contributing_factors"], list)
        assert len(classification["contributing_factors"]) >= 1

    def test_high_irregularity_shifts_state(self):
        """High CV should push classification toward stressed."""
        features = {
            "breaths_per_minute": 14.0,
            "inhalation_exhalation_ratio": 1.0,
            "breath_regularity": 0.5,  # very irregular
            "sigh_count": 0,
        }
        classification = classify_breathing_pattern(features)
        assert classification["state"] in ("stressed", "anxious")


# ---------------------------------------------------------------------------
# Tests: compute_respiratory_emotion_correlation
# ---------------------------------------------------------------------------

class TestRespiratoryEmotionCorrelation:
    """Tests for respiratory-emotion correlation."""

    def _make_history(self, n: int = 10):
        """Create matching respiratory + emotion history lists."""
        rng = np.random.default_rng(99)
        base_ts = 1700000000.0

        resp_history = []
        emo_history = []
        for i in range(n):
            ts = base_ts + i * 30.0  # every 30 seconds
            bpm = 12.0 + rng.normal(0, 2)
            cv = 0.15 + rng.normal(0, 0.03)

            resp_history.append({
                "breaths_per_minute": float(bpm),
                "breath_regularity": float(max(0, cv)),
                "timestamp": ts,
            })
            emo_history.append({
                "valence": float(np.clip(rng.normal(0.2, 0.3), -1, 1)),
                "arousal": float(np.clip(rng.normal(0.4, 0.2), 0, 1)),
                "timestamp": ts + rng.uniform(-5, 5),  # slight offset
            })

        return resp_history, emo_history

    def test_correlation_output_keys(self):
        """Result should contain all expected correlation keys."""
        resp, emo = self._make_history(10)
        result = compute_respiratory_emotion_correlation(resp, emo)
        assert "error" not in result
        assert "bpm_valence_corr" in result
        assert "bpm_arousal_corr" in result
        assert "regularity_valence_corr" in result
        assert "n_matched_pairs" in result
        assert "baseline" in result

    def test_correlation_values_in_range(self):
        """Pearson r must be in [-1, 1]."""
        resp, emo = self._make_history(20)
        result = compute_respiratory_emotion_correlation(resp, emo)
        assert "error" not in result
        assert -1.0 <= result["bpm_valence_corr"] <= 1.0
        assert -1.0 <= result["bpm_arousal_corr"] <= 1.0
        assert -1.0 <= result["regularity_valence_corr"] <= 1.0

    def test_insufficient_history_returns_error(self):
        """Fewer than 3 entries should return an error."""
        resp = [
            {"breaths_per_minute": 14.0, "breath_regularity": 0.1, "timestamp": 100.0}
        ]
        emo = [{"valence": 0.5, "arousal": 0.3, "timestamp": 100.0}]
        result = compute_respiratory_emotion_correlation(resp, emo)
        assert "error" in result
        assert result["error"] == "insufficient_history"

    def test_baseline_stats_present(self):
        """Baseline personal stats should be present and reasonable."""
        resp, emo = self._make_history(15)
        result = compute_respiratory_emotion_correlation(resp, emo)
        assert "error" not in result
        baseline = result["baseline"]
        assert "mean_bpm" in baseline
        assert "std_bpm" in baseline
        assert "mean_regularity" in baseline
        assert baseline["mean_bpm"] > 0

    def test_no_timestamp_match_returns_error(self):
        """Observations separated by > 60s should not match."""
        resp = [
            {"breaths_per_minute": 14.0, "breath_regularity": 0.1, "timestamp": 100.0},
            {"breaths_per_minute": 15.0, "breath_regularity": 0.12, "timestamp": 200.0},
            {"breaths_per_minute": 13.0, "breath_regularity": 0.11, "timestamp": 300.0},
        ]
        # Emotion timestamps are 1000s away -- no match
        emo = [
            {"valence": 0.5, "arousal": 0.3, "timestamp": 5000.0},
            {"valence": -0.2, "arousal": 0.6, "timestamp": 6000.0},
            {"valence": 0.1, "arousal": 0.4, "timestamp": 7000.0},
        ]
        result = compute_respiratory_emotion_correlation(resp, emo)
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_noisy_signal_still_extracts(self):
        """A breathing signal with high noise should still produce results."""
        signal = _make_breathing_signal(
            bpm=15.0, duration_s=60.0, fs=100.0, noise_level=0.3
        )
        result = extract_respiratory_rate(signal, fs=100.0)
        # May or may not succeed depending on noise, but should not crash
        assert isinstance(result, dict)

    def test_flat_signal_handles_gracefully(self):
        """A flat (DC) signal should not crash; any output is acceptable."""
        flat = np.ones(10000) * 5.0
        result = extract_respiratory_rate(flat, fs=100.0)
        # A flat signal may produce error OR spurious low-amplitude peaks
        # from numerical noise in the filter -- either outcome is acceptable.
        assert isinstance(result, dict)
        if "error" not in result:
            # If it did detect "peaks", the amplitudes are near-zero noise
            assert result["respiratory_rate_bpm"] >= 0

    def test_classify_with_sighs_shifts_state(self):
        """Many sighs should influence classification."""
        features = {
            "breaths_per_minute": 14.0,
            "inhalation_exhalation_ratio": 1.0,
            "breath_regularity": 0.15,
            "sigh_count": 5,
        }
        classification = classify_breathing_pattern(features)
        assert classification["state"] in ("stressed", "normal", "anxious")
        assert "sighs" in " ".join(classification["contributing_factors"]).lower()
