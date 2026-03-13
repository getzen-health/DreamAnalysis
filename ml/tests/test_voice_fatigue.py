"""Tests for VoiceFatigueScanner — issue #377.

Uses synthetic audio (sine waves with known properties) to verify:
  - Fatigue index stays within [0, 100]
  - Baseline comparison logic works correctly
  - Edge cases (silence, very short audio) are handled gracefully
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest

# Ensure ml/ is importable from the tests directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.voice_fatigue_model import (
    FatigueResult,
    VoiceFatigueScanner,
    _extract_hnr,
    _extract_jitter,
    _extract_shimmer,
    _extract_speaking_rate_proxy,
    get_voice_fatigue_scanner,
)

_SR = 22050


# ── Synthetic audio helpers ───────────────────────────────────────────────────

def _sine_wave(
    freq: float = 200.0,
    duration: float = 3.0,
    sr: int = _SR,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Pure sine wave — maximally periodic, low jitter/shimmer, high HNR."""
    t = np.arange(int(sr * duration)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noisy_sine(
    freq: float = 200.0,
    duration: float = 3.0,
    sr: int = _SR,
    noise_level: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Sine wave with additive Gaussian noise — lower HNR, higher jitter."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration)) / sr
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    noise = rng.normal(0, noise_level, len(t))
    return (signal + noise).astype(np.float32)


def _jittery_sine(
    freq: float = 200.0,
    duration: float = 3.0,
    sr: int = _SR,
    jitter_amount: float = 0.05,
    seed: int = 7,
) -> np.ndarray:
    """Sine wave with random period-to-period pitch variation (high jitter)."""
    rng = np.random.default_rng(seed)
    n_samples = int(sr * duration)
    t = np.zeros(n_samples, dtype=np.float64)
    phase = 0.0
    for i in range(n_samples):
        freq_perturbed = freq * (1.0 + rng.normal(0, jitter_amount))
        phase += 2.0 * np.pi * freq_perturbed / sr
        t[i] = 0.5 * np.sin(phase)
    return t.astype(np.float32)


def _silent_audio(duration: float = 3.0, sr: int = _SR) -> np.ndarray:
    """True silence."""
    return np.zeros(int(sr * duration), dtype=np.float32)


def _very_short_audio(sr: int = _SR) -> np.ndarray:
    """Audio shorter than the minimum threshold (< 1 second)."""
    return _sine_wave(duration=0.3, sr=sr)


# ── FatigueResult output contract tests ──────────────────────────────────────

class TestFatigueResultFields:
    """Verify the FatigueResult dataclass has all required fields."""

    def test_required_fields_present(self):
        scanner = VoiceFatigueScanner()
        result = scanner.scan(_sine_wave(), sr=_SR)
        assert hasattr(result, "fatigue_index")
        assert hasattr(result, "hnr_db")
        assert hasattr(result, "hnr_delta")
        assert hasattr(result, "jitter_pct")
        assert hasattr(result, "jitter_ratio")
        assert hasattr(result, "shimmer_db")
        assert hasattr(result, "shimmer_ratio")
        assert hasattr(result, "speaking_rate_proxy")
        assert hasattr(result, "speaking_rate_ratio")
        assert hasattr(result, "confidence")
        assert hasattr(result, "recommendations")
        assert hasattr(result, "baseline_used")

    def test_recommendations_is_list(self):
        scanner = VoiceFatigueScanner()
        result = scanner.scan(_sine_wave(), sr=_SR)
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) >= 1


# ── Fatigue index boundary tests ──────────────────────────────────────────────

class TestFatigueIndexBoundaries:
    """fatigue_index must always be in [0, 100]."""

    def test_clean_sine_within_bounds(self):
        scanner = VoiceFatigueScanner()
        result = scanner.scan(_sine_wave(), sr=_SR)
        assert 0.0 <= result.fatigue_index <= 100.0

    def test_noisy_sine_within_bounds(self):
        scanner = VoiceFatigueScanner()
        result = scanner.scan(_noisy_sine(), sr=_SR)
        assert 0.0 <= result.fatigue_index <= 100.0

    def test_jittery_sine_within_bounds(self):
        scanner = VoiceFatigueScanner()
        result = scanner.scan(_jittery_sine(), sr=_SR)
        assert 0.0 <= result.fatigue_index <= 100.0

    def test_silence_within_bounds(self):
        """Silent audio has zero biomarkers — fatigue should still be in range."""
        scanner = VoiceFatigueScanner()
        result = scanner.scan(_silent_audio(), sr=_SR)
        assert 0.0 <= result.fatigue_index <= 100.0

    def test_short_audio_returns_zero_confidence(self):
        """Audio shorter than minimum threshold → low/zero confidence, valid result."""
        scanner = VoiceFatigueScanner()
        result = scanner.scan(_very_short_audio(), sr=_SR)
        assert 0.0 <= result.fatigue_index <= 100.0
        assert result.confidence == 0.0

    def test_random_noise_within_bounds(self):
        """Random Gaussian noise must also produce in-range index."""
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 0.5, int(_SR * 3)).astype(np.float32)
        scanner = VoiceFatigueScanner()
        result = scanner.scan(noise, sr=_SR)
        assert 0.0 <= result.fatigue_index <= 100.0


# ── Baseline comparison tests ─────────────────────────────────────────────────

class TestBaselineComparison:
    """Verify that baseline shifts the fatigue score in the expected direction."""

    def test_baseline_flag_set_when_provided(self):
        scanner = VoiceFatigueScanner()
        baseline = {
            "hnr_db": 20.0,
            "jitter_pct": 0.5,
            "shimmer_db": 0.3,
            "speaking_rate_proxy": 0.48,
        }
        result = scanner.scan(_sine_wave(), sr=_SR, baseline=baseline)
        assert result.baseline_used is True

    def test_no_baseline_flag_when_none(self):
        scanner = VoiceFatigueScanner()
        result = scanner.scan(_sine_wave(), sr=_SR, baseline=None)
        assert result.baseline_used is False

    def test_very_generous_baseline_raises_fatigue(self):
        """If baseline is much 'better' than current audio, fatigue index rises."""
        scanner = VoiceFatigueScanner()
        # Set an artificially perfect baseline
        generous_baseline = {
            "hnr_db": 40.0,        # much higher than any real voice
            "jitter_pct": 0.05,    # near-zero perturbation
            "shimmer_db": 0.05,
            "speaking_rate_proxy": 0.80,
        }
        result_with_baseline = scanner.scan(_noisy_sine(), sr=_SR, baseline=generous_baseline)
        result_no_baseline = scanner.scan(_noisy_sine(), sr=_SR, baseline=None)
        # With the generous baseline the fatigue should be higher
        assert result_with_baseline.fatigue_index >= result_no_baseline.fatigue_index

    def test_degraded_baseline_lowers_fatigue(self):
        """If baseline is already 'bad', current audio looks less fatigued relative to it."""
        scanner = VoiceFatigueScanner()
        # Set a very poor baseline (as if the user always sounds fatigued)
        poor_baseline = {
            "hnr_db": 2.0,
            "jitter_pct": 4.0,
            "shimmer_db": 2.0,
            "speaking_rate_proxy": 0.10,
        }
        result_with_baseline = scanner.scan(_noisy_sine(), sr=_SR, baseline=poor_baseline)
        result_no_baseline = scanner.scan(_noisy_sine(), sr=_SR, baseline=None)
        # With the poor baseline the fatigue should be lower (or equal)
        assert result_with_baseline.fatigue_index <= result_no_baseline.fatigue_index

    def test_partial_baseline_accepted(self):
        """A baseline with only some keys should not raise an exception."""
        scanner = VoiceFatigueScanner()
        partial_baseline = {"hnr_db": 18.0}  # missing jitter/shimmer/rate
        result = scanner.scan(_sine_wave(), sr=_SR, baseline=partial_baseline)
        assert 0.0 <= result.fatigue_index <= 100.0
        assert result.baseline_used is True


# ── Clean vs fatigued voice ordering tests ───────────────────────────────────

class TestFatigueOrdering:
    """Noisy / perturbed audio should score at least as high as a pure sine."""

    def test_noisy_not_less_fatigued_than_clean(self):
        """A noise-contaminated signal should generally score >= a clean sine."""
        scanner = VoiceFatigueScanner()
        clean_result = scanner.scan(_sine_wave(noise_level=0.0) if hasattr(_sine_wave, "noise_level") else _sine_wave(), sr=_SR)
        noisy_result = scanner.scan(_noisy_sine(noise_level=0.5), sr=_SR)
        # Allow a small tolerance — the noise should either increase fatigue or keep it the same
        assert noisy_result.fatigue_index >= clean_result.fatigue_index - 5.0


# ── Biomarker extraction unit tests ──────────────────────────────────────────

class TestBiomarkerExtraction:
    """Low-level extraction functions should return non-negative finite values."""

    def test_hnr_non_negative_for_sine(self):
        y = _sine_wave()
        hnr = _extract_hnr(y, _SR)
        assert hnr >= 0.0
        assert np.isfinite(hnr)

    def test_jitter_non_negative_for_sine(self):
        y = _sine_wave()
        jitter = _extract_jitter(y, _SR)
        assert jitter >= 0.0
        assert np.isfinite(jitter)

    def test_shimmer_non_negative_for_sine(self):
        y = _sine_wave()
        shimmer = _extract_shimmer(y, _SR)
        assert shimmer >= 0.0
        assert np.isfinite(shimmer)

    def test_speaking_rate_proxy_between_0_and_1(self):
        y = _sine_wave()
        rate = _extract_speaking_rate_proxy(y, _SR)
        assert 0.0 <= rate <= 1.0

    def test_silence_hnr_is_zero(self):
        """Silent audio has no voiced frames → HNR should be 0."""
        hnr = _extract_hnr(_silent_audio(), _SR)
        assert hnr == 0.0

    def test_silence_jitter_is_zero(self):
        """Silent audio has no voiced frames → jitter should be 0."""
        jitter = _extract_jitter(_silent_audio(), _SR)
        assert jitter == 0.0


# ── Singleton tests ───────────────────────────────────────────────────────────

class TestSingleton:
    def test_singleton_returns_same_instance(self):
        s1 = get_voice_fatigue_scanner()
        s2 = get_voice_fatigue_scanner()
        assert s1 is s2

    def test_singleton_is_voice_fatigue_scanner(self):
        s = get_voice_fatigue_scanner()
        assert isinstance(s, VoiceFatigueScanner)
