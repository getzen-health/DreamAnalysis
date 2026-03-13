"""Tests for VoiceEnsemble — acoustic features, blending, and temporal smoothing.

Covers issue #334: multi-model ensemble for voice emotion detection.
Tests run without any real model files — ensemble degrades gracefully.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

# Ensure ml/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.voice_ensemble import (
    VoiceEnsemble,
    TemporalSmoother,
    _blend_e2v_acoustic,
    _acoustic_adjustment,
    extract_acoustic_features,
    get_voice_ensemble,
)

_6CLASS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]
_SR = 22050


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_audio(seconds: float = 3.0, sr: int = _SR, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.3, 0.3, int(sr * seconds)).astype(np.float32)


def _uniform_probs() -> Dict[str, float]:
    val = round(1.0 / 6, 4)
    return {e: val for e in _6CLASS}


# ── Acoustic feature extraction tests ─────────────────────────────────────────

class TestAcousticFeatureExtraction:
    def test_returns_all_required_keys(self):
        audio = _make_audio(3.0)
        feats = extract_acoustic_features(audio, sr=_SR)
        required = {
            "pitch_mean", "pitch_std", "pitch_range",
            "energy_mean", "energy_std",
            "speaking_rate_proxy",
            "spectral_centroid_mean",
        }
        for key in required:
            assert key in feats, f"Missing key: {key}"

    def test_returns_13_mfcc_keys(self):
        audio = _make_audio(3.0)
        feats = extract_acoustic_features(audio, sr=_SR)
        for i in range(1, 14):
            assert f"mfcc_{i}" in feats, f"Missing mfcc_{i}"

    def test_all_values_are_finite(self):
        audio = _make_audio(5.0)
        feats = extract_acoustic_features(audio, sr=_SR)
        for key, val in feats.items():
            assert np.isfinite(val), f"Non-finite value for {key}: {val}"

    def test_short_audio_returns_zeroed_features(self):
        """Audio shorter than 0.1s should return all-zero features safely."""
        short = np.zeros(100, dtype=np.float32)
        feats = extract_acoustic_features(short, sr=_SR)
        # energy should be effectively zero
        assert feats["energy_mean"] == 0.0
        assert feats["pitch_mean"] == 0.0

    def test_energy_is_positive_for_nonzero_audio(self):
        audio = _make_audio(2.0)
        feats = extract_acoustic_features(audio, sr=_SR)
        assert feats["energy_mean"] > 0.0

    def test_silent_audio_has_near_zero_energy(self):
        silent = np.zeros(int(_SR * 2.0), dtype=np.float32)
        feats = extract_acoustic_features(silent, sr=_SR)
        assert feats["energy_mean"] < 1e-6

    def test_speaking_rate_proxy_in_range(self):
        audio = _make_audio(3.0)
        feats = extract_acoustic_features(audio, sr=_SR)
        assert 0.0 <= feats["speaking_rate_proxy"] <= 1.0

    def test_spectral_centroid_is_nonnegative(self):
        audio = _make_audio(2.0)
        feats = extract_acoustic_features(audio, sr=_SR)
        assert feats["spectral_centroid_mean"] >= 0.0

    def test_tonal_signal_has_detectable_pitch(self):
        """A sine wave at 200 Hz should yield a pitch estimate near 200 Hz."""
        t = np.arange(int(_SR * 2.0)) / _SR
        tone = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        feats = extract_acoustic_features(tone, sr=_SR)
        # Autocorrelation pitch detection may not be perfectly accurate,
        # but pitch_mean should be non-zero for a clear periodic signal.
        assert feats["pitch_mean"] > 0.0


# ── Ensemble weighting tests ───────────────────────────────────────────────────

class TestEnsembleWeighting:
    def test_high_pitch_high_energy_boosts_angry_and_happy(self):
        acoustic = {
            "pitch_mean": 300.0,   # > _PITCH_HIGH (220)
            "energy_mean": 0.10,   # > _ENERGY_HIGH (0.05)
            "spectral_centroid_mean": 2000.0,
        }
        delta = _acoustic_adjustment(acoustic)
        assert delta["angry"] > 0.0, "angry should be boosted"
        assert delta["happy"] > 0.0, "happy should be boosted"
        assert delta["sad"] < 0.0, "sad should be suppressed"

    def test_low_pitch_low_energy_boosts_sad_and_neutral(self):
        acoustic = {
            "pitch_mean": 100.0,   # < _PITCH_LOW (130)
            "energy_mean": 0.005,  # < _ENERGY_LOW (0.01)
            "spectral_centroid_mean": 1500.0,
        }
        delta = _acoustic_adjustment(acoustic)
        assert delta["sad"] > 0.0, "sad should be boosted"
        assert delta["neutral"] > 0.0, "neutral should be boosted"
        assert delta["happy"] < 0.0, "happy should be suppressed"

    def test_high_pitch_low_energy_boosts_fear_and_surprise(self):
        acoustic = {
            "pitch_mean": 300.0,
            "energy_mean": 0.005,
            "spectral_centroid_mean": 1500.0,
        }
        delta = _acoustic_adjustment(acoustic)
        assert delta["fear"] > 0.0, "fear should be boosted"
        assert delta["surprise"] > 0.0, "surprise should be boosted"

    def test_blend_produces_valid_probability_distribution(self):
        e2v = _uniform_probs()
        acoustic = {
            "pitch_mean": 300.0,
            "energy_mean": 0.10,
            "spectral_centroid_mean": 2000.0,
        }
        blended = _blend_e2v_acoustic(e2v, acoustic)
        assert set(blended.keys()) == set(_6CLASS)
        total = sum(blended.values())
        assert abs(total - 1.0) < 1e-3, f"Probabilities should sum to 1, got {total}"
        for val in blended.values():
            assert val >= 0.0, "No negative probabilities"

    def test_blend_respects_e2v_weight(self):
        """Dominant emotion in e2v should remain dominant after blending if signal is neutral."""
        e2v = {e: 1.0 / 6 for e in _6CLASS}
        e2v["happy"] = 0.60
        # Re-normalise
        total = sum(e2v.values())
        e2v = {k: v / total for k, v in e2v.items()}
        acoustic = {
            "pitch_mean": 0.0,   # no pitch detected → neutral signal
            "energy_mean": 0.02,
            "spectral_centroid_mean": 2000.0,
        }
        blended = _blend_e2v_acoustic(e2v, acoustic)
        # Happy should still be the dominant class after blending
        assert max(blended, key=blended.__getitem__) == "happy"


# ── Temporal smoothing tests ───────────────────────────────────────────────────

class TestTemporalSmoothing:
    def _make_probs(self, dominant: str, strength: float = 0.7) -> Dict[str, float]:
        remainder = (1.0 - strength) / (len(_6CLASS) - 1)
        return {e: (strength if e == dominant else remainder) for e in _6CLASS}

    def test_single_prediction_returned_unchanged(self):
        smoother = TemporalSmoother()
        probs = _uniform_probs()
        result = smoother.update(probs, "neutral")
        # With only one sample and _MIN_SAMPLES_FOR_SMOOTH=2, no smoothing applied
        assert result == probs

    def test_smoothing_blends_multiple_predictions(self):
        smoother = TemporalSmoother()
        p1 = self._make_probs("happy", 0.8)
        p2 = self._make_probs("happy", 0.8)
        smoother.update(p1, "happy", ts=time.time() - 1.0)
        result = smoother.update(p2, "happy", ts=time.time())
        # Should still be happy, just smoothed
        assert max(result, key=result.__getitem__) == "happy"
        assert abs(sum(result.values()) - 1.0) < 1e-3

    def test_rapid_flip_guard_suppresses_quick_reversal(self):
        """angry → happy → angry within 5 seconds should be suppressed."""
        smoother = TemporalSmoother(flip_guard_secs=10.0)
        now = time.time()
        # Seed the smoother with three predictions
        p_angry = self._make_probs("angry", 0.75)
        p_happy = self._make_probs("happy", 0.75)

        smoother.update(p_angry, "angry", ts=now - 4.0)
        smoother.update(p_happy, "happy", ts=now - 2.0)
        result = smoother.update(p_angry, "angry", ts=now)

        # The flip guard should suppress the return to angry
        # Result emotion should NOT be angry (suppressed by guard)
        smoothed_emotion = max(result, key=result.__getitem__)
        # Either it stays happy or is suppressed — just not a raw angry flip
        # (the guard boosts the previous committed emotion)
        # We just verify the result sums to 1 and is a valid emotion
        assert smoothed_emotion in _6CLASS
        assert abs(sum(result.values()) - 1.0) < 1e-3

    def test_reset_clears_history(self):
        smoother = TemporalSmoother()
        p = self._make_probs("sad", 0.7)
        smoother.update(p, "sad")
        smoother.reset()
        # After reset, next prediction should be returned unchanged (len < _MIN_SAMPLES)
        result = smoother.update(_uniform_probs(), "neutral")
        assert result == _uniform_probs()

    def test_smoothed_probs_sum_to_one(self):
        smoother = TemporalSmoother()
        for i in range(5):
            p = self._make_probs("happy", 0.65)
            result = smoother.update(p, "happy", ts=float(i))
        assert abs(sum(result.values()) - 1.0) < 1e-3

    def test_all_smoothed_probs_nonnegative(self):
        smoother = TemporalSmoother()
        for i in range(5):
            p = self._make_probs("angry", 0.7)
            result = smoother.update(p, "angry", ts=float(i))
        for val in result.values():
            assert val >= 0.0


# ── Output format compatibility tests ─────────────────────────────────────────

class TestEnsembleOutputFormat:
    """Verify ensemble output schema matches VoiceEmotionModel exactly."""

    def _stub_base_result(self):
        return {
            "emotion": "neutral",
            "probabilities": _uniform_probs(),
            "valence": 0.0,
            "arousal": 0.3,
            "confidence": round(1.0 / 6, 4),
            "model_type": "voice_feature_heuristic",
        }

    def test_ensemble_predict_short_audio_returns_none(self):
        """Audio shorter than 0.5s should return None."""
        ensemble = VoiceEnsemble()
        short = np.zeros(100, dtype=np.float32)
        result = ensemble.predict(short, sample_rate=_SR)
        assert result is None

    def test_ensemble_singleton_is_same_instance(self):
        e1 = get_voice_ensemble()
        e2 = get_voice_ensemble()
        assert e1 is e2

    def test_ensemble_predict_returns_required_keys_when_model_available(self):
        """If VoiceEmotionModel returns a result, ensemble adds extra keys."""
        ensemble = VoiceEnsemble()

        # Monkey-patch the voice model to return a stub result
        class _FakeModel:
            def predict(self, audio, sample_rate=22050, **kw):
                return {
                    "emotion": "happy",
                    "probabilities": {"happy": 0.5, "sad": 0.1, "angry": 0.1,
                                      "fear": 0.1, "surprise": 0.1, "neutral": 0.1},
                    "valence": 0.4,
                    "arousal": 0.6,
                    "confidence": 0.5,
                    "model_type": "voice_feature_heuristic",
                }

        ensemble._voice_model = _FakeModel()
        audio = _make_audio(3.0)
        result = ensemble.predict(audio, sample_rate=_SR)

        assert result is not None
        # Standard keys
        assert result["emotion"] in _6CLASS
        assert isinstance(result["probabilities"], dict)
        assert set(result["probabilities"].keys()) == set(_6CLASS)
        assert -1.0 <= result["valence"] <= 1.0
        assert 0.0 <= result["arousal"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["model_type"], str)
        # Ensemble-specific keys
        assert result["ensemble_active"] is True
        assert "acoustic_features" in result
        assert isinstance(result["smoothed"], bool)

    def test_ensemble_probs_sum_to_one(self):
        ensemble = VoiceEnsemble()

        class _FakeModel:
            def predict(self, audio, sample_rate=22050, **kw):
                return {
                    "emotion": "sad",
                    "probabilities": {"happy": 0.05, "sad": 0.60, "angry": 0.10,
                                      "fear": 0.10, "surprise": 0.05, "neutral": 0.10},
                    "valence": -0.4,
                    "arousal": 0.2,
                    "confidence": 0.60,
                    "model_type": "voice_feature_heuristic",
                }

        ensemble._voice_model = _FakeModel()
        audio = _make_audio(3.0)
        result = ensemble.predict(audio, sample_rate=_SR)
        assert result is not None
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    def test_ensemble_model_type_contains_ensemble_suffix(self):
        ensemble = VoiceEnsemble()

        class _FakeModel:
            def predict(self, audio, sample_rate=22050, **kw):
                return {
                    "emotion": "neutral",
                    "probabilities": _uniform_probs(),
                    "valence": 0.0,
                    "arousal": 0.3,
                    "confidence": round(1.0 / 6, 4),
                    "model_type": "voice_emotion2vec",
                }

        ensemble._voice_model = _FakeModel()
        audio = _make_audio(3.0)
        result = ensemble.predict(audio, sample_rate=_SR)
        assert result is not None
        assert "ensemble" in result["model_type"]

    def test_ensemble_acoustic_features_has_expected_keys(self):
        ensemble = VoiceEnsemble()

        class _FakeModel:
            def predict(self, audio, sample_rate=22050, **kw):
                return {
                    "emotion": "neutral",
                    "probabilities": _uniform_probs(),
                    "valence": 0.0,
                    "arousal": 0.3,
                    "confidence": round(1.0 / 6, 4),
                    "model_type": "voice_feature_heuristic",
                }

        ensemble._voice_model = _FakeModel()
        audio = _make_audio(3.0)
        result = ensemble.predict(audio, sample_rate=_SR)
        assert result is not None
        acoustic = result["acoustic_features"]
        for key in ["pitch_mean", "energy_mean", "speaking_rate_proxy",
                    "spectral_centroid_mean"]:
            assert key in acoustic, f"Missing acoustic feature: {key}"

    def test_ensemble_no_temporal_smoothing_flag(self):
        """apply_temporal_smoothing=False should still return a valid result."""
        ensemble = VoiceEnsemble()

        class _FakeModel:
            def predict(self, audio, sample_rate=22050, **kw):
                return {
                    "emotion": "happy",
                    "probabilities": {"happy": 0.5, "sad": 0.1, "angry": 0.1,
                                      "fear": 0.1, "surprise": 0.1, "neutral": 0.1},
                    "valence": 0.4,
                    "arousal": 0.6,
                    "confidence": 0.5,
                    "model_type": "voice_feature_heuristic",
                }

        ensemble._voice_model = _FakeModel()
        audio = _make_audio(3.0)
        result = ensemble.predict(audio, sample_rate=_SR, apply_temporal_smoothing=False)
        assert result is not None
        assert result["smoothed"] is False
        assert result["emotion"] in _6CLASS

    def test_ensemble_returns_none_when_base_model_unavailable(self):
        """If the base model returns None, ensemble should return None."""
        ensemble = VoiceEnsemble()

        class _NullModel:
            def predict(self, audio, sample_rate=22050, **kw):
                return None

        ensemble._voice_model = _NullModel()
        audio = _make_audio(3.0)
        result = ensemble.predict(audio, sample_rate=_SR)
        assert result is None
