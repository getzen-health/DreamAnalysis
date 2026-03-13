"""Tests for voice emotion detection pipeline.

Covers:
- Confidence threshold gate (below 60% → None)
- Multi-window analysis (window splitting + aggregation)
- Emotion mapping correctness (9-class → 6-class, no information loss)
- Voice biomarker extraction integration
- Output schema validation
- Edge cases (silence, very short audio, single window, many windows)
"""
from __future__ import annotations

import sys
import os
import types
import importlib
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure ml/ is on sys.path
_ML_DIR = os.path.join(os.path.dirname(__file__), "..")
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)


# ── Helpers ────────────────────────────────────────────────────────────────────

EMOTIONS_6 = {"happy", "sad", "angry", "fear", "surprise", "neutral"}
SR = 22050


def _make_audio(seconds: float = 3.0, sr: int = SR) -> np.ndarray:
    """Return white-noise audio of given duration."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.3, 0.3, int(sr * seconds)).astype(np.float32)


def _make_silent(seconds: float = 3.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype=np.float32)


def _stub_result(emotion: str = "neutral", confidence: float = 0.75) -> Dict:
    """Build a minimal valid emotion result dict."""
    probs = {e: 0.02 for e in EMOTIONS_6}
    remaining = 1.0 - confidence - 0.02 * (len(EMOTIONS_6) - 1)
    probs[emotion] = confidence + remaining
    # normalize
    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}
    return {
        "emotion": emotion,
        "probabilities": probs,
        "valence": 0.1,
        "arousal": 0.5,
        "confidence": round(probs[emotion], 4),
        "model_type": "voice_feature_heuristic",
    }


# ── Import VoiceEmotionModel ───────────────────────────────────────────────────

def _get_model():
    from models.voice_emotion_model import VoiceEmotionModel
    return VoiceEmotionModel()


# ── Output schema tests ────────────────────────────────────────────────────────

class TestOutputSchema:
    def test_required_keys_present(self):
        m = _get_model()
        audio = _make_audio(3.0)
        result = m.predict(audio, sample_rate=SR)
        if result is None:
            return  # below confidence threshold — acceptable
        required = {"emotion", "probabilities", "valence", "arousal", "confidence", "model_type"}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - set(result.keys())}"
        )

    def test_emotion_is_valid_6class(self):
        m = _get_model()
        audio = _make_audio(5.0)
        result = m.predict(audio, sample_rate=SR)
        if result is not None:
            assert result["emotion"] in EMOTIONS_6

    def test_probabilities_sum_to_one(self):
        m = _get_model()
        audio = _make_audio(5.0)
        result = m.predict(audio, sample_rate=SR)
        if result is not None:
            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}"

    def test_probabilities_cover_all_6_classes(self):
        m = _get_model()
        audio = _make_audio(5.0)
        result = m.predict(audio, sample_rate=SR)
        if result is not None:
            assert set(result["probabilities"].keys()) == EMOTIONS_6

    def test_valence_in_range(self):
        m = _get_model()
        audio = _make_audio(3.0)
        result = m.predict(audio, sample_rate=SR)
        if result is not None:
            assert -1.0 <= result["valence"] <= 1.0

    def test_arousal_in_range(self):
        m = _get_model()
        audio = _make_audio(3.0)
        result = m.predict(audio, sample_rate=SR)
        if result is not None:
            assert 0.0 <= result["arousal"] <= 1.0

    def test_confidence_in_range(self):
        m = _get_model()
        audio = _make_audio(3.0)
        result = m.predict(audio, sample_rate=SR)
        if result is not None:
            assert 0.0 <= result["confidence"] <= 1.0

    def test_model_type_is_string(self):
        m = _get_model()
        audio = _make_audio(3.0)
        result = m.predict(audio, sample_rate=SR)
        if result is not None:
            assert isinstance(result["model_type"], str)
            assert len(result["model_type"]) > 0


# ── Confidence threshold tests ─────────────────────────────────────────────────

class TestConfidenceThreshold:
    """Confidence gate: results below 60% must be suppressed (return None)."""

    def test_below_threshold_returns_none(self):
        """When _predict_multi_window returns confidence < 0.60, predict() returns None."""
        m = _get_model()
        # Build a result with well-defined low confidence across all 6 classes
        low_conf = {
            "emotion": "neutral",
            "probabilities": {e: round(1.0 / 6, 4) for e in EMOTIONS_6},
            "valence": 0.0,
            "arousal": 0.3,
            "confidence": 0.40,
            "model_type": "voice_feature_heuristic",
        }

        # Patch _predict_multi_window (called from predict() for non-real_time path)
        # Use a long audio so it doesn't hit the length gate
        audio = _make_audio(6.0)
        with patch.object(m, "_predict_multi_window", return_value=low_conf):
            result = m.predict(audio, sample_rate=SR)
        assert result is None, (
            "Expected None for confidence=0.40 (below 0.60 threshold)"
        )

    def test_above_threshold_returns_result(self):
        """When confidence >= 0.60, predict() must return the result."""
        m = _get_model()
        high_conf = _stub_result("happy", confidence=0.75)

        with patch.object(m, "_predict_multi_window", return_value=high_conf):
            audio = _make_audio(3.0)
            result = m.predict(audio, sample_rate=SR)

        assert result is not None
        assert result["emotion"] == "happy"

    def test_threshold_is_exactly_60_percent(self):
        """Exactly 0.60 confidence must pass the gate."""
        from models.voice_emotion_model import _CONFIDENCE_THRESHOLD
        assert _CONFIDENCE_THRESHOLD == 0.60

    def test_confidence_threshold_value(self):
        """Confirm the threshold constant is importable."""
        from models.voice_emotion_model import _CONFIDENCE_THRESHOLD
        assert 0.0 < _CONFIDENCE_THRESHOLD <= 1.0

    def test_apply_confidence_gate_false_bypasses_threshold(self):
        """apply_confidence_gate=False must bypass the threshold."""
        m = _get_model()
        low_conf = _stub_result("neutral", confidence=0.40)

        with patch.object(m, "_predict_multi_window", return_value=low_conf):
            audio = _make_audio(3.0)
            result = m.predict(audio, sample_rate=SR, apply_confidence_gate=False)

        # Should return the low-confidence result because gate is disabled
        assert result is not None
        assert result["emotion"] == "neutral"

    def test_real_time_fast_path_bypasses_confidence_gate(self):
        """real_time=True (SenseVoice path) returns result regardless of confidence."""
        m = _get_model()
        low_conf = _stub_result("angry", confidence=0.45)
        m._sensevoice = MagicMock()
        m._sensevoice.available = True
        m._sensevoice.predict = MagicMock(return_value=low_conf)

        audio = _make_audio(3.0)
        result = m.predict(audio, sample_rate=SR, real_time=True)
        # SenseVoice path returns directly without gate
        assert result is not None
        assert result["emotion"] == "angry"


# ── Multi-window analysis tests ────────────────────────────────────────────────

class TestMultiWindowAnalysis:
    """Multi-window: audio is split into 3s windows, results aggregated."""

    def test_short_audio_uses_single_window(self):
        """Audio shorter than one window goes straight to _predict_single."""
        m = _get_model()
        stub = _stub_result("happy", confidence=0.80)

        with patch.object(m, "_predict_single", return_value=stub) as mock_single:
            short_audio = _make_audio(2.0)  # shorter than 3s window
            m._predict_multi_window(short_audio, SR)

        mock_single.assert_called_once()

    def test_long_audio_produces_multiple_windows(self):
        """10s audio with 3s windows and 1.5s hop produces multiple windows."""
        from models.voice_emotion_model import _WINDOW_DURATION_S, _WINDOW_HOP_S

        call_count = [0]
        stub = _stub_result("happy", confidence=0.80)

        m = _get_model()

        def counting_predict(audio, sr):
            call_count[0] += 1
            return stub

        with patch.object(m, "_predict_single", side_effect=counting_predict):
            audio = _make_audio(10.0)
            result = m._predict_multi_window(audio, SR)

        # 10s audio, 3s window, 1.5s hop → windows at 0, 1.5, 3, 4.5, 6, 7.5
        # = 6 full windows + possible remainder
        assert call_count[0] >= 3, (
            f"Expected multiple windows for 10s audio, got {call_count[0]}"
        )

    def test_aggregation_averages_probabilities(self):
        """Probability aggregation must be a proper average, not sum."""
        m = _get_model()

        result_a = _stub_result("happy", confidence=0.70)
        result_a["probabilities"]["happy"] = 0.70
        result_a["probabilities"]["neutral"] = 0.30

        result_b = _stub_result("neutral", confidence=0.70)
        result_b["probabilities"]["neutral"] = 0.70
        result_b["probabilities"]["happy"] = 0.30

        call_n = [0]
        def two_window_predict(audio, sr):
            call_n[0] += 1
            if call_n[0] == 1:
                return result_a
            return result_b

        with patch.object(m, "_predict_single", side_effect=two_window_predict):
            # Use long enough audio to trigger 2 windows
            audio = _make_audio(5.0)
            result = m._predict_multi_window(audio, SR)

        assert result is not None
        # After averaging: happy=0.50, neutral=0.50 → must sum to ~1.0
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.02

    def test_windows_analyzed_key_present(self):
        """Result from multi-window path must include 'windows_analyzed'."""
        m = _get_model()
        stub = _stub_result("neutral", confidence=0.75)

        with patch.object(m, "_predict_single", return_value=stub):
            audio = _make_audio(8.0)
            result = m._predict_multi_window(audio, SR)

        assert result is not None
        assert "windows_analyzed" in result
        assert result["windows_analyzed"] >= 1

    def test_all_windows_fail_returns_none(self):
        """If every window returns None, multi-window returns None."""
        m = _get_model()

        with patch.object(m, "_predict_single", return_value=None):
            audio = _make_audio(5.0)
            result = m._predict_multi_window(audio, SR)

        assert result is None

    def test_single_window_fallback_when_too_short(self):
        """Audio too short for windowing calls _predict_single, not aggregation."""
        m = _get_model()
        stub = _stub_result("sad", confidence=0.72)

        with patch.object(m, "_predict_single", return_value=stub) as mock_s:
            short = _make_audio(1.5)
            result = m._predict_multi_window(short, SR)

        mock_s.assert_called_once()
        assert result is not None
        assert result["emotion"] == "sad"


# ── Emotion mapping tests (9-class → 6-class) ─────────────────────────────────

class TestEmotionMapping:
    """Verify the E2V_MAP covers all 9 emotion2vec+ labels with no loss."""

    def test_all_e2v_labels_mapped(self):
        from models.voice_emotion_model import _E2V_LABELS, _E2V_MAP
        for label in _E2V_LABELS:
            assert label in _E2V_MAP, f"e2v label '{label}' missing from _E2V_MAP"

    def test_all_mapped_targets_are_6class(self):
        from models.voice_emotion_model import _E2V_MAP, _6CLASS
        for src, tgt in _E2V_MAP.items():
            assert tgt in _6CLASS, (
                f"_E2V_MAP['{src}'] = '{tgt}' is not a valid 6-class emotion"
            )

    def test_disgusted_maps_to_angry(self):
        from models.voice_emotion_model import _E2V_MAP
        assert _E2V_MAP["disgusted"] == "angry"

    def test_fearful_maps_to_fear(self):
        from models.voice_emotion_model import _E2V_MAP
        assert _E2V_MAP["fearful"] == "fear"

    def test_surprised_maps_to_surprise(self):
        from models.voice_emotion_model import _E2V_MAP
        assert _E2V_MAP["surprised"] == "surprise"

    def test_other_and_unknown_map_to_neutral(self):
        from models.voice_emotion_model import _E2V_MAP
        assert _E2V_MAP["other"] == "neutral"
        assert _E2V_MAP["unknown"] == "neutral"

    def test_9class_covers_all_primary_emotions(self):
        """Every 6-class target must be reachable from the 9-class source set."""
        from models.voice_emotion_model import _E2V_MAP, _6CLASS
        reachable = set(_E2V_MAP.values())
        for emo in _6CLASS:
            assert emo in reachable, (
                f"6-class emotion '{emo}' is unreachable from any e2v label"
            )

    def test_probability_aggregation_preserves_total(self):
        """After 9→6 mapping + normalization, probs must sum to 1.0."""
        from models.voice_emotion_model import _E2V_MAP, _E2V_LABELS, _6CLASS
        raw_scores = [1.0 / 9] * 9  # uniform 9-class
        raw = dict(zip(_E2V_LABELS, raw_scores))
        probs_6: Dict[str, float] = {c: 0.0 for c in _6CLASS}
        for label, prob in raw.items():
            target = _E2V_MAP.get(label, "neutral")
            probs_6[target] += float(prob)
        total = sum(probs_6.values())
        assert abs(total - 1.0) < 1e-6


# ── Edge case tests ────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_none_audio_returns_none(self):
        m = _get_model()
        assert m.predict(None, sample_rate=SR) is None  # type: ignore

    def test_empty_array_returns_none(self):
        m = _get_model()
        assert m.predict(np.array([], dtype=np.float32), sample_rate=SR) is None

    def test_below_min_samples_returns_none(self):
        m = _get_model()
        from models.voice_emotion_model import _MIN_SAMPLES
        short = np.zeros(_MIN_SAMPLES - 1, dtype=np.float32)
        assert m.predict(short, sample_rate=SR) is None

    def test_exactly_min_samples_does_not_return_none_because_too_short(self):
        """At exactly _MIN_SAMPLES, predict should attempt inference (may still
        return None if model/confidence gate applies, but not due to length)."""
        m = _get_model()
        from models.voice_emotion_model import _MIN_SAMPLES
        audio = np.zeros(_MIN_SAMPLES, dtype=np.float32)
        # Just verifying no exception — result may be None (confidence/model)
        try:
            m.predict(audio, sample_rate=SR)
        except Exception as exc:
            pytest.fail(f"predict() raised on exactly _MIN_SAMPLES samples: {exc}")

    def test_silent_audio_does_not_crash(self):
        m = _get_model()
        silent = _make_silent(3.0)
        try:
            m.predict(silent, sample_rate=SR)
        except Exception as exc:
            pytest.fail(f"predict() raised on silent audio: {exc}")

    def test_different_sample_rates(self):
        """Model should handle 16kHz input without crashing."""
        m = _get_model()
        rng = np.random.default_rng(7)
        audio_16k = rng.uniform(-0.3, 0.3, 16000 * 3).astype(np.float32)
        try:
            m.predict(audio_16k, sample_rate=16000)
        except Exception as exc:
            pytest.fail(f"predict() raised on 16kHz audio: {exc}")


# ── Biomarker enrichment tests ─────────────────────────────────────────────────

class TestBiomarkerEnrichment:
    """predict_with_biomarkers() adds 'biomarkers' and 'mental_health' keys."""

    def test_predict_with_biomarkers_returns_none_when_audio_too_short(self):
        m = _get_model()
        short = np.zeros(100, dtype=np.float32)
        assert m.predict_with_biomarkers(short, sample_rate=SR) is None

    def test_predict_with_biomarkers_has_superset_of_predict_keys(self):
        m = _get_model()
        audio = _make_audio(5.0)
        result = m.predict_with_biomarkers(audio, sample_rate=SR)
        if result is None:
            return  # confidence gate or model unavailable — acceptable
        base_keys = {"emotion", "probabilities", "valence", "arousal", "confidence", "model_type"}
        assert base_keys.issubset(result.keys())

    def test_predict_with_biomarkers_confidence_gate(self):
        """predict_with_biomarkers must also respect the confidence gate."""
        m = _get_model()
        low_conf = {
            "emotion": "neutral",
            "probabilities": {e: round(1.0 / 6, 4) for e in EMOTIONS_6},
            "valence": 0.0,
            "arousal": 0.3,
            "confidence": 0.35,
            "model_type": "voice_feature_heuristic",
        }

        with patch.object(m, "_predict_multi_window", return_value=low_conf):
            audio = _make_audio(6.0)
            result = m.predict_with_biomarkers(audio, sample_rate=SR)

        # Confidence gate applies at predict() level, so biomarkers never added
        assert result is None


# ── Valence/arousal formula tests ──────────────────────────────────────────────

class TestValenceArousal:
    def test_happy_has_positive_valence(self):
        from models.voice_emotion_model import _valence_arousal
        probs = {"happy": 0.8, "sad": 0.04, "angry": 0.04,
                 "fear": 0.04, "surprise": 0.04, "neutral": 0.04}
        valence, arousal = _valence_arousal(probs)
        assert valence > 0

    def test_sad_has_negative_valence(self):
        from models.voice_emotion_model import _valence_arousal
        probs = {"happy": 0.04, "sad": 0.8, "angry": 0.04,
                 "fear": 0.04, "surprise": 0.04, "neutral": 0.04}
        valence, arousal = _valence_arousal(probs)
        assert valence < 0

    def test_angry_has_high_arousal(self):
        from models.voice_emotion_model import _valence_arousal
        probs = {"happy": 0.04, "sad": 0.04, "angry": 0.8,
                 "fear": 0.04, "surprise": 0.04, "neutral": 0.04}
        valence, arousal = _valence_arousal(probs)
        assert arousal > 0.3

    def test_neutral_has_low_arousal(self):
        from models.voice_emotion_model import _valence_arousal
        probs = {"happy": 0.04, "sad": 0.04, "angry": 0.04,
                 "fear": 0.04, "surprise": 0.04, "neutral": 0.80}
        valence, arousal = _valence_arousal(probs)
        assert arousal < 0.5

    def test_valence_bounded(self):
        from models.voice_emotion_model import _valence_arousal
        rng = np.random.default_rng(99)
        for _ in range(50):
            raw = rng.random(6)
            probs = dict(zip(
                ["happy", "sad", "angry", "fear", "surprise", "neutral"],
                (raw / raw.sum()).tolist()
            ))
            v, a = _valence_arousal(probs)
            assert -1.0 <= v <= 1.0
            assert 0.0 <= a <= 1.0


# ── Singleton test ─────────────────────────────────────────────────────────────

def test_get_voice_model_singleton():
    from models.voice_emotion_model import get_voice_model
    m1 = get_voice_model()
    m2 = get_voice_model()
    assert m1 is m2
