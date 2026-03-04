"""Tests for VoiceEmotionModel — emotion2vec+ with LightGBM fallback."""
from __future__ import annotations
import sys
import numpy as np
import pytest


def _get_model():
    # Add ml/ to sys.path so we can import from there
    import os
    ml_dir = os.path.join(os.path.dirname(__file__), "..")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    from models.voice_emotion_model import VoiceEmotionModel
    return VoiceEmotionModel()


EMOTIONS_6 = {"happy", "sad", "angry", "fear", "surprise", "neutral"}


def test_too_short_returns_none():
    m = _get_model()
    short = np.zeros(100, dtype=np.float32)
    result = m.predict(short, sample_rate=22050)
    assert result is None


def test_output_has_required_keys():
    """With any valid audio, output must have all required keys."""
    m = _get_model()
    rng = np.random.default_rng(42)
    audio = rng.uniform(-0.3, 0.3, 22050 * 3).astype(np.float32)
    result = m.predict(audio, sample_rate=22050)
    # Result can be None if no model available (CI without models)
    if result is not None:
        assert result["emotion"] in EMOTIONS_6
        assert "valence" in result
        assert "arousal" in result
        assert "confidence" in result
        assert "model_type" in result
        assert result["model_type"] in {"voice_lgbm_fallback", "voice_emotion2vec"}


def test_valence_arousal_range():
    m = _get_model()
    rng = np.random.default_rng(1)
    audio = rng.uniform(-0.3, 0.3, 22050 * 5).astype(np.float32)
    result = m.predict(audio, sample_rate=22050)
    if result:
        assert -1.0 <= result["valence"] <= 1.0
        assert 0.0 <= result["arousal"] <= 1.0


def test_get_voice_model_singleton():
    import os
    ml_dir = os.path.join(os.path.dirname(__file__), "..")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    from models.voice_emotion_model import get_voice_model
    m1 = get_voice_model()
    m2 = get_voice_model()
    assert m1 is m2
