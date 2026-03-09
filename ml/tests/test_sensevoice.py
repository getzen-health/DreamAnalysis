"""Tests for SenseVoice integration (graceful degradation if not installed)."""
import numpy as np
import pytest
from models.voice_emotion_model import SenseVoiceEmotionDetector


def _synthetic_audio(seconds: float = 3.0, fs: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.1, int(seconds * fs)).astype(np.float32)


def test_sensevoice_initializes():
    """SenseVoice should initialize without crashing even if model unavailable."""
    detector = SenseVoiceEmotionDetector()
    assert hasattr(detector, "available")
    assert isinstance(detector.available, bool)


def test_unavailable_returns_none():
    """If SenseVoice unavailable, predict should return None not raise."""
    detector = SenseVoiceEmotionDetector()
    if not detector.available:
        result = detector.predict(_synthetic_audio())
        assert result is None


def test_parse_emotion_happy():
    detector = SenseVoiceEmotionDetector()
    result = detector._parse_emotion("<|HAPPY|>I'm feeling great today<|/HAPPY|>")
    assert result == "happy"


def test_parse_emotion_angry():
    detector = SenseVoiceEmotionDetector()
    result = detector._parse_emotion("<|ANGRY|>This is frustrating<|/ANGRY|>")
    assert result == "angry"


def test_parse_emotion_fallback():
    detector = SenseVoiceEmotionDetector()
    result = detector._parse_emotion("plain text with no tags")
    assert result == "neutral"


def test_clean_text():
    detector = SenseVoiceEmotionDetector()
    cleaned = detector._clean_text("<|HAPPY|>Hello world<|/HAPPY|>")
    assert "HAPPY" not in cleaned
    assert "Hello world" in cleaned


def test_voice_model_has_sensevoice():
    """VoiceEmotionModel should have _sensevoice attribute."""
    from models.voice_emotion_model import VoiceEmotionModel
    model = VoiceEmotionModel()
    assert hasattr(model, "_sensevoice")
    assert isinstance(model._sensevoice, SenseVoiceEmotionDetector)


def test_voice_model_real_time_false_skips_sensevoice():
    """real_time=False should not invoke SenseVoice even if available."""
    from models.voice_emotion_model import VoiceEmotionModel
    model = VoiceEmotionModel()
    # Force _sensevoice to appear available but track calls
    calls = []

    original_predict = model._sensevoice.predict

    def mock_predict(audio, fs=16000):
        calls.append(True)
        return original_predict(audio, fs=fs)

    model._sensevoice._available = False  # ensure it's treated as unavailable
    audio = _synthetic_audio(2.0)
    # real_time=False — SenseVoice path should be skipped entirely
    model.predict(audio, sample_rate=16000, real_time=False)
    assert len(calls) == 0, "SenseVoice should not be called when real_time=False"
