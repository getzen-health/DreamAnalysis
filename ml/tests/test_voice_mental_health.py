"""Tests for multi-condition mental health voice screening."""
import numpy as np
import pytest
from models.voice_mental_health import VoiceMentalHealthScreener, CONDITIONS, _risk_level


FS = 16000


def _synthetic_speech(seconds: float = 35.0) -> np.ndarray:
    rng = np.random.default_rng(123)
    t = np.linspace(0, seconds, int(seconds * FS))
    # Simulate voiced speech: 150Hz fundamental + harmonics + noise
    audio = (
        0.3 * np.sin(2 * np.pi * 150 * t)
        + 0.15 * np.sin(2 * np.pi * 300 * t)
        + 0.05 * rng.normal(0, 1, len(t))
    ).astype(np.float32)
    return audio


@pytest.fixture(scope="module")
def screener():
    return VoiceMentalHealthScreener()


def test_screener_initializes(screener):
    assert screener is not None


def test_screen_returns_all_conditions(screener):
    audio = _synthetic_speech(35.0)
    result = screener.screen(audio, fs=FS)
    assert "conditions" in result
    for cond in CONDITIONS:
        assert cond in result["conditions"]


def test_risk_scores_in_range(screener):
    audio = _synthetic_speech(35.0)
    result = screener.screen(audio, fs=FS)
    for cond in CONDITIONS:
        score = result["conditions"][cond]["risk_score"]
        assert 0.0 <= score <= 1.0, f"{cond} score {score} out of range"


def test_risk_levels_valid(screener):
    audio = _synthetic_speech(35.0)
    result = screener.screen(audio, fs=FS)
    valid_levels = {"minimal", "mild", "moderate", "elevated", "unknown"}
    for cond in CONDITIONS:
        level = result["conditions"][cond]["risk_level"]
        assert level in valid_levels, f"Invalid risk level for {cond}: {level}"


def test_too_short_audio_handled(screener):
    """Short audio should return error gracefully, not crash."""
    short = np.zeros(1000, dtype=np.float32)
    result = screener.screen(short, fs=FS)
    assert "conditions" in result
    assert result["overall_risk"] == "unknown"


def test_risk_level_function():
    assert _risk_level(0.0) == "minimal"
    assert _risk_level(0.35) == "mild"
    assert _risk_level(0.55) == "moderate"
    assert _risk_level(0.75) == "elevated"


def test_result_has_disclaimer(screener):
    audio = _synthetic_speech(35.0)
    result = screener.screen(audio, fs=FS)
    assert "disclaimer" in result


def test_result_has_recommendations(screener):
    audio = _synthetic_speech(35.0)
    result = screener.screen(audio, fs=FS)
    assert "recommendations" in result
    assert len(result["recommendations"]) >= 1


def test_overall_risk_field_present(screener):
    audio = _synthetic_speech(35.0)
    result = screener.screen(audio, fs=FS)
    assert "overall_risk" in result
    assert result["overall_risk"] in {"minimal", "mild", "moderate", "elevated", "unknown"}


def test_method_field_present(screener):
    audio = _synthetic_speech(35.0)
    result = screener.screen(audio, fs=FS)
    assert "method" in result
    assert result["method"] in {"whisper_encoder", "prosodic_heuristic", "none"}


def test_singleton_returns_same_instance():
    from models.voice_mental_health import get_mh_screener
    s1 = get_mh_screener()
    s2 = get_mh_screener()
    assert s1 is s2


def test_resampling_handled(screener):
    """Audio at 22050 Hz should be resampled and produce valid results."""
    fs_in = 22050
    audio = _synthetic_speech(35.0)
    # Stretch to simulate 22050 Hz input
    n_out = int(len(audio) * fs_in / FS)
    indices = np.round(np.linspace(0, len(audio) - 1, n_out)).astype(int)
    audio_22k = audio[indices]
    result = screener.screen(audio_22k, fs=fs_in)
    assert "conditions" in result
    for cond in CONDITIONS:
        assert 0.0 <= result["conditions"][cond]["risk_score"] <= 1.0
