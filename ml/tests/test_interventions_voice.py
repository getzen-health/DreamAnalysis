"""Tests for voice emotion triggers in the intervention engine."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _get_check_func():
    """Return the sync core logic (not the async FastAPI handler)."""
    from api.routes.interventions import _check_intervention_logic
    return _check_intervention_logic


def test_check_accepts_voice_emotion_param():
    """The check endpoint must accept voice_emotion without raising."""
    from api.routes.interventions import CheckRequest, _state
    _state.clear()

    req = CheckRequest(
        user_id="test_voice_001",
        voice_emotion={
            "emotion": "sad",
            "valence": -0.5,
            "arousal": 0.3,
            "confidence": 0.8,
            "probabilities": {},
            "model_type": "voice_emotion2vec",
        },
    )
    # Should not raise
    result = _get_check_func()(req)
    assert isinstance(result, dict)


def test_check_voice_high_arousal_may_trigger():
    """High arousal voice (stress proxy) should be evaluated for intervention."""
    from api.routes.interventions import CheckRequest, _state
    _state.clear()

    req = CheckRequest(
        user_id="test_voice_002",
        voice_emotion={
            "emotion": "angry",
            "valence": -0.7,
            "arousal": 0.85,
            "confidence": 0.9,
            "probabilities": {},
            "model_type": "voice_emotion2vec",
        },
    )
    result = _get_check_func()(req)
    assert isinstance(result, dict)
    # High arousal + high confidence + negative valence → should trigger voice_stress
    assert result["has_recommendation"] is True
    assert result["intervention"]["type"] == "voice_stress"
    assert result["intervention"]["source"] == "voice"


def test_check_without_voice_still_works():
    """Existing EEG-only check path must not be broken."""
    from api.routes.interventions import CheckRequest, _state
    _state.clear()

    req = CheckRequest(user_id="test_voice_003")
    result = _get_check_func()(req)
    assert isinstance(result, dict)


def test_voice_low_confidence_ignored():
    """Voice readings below 0.5 confidence must not trigger interventions."""
    from api.routes.interventions import CheckRequest, _state
    _state.clear()

    req = CheckRequest(
        user_id="test_voice_004",
        voice_emotion={
            "emotion": "angry",
            "valence": -0.9,
            "arousal": 0.95,
            "confidence": 0.3,  # below threshold
            "probabilities": {},
            "model_type": "voice_emotion2vec",
        },
    )
    result = _get_check_func()(req)
    assert isinstance(result, dict)
    # Low confidence → no intervention from voice
    assert result["has_recommendation"] is False


def test_voice_neutral_does_not_trigger():
    """Neutral / calm voice should not fire a voice_stress intervention."""
    from api.routes.interventions import CheckRequest, _state
    _state.clear()

    req = CheckRequest(
        user_id="test_voice_005",
        voice_emotion={
            "emotion": "neutral",
            "valence": 0.1,
            "arousal": 0.4,
            "confidence": 0.9,
            "probabilities": {},
            "model_type": "voice_emotion2vec",
        },
    )
    result = _get_check_func()(req)
    assert isinstance(result, dict)
    assert result["has_recommendation"] is False
