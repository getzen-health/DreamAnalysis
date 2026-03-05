"""Test that voice cache is correctly retrieved and available for fusion."""
import time


def test_voice_cache_roundtrip():
    """Voice result stored in cache is retrievable."""
    from api.routes.voice_watch import _VOICE_CACHE, get_latest_voice, cache_voice_result, CacheRequest

    _VOICE_CACHE.clear()
    payload = {
        "emotion": "happy", "valence": 0.7, "arousal": 0.6,
        "confidence": 0.85, "probabilities": {}, "model_type": "voice_emotion2vec",
    }
    req = CacheRequest(user_id="ws_test", emotion_result=payload)
    cache_voice_result(req)
    result = get_latest_voice("ws_test")
    assert result is not None
    assert result["emotion"] == "happy"
    _VOICE_CACHE.clear()


def test_multimodal_fusion_accepts_voice_result():
    """MultimodalEmotionFusion.fuse() must accept voice_result kwarg."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from api.routes._shared import fusion_model

    eeg = {
        "emotion": "neutral", "valence": 0.0, "arousal": 0.5,
        "confidence": 0.7, "probabilities": {}, "model_type": "eeg",
        "stress_index": 0.3, "focus_index": 0.5,
    }
    bio = {}
    voice = {
        "emotion": "happy", "valence": 0.8, "arousal": 0.7,
        "confidence": 0.85, "probabilities": {}, "model_type": "voice_emotion2vec",
    }

    from models.multimodal_emotion_fusion import BiometricSnapshot
    bio_snap = BiometricSnapshot()

    # Should not raise
    result = fusion_model.fuse(eeg, bio_snap, voice_result=voice)
    assert isinstance(result, dict)
    assert "emotion" in result
    # Voice should have blended valence upward from 0.0 toward 0.8
    assert result["valence"] > 0.0, f"Expected valence > 0 after voice blend, got {result['valence']}"
    # Voice metadata should be present
    assert result.get("voice_emotion") == "happy"
    assert result.get("voice_confidence") == 0.85


def test_fusion_without_voice_result_unchanged():
    """fuse() with voice_result=None returns same structure as before (no regression)."""
    from models.multimodal_emotion_fusion import MultimodalEmotionFusion, BiometricSnapshot

    fusion = MultimodalEmotionFusion()
    eeg = {
        "emotion": "neutral", "valence": 0.0, "arousal": 0.5,
        "confidence": 0.6, "probabilities": {}, "model_type": "eeg",
        "stress_index": 0.4, "focus_index": 0.5,
    }
    bio = BiometricSnapshot()
    result = fusion.fuse(eeg, bio, voice_result=None)
    assert isinstance(result, dict)
    assert "valence" in result
    assert "arousal" in result
    assert result.get("voice_emotion") is None
    assert result.get("voice_confidence") is None


def test_fusion_voice_low_confidence_ignored():
    """Voice result with confidence <= 0.3 must NOT affect valence/arousal."""
    from models.multimodal_emotion_fusion import MultimodalEmotionFusion, BiometricSnapshot

    fusion = MultimodalEmotionFusion()
    eeg = {
        "emotion": "neutral", "valence": 0.0, "arousal": 0.5,
        "confidence": 0.7, "probabilities": {}, "model_type": "eeg",
        "stress_index": 0.3, "focus_index": 0.5,
    }
    bio = BiometricSnapshot()
    # Low-confidence voice result
    voice_low = {
        "emotion": "angry", "valence": -0.9, "arousal": 0.95,
        "confidence": 0.2, "probabilities": {}, "model_type": "voice_emotion2vec",
    }
    result_no_voice = fusion.fuse(eeg, bio, voice_result=None)
    result_low_voice = fusion.fuse(eeg, bio, voice_result=voice_low)

    # Low confidence voice should not shift valence or arousal
    assert abs(result_no_voice["valence"] - result_low_voice["valence"]) < 1e-6, (
        "Low-confidence voice should not change valence"
    )
    assert abs(result_no_voice["arousal"] - result_low_voice["arousal"]) < 1e-6, (
        "Low-confidence voice should not change arousal"
    )
