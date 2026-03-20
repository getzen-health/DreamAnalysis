"""Tests for adaptive quality-weighted EEG-voice fusion in MultimodalEmotionFusion.

Covers:
- Voice weight scales with voice confidence (higher confidence = more weight)
- Voice weight decreases when EEG confidence is much higher
- Disagreement detection when EEG and voice predict different emotions
- Disagreement penalty on biometric_confidence
- New output keys: voice_weight, eeg_weight, voice_disagreement
- Backward compatibility: no voice still works, low confidence still excluded
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.multimodal_emotion_fusion import BiometricSnapshot, MultimodalEmotionFusion


@pytest.fixture
def fusion():
    return MultimodalEmotionFusion()


@pytest.fixture
def base_eeg():
    return {
        "emotion": "neutral",
        "valence": 0.0,
        "arousal": 0.5,
        "confidence": 0.7,
        "probabilities": {},
        "model_type": "eeg",
        "stress_index": 0.3,
        "focus_index": 0.5,
        "relaxation_index": 0.5,
    }


@pytest.fixture
def bio():
    return BiometricSnapshot()


@pytest.fixture
def high_conf_voice():
    return {
        "emotion": "happy",
        "valence": 0.8,
        "arousal": 0.7,
        "confidence": 0.90,
        "snr_db": 25.0,
        "probabilities": {},
        "model_type": "voice_emotion2vec",
    }


@pytest.fixture
def low_conf_voice():
    return {
        "emotion": "sad",
        "valence": -0.6,
        "arousal": 0.2,
        "confidence": 0.35,
        "snr_db": 8.0,
        "probabilities": {},
        "model_type": "voice_emotion2vec",
    }


# ---- Output structure -------------------------------------------------------


def test_new_keys_present_with_voice(fusion, base_eeg, bio, high_conf_voice):
    result = fusion.fuse(base_eeg, bio, voice_result=high_conf_voice)
    assert "voice_weight" in result
    assert "eeg_weight" in result
    assert "voice_disagreement" in result


def test_new_keys_present_without_voice(fusion, base_eeg, bio):
    result = fusion.fuse(base_eeg, bio, voice_result=None)
    assert result["voice_weight"] == 0.0
    assert result["eeg_weight"] == 1.0
    assert result["voice_disagreement"] is False


def test_weights_sum_to_one(fusion, base_eeg, bio, high_conf_voice):
    result = fusion.fuse(base_eeg, bio, voice_result=high_conf_voice)
    total = result["voice_weight"] + result["eeg_weight"]
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"


# ---- Adaptive weighting -----------------------------------------------------


def test_high_voice_conf_gets_more_weight(fusion, base_eeg, bio):
    """High voice confidence should get a larger voice weight."""
    high_voice = {
        "emotion": "happy", "valence": 0.8, "arousal": 0.7,
        "confidence": 0.90, "probabilities": {},
    }
    low_voice = {
        "emotion": "happy", "valence": 0.8, "arousal": 0.7,
        "confidence": 0.40, "probabilities": {},
    }
    result_high = fusion.fuse(base_eeg, bio, voice_result=high_voice)
    result_low = fusion.fuse(base_eeg, bio, voice_result=low_voice)
    assert result_high["voice_weight"] > result_low["voice_weight"], (
        f"High-conf voice weight {result_high['voice_weight']} should exceed "
        f"low-conf voice weight {result_low['voice_weight']}"
    )


def test_low_eeg_conf_gives_voice_more_weight(fusion, bio, high_conf_voice):
    """When EEG confidence is low, voice should get a larger share."""
    low_eeg = {
        "emotion": "neutral", "valence": 0.0, "arousal": 0.5,
        "confidence": 0.20, "probabilities": {},
        "stress_index": 0.3, "focus_index": 0.5, "relaxation_index": 0.5,
    }
    high_eeg = {
        "emotion": "neutral", "valence": 0.0, "arousal": 0.5,
        "confidence": 0.90, "probabilities": {},
        "stress_index": 0.3, "focus_index": 0.5, "relaxation_index": 0.5,
    }
    result_low_eeg = fusion.fuse(low_eeg, bio, voice_result=high_conf_voice)
    result_high_eeg = fusion.fuse(high_eeg, bio, voice_result=high_conf_voice)
    assert result_low_eeg["voice_weight"] > result_high_eeg["voice_weight"]


def test_equal_confidence_gives_balanced_weight(fusion, bio):
    """Equal EEG and voice confidence should yield voice weight near 0.225 (half of 0.45)."""
    eeg = {
        "emotion": "neutral", "valence": 0.0, "arousal": 0.5,
        "confidence": 0.70, "probabilities": {},
        "stress_index": 0.3, "focus_index": 0.5, "relaxation_index": 0.5,
    }
    voice = {
        "emotion": "happy", "valence": 0.8, "arousal": 0.7,
        "confidence": 0.70, "probabilities": {},
    }
    result = fusion.fuse(eeg, bio, voice_result=voice)
    # With equal confidence, relative_voice = 0.5, voice_weight = 0.45 * 0.5 = 0.225
    assert abs(result["voice_weight"] - 0.225) < 0.01


def test_voice_weight_capped_at_max(fusion, bio):
    """Voice weight should never exceed _VOICE_MAX_WEIGHT (0.45)."""
    eeg = {
        "emotion": "neutral", "valence": 0.0, "arousal": 0.5,
        "confidence": 0.05,  # very low EEG confidence
        "probabilities": {},
        "stress_index": 0.3, "focus_index": 0.5, "relaxation_index": 0.5,
    }
    voice = {
        "emotion": "happy", "valence": 0.8, "arousal": 0.7,
        "confidence": 1.0,  # perfect voice confidence
        "probabilities": {},
    }
    result = fusion.fuse(eeg, bio, voice_result=voice)
    assert result["voice_weight"] <= 0.45 + 1e-6


# ---- Disagreement detection --------------------------------------------------


def test_disagreement_detected(fusion, base_eeg, bio):
    """When EEG says neutral and voice says happy, disagreement should be flagged."""
    voice = {
        "emotion": "happy", "valence": 0.8, "arousal": 0.7,
        "confidence": 0.85, "probabilities": {},
    }
    result = fusion.fuse(base_eeg, bio, voice_result=voice)
    assert result["voice_disagreement"] is True


def test_agreement_detected(fusion, bio):
    """When both EEG and voice agree, no disagreement flagged."""
    eeg = {
        "emotion": "happy", "valence": 0.5, "arousal": 0.6,
        "confidence": 0.7, "probabilities": {},
        "stress_index": 0.3, "focus_index": 0.5, "relaxation_index": 0.5,
    }
    voice = {
        "emotion": "happy", "valence": 0.8, "arousal": 0.7,
        "confidence": 0.85, "probabilities": {},
    }
    result = fusion.fuse(eeg, bio, voice_result=voice)
    assert result["voice_disagreement"] is False


def test_disagreement_lowers_confidence(fusion, base_eeg, bio):
    """Disagreement should lower biometric_confidence by 15%."""
    agreeing_voice = {
        "emotion": "neutral", "valence": 0.1, "arousal": 0.5,
        "confidence": 0.85, "probabilities": {},
    }
    disagreeing_voice = {
        "emotion": "angry", "valence": -0.5, "arousal": 0.9,
        "confidence": 0.85, "probabilities": {},
    }
    result_agree = fusion.fuse(base_eeg, bio, voice_result=agreeing_voice)
    result_disagree = fusion.fuse(base_eeg, bio, voice_result=disagreeing_voice)
    # Both have the same signal count, so the only difference is the 15% penalty
    assert result_disagree["biometric_confidence"] < result_agree["biometric_confidence"]


def test_no_disagreement_without_voice(fusion, base_eeg, bio):
    """No voice result means no disagreement."""
    result = fusion.fuse(base_eeg, bio, voice_result=None)
    assert result["voice_disagreement"] is False


# ---- Backward compatibility -------------------------------------------------


def test_low_confidence_voice_excluded(fusion, base_eeg, bio, low_conf_voice):
    """Voice with confidence <= 0.3 should still be excluded entirely."""
    very_low_voice = {
        "emotion": "angry", "valence": -0.9, "arousal": 0.95,
        "confidence": 0.2, "probabilities": {},
    }
    result_no_voice = fusion.fuse(base_eeg, bio, voice_result=None)
    result_excluded = fusion.fuse(base_eeg, bio, voice_result=very_low_voice)
    assert abs(result_no_voice["valence"] - result_excluded["valence"]) < 1e-6
    assert abs(result_no_voice["arousal"] - result_excluded["arousal"]) < 1e-6
    assert result_excluded["voice_weight"] == 0.0


def test_voice_still_blends_valence_upward(fusion, base_eeg, bio, high_conf_voice):
    """Voice with positive valence should still shift fused valence upward."""
    result = fusion.fuse(base_eeg, bio, voice_result=high_conf_voice)
    assert result["valence"] > 0.0, (
        f"Expected valence > 0 after blending with happy voice, got {result['valence']}"
    )


def test_existing_keys_preserved(fusion, base_eeg, bio, high_conf_voice):
    """All pre-existing output keys must still be present."""
    result = fusion.fuse(base_eeg, bio, voice_result=high_conf_voice)
    required = {
        "stress_index", "valence", "arousal", "focus_index", "relaxation_index",
        "emotion", "sleep_debt", "circadian_phase", "dream_readiness",
        "biometric_confidence", "signal_count", "signals_used", "model_type",
        "confidence", "probabilities", "voice_emotion", "voice_confidence",
    }
    missing = required - set(result.keys())
    assert not missing, f"Missing keys: {missing}"
