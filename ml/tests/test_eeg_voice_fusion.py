"""Tests for EEG+Voice decision-level multimodal emotion fusion.

Covers:
- Instantiation and default state
- fuse() with equal-quality inputs
- fuse() with quality-adjusted weighting
- set_weights() custom weight configuration
- Conflict detection between modalities
- Agreement scoring
- Dominant modality detection
- Probability normalization invariants
- Session statistics tracking
- History accumulation and retrieval
- reset() clears state
- Edge cases: zero quality, extreme skew, single-emotion dominance
- Output structure validation
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add ml/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.eeg_voice_fusion import EEGVoiceFusion, EMOTIONS_6


# ---- Fixtures ----------------------------------------------------------------

@pytest.fixture
def fusion():
    """Fresh EEGVoiceFusion instance with default weights."""
    return EEGVoiceFusion()


@pytest.fixture
def happy_eeg_probs():
    """EEG prediction strongly favoring happy."""
    return {
        "happy": 0.70, "sad": 0.05, "angry": 0.05,
        "fear": 0.05, "surprise": 0.10, "neutral": 0.05,
    }


@pytest.fixture
def happy_voice_probs():
    """Voice prediction strongly favoring happy (agreement with EEG)."""
    return {
        "happy": 0.65, "sad": 0.05, "angry": 0.05,
        "fear": 0.05, "surprise": 0.10, "neutral": 0.10,
    }


@pytest.fixture
def sad_voice_probs():
    """Voice prediction strongly favoring sad (conflict with happy EEG)."""
    return {
        "happy": 0.05, "sad": 0.70, "angry": 0.05,
        "fear": 0.05, "surprise": 0.05, "neutral": 0.10,
    }


@pytest.fixture
def uniform_probs():
    """Uniform probability distribution across all 6 emotions."""
    return {e: 1.0 / 6.0 for e in EMOTIONS_6}


# ---- Instantiation -----------------------------------------------------------

def test_instantiation_defaults(fusion):
    assert fusion is not None


def test_default_weights(fusion):
    result = fusion.fuse(
        {"happy": 0.5, "sad": 0.1, "angry": 0.1, "fear": 0.1, "surprise": 0.1, "neutral": 0.1},
        {"happy": 0.5, "sad": 0.1, "angry": 0.1, "fear": 0.1, "surprise": 0.1, "neutral": 0.1},
    )
    weights = result["modality_weights"]
    # Default: EEG 0.6, voice 0.4 (before quality adjustment)
    assert abs(weights["eeg_base"] - 0.6) < 1e-6
    assert abs(weights["voice_base"] - 0.4) < 1e-6


# ---- fuse() output structure -------------------------------------------------

def test_fuse_returns_dict(fusion, happy_eeg_probs, happy_voice_probs):
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs)
    assert isinstance(result, dict)


def test_fuse_has_required_keys(fusion, happy_eeg_probs, happy_voice_probs):
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs)
    required = {
        "fused_probabilities", "fused_emotion", "confidence",
        "agreement_score", "conflict_detected", "modality_weights",
        "dominant_modality",
    }
    assert required.issubset(result.keys()), f"Missing: {required - result.keys()}"


def test_fused_emotion_in_emotions_6(fusion, happy_eeg_probs, happy_voice_probs):
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs)
    assert result["fused_emotion"] in EMOTIONS_6


def test_fused_probabilities_keys(fusion, happy_eeg_probs, happy_voice_probs):
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs)
    assert set(result["fused_probabilities"].keys()) == set(EMOTIONS_6)


def test_fused_probabilities_sum_to_one(fusion, happy_eeg_probs, happy_voice_probs):
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs)
    total = sum(result["fused_probabilities"].values())
    assert abs(total - 1.0) < 1e-5, f"Sum is {total}, expected ~1.0"


def test_fused_probabilities_non_negative(fusion, happy_eeg_probs, happy_voice_probs):
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs)
    for emotion, prob in result["fused_probabilities"].items():
        assert prob >= 0.0, f"{emotion} has negative probability {prob}"


def test_confidence_range(fusion, happy_eeg_probs, happy_voice_probs):
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs)
    assert 0.0 <= result["confidence"] <= 1.0


def test_agreement_score_range(fusion, happy_eeg_probs, happy_voice_probs):
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs)
    assert 0.0 <= result["agreement_score"] <= 1.0


# ---- Agreement and conflict detection ----------------------------------------

def test_high_agreement_when_both_agree(fusion, happy_eeg_probs, happy_voice_probs):
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs)
    # Both predict happy strongly -> high agreement
    assert result["agreement_score"] > 0.6
    assert result["conflict_detected"] is False


def test_conflict_detected_when_disagree(fusion, happy_eeg_probs, sad_voice_probs):
    result = fusion.fuse(happy_eeg_probs, sad_voice_probs)
    assert result["conflict_detected"] is True


def test_low_agreement_when_disagree(fusion, happy_eeg_probs, sad_voice_probs):
    result = fusion.fuse(happy_eeg_probs, sad_voice_probs)
    assert result["agreement_score"] < 0.5


def test_perfect_agreement_identical_inputs(fusion):
    probs = {"happy": 0.5, "sad": 0.1, "angry": 0.1, "fear": 0.1, "surprise": 0.1, "neutral": 0.1}
    result = fusion.fuse(probs, probs.copy())
    assert result["agreement_score"] > 0.95
    assert result["conflict_detected"] is False


# ---- Quality-adjusted weighting ----------------------------------------------

def test_eeg_quality_affects_weight(fusion, happy_eeg_probs, sad_voice_probs):
    # High EEG quality, low voice quality -> should lean toward EEG (happy)
    result = fusion.fuse(happy_eeg_probs, sad_voice_probs, eeg_quality=1.0, voice_quality=0.1)
    assert result["fused_emotion"] == "happy"


def test_voice_quality_affects_weight(fusion, happy_eeg_probs, sad_voice_probs):
    # Low EEG quality, high voice quality -> should lean toward voice (sad)
    result = fusion.fuse(happy_eeg_probs, sad_voice_probs, eeg_quality=0.1, voice_quality=1.0)
    assert result["fused_emotion"] == "sad"


def test_zero_eeg_quality_uses_voice_only(fusion, happy_eeg_probs, sad_voice_probs):
    result = fusion.fuse(happy_eeg_probs, sad_voice_probs, eeg_quality=0.0, voice_quality=1.0)
    # Should be entirely voice-driven
    assert result["fused_emotion"] == "sad"
    assert result["dominant_modality"] == "voice"


def test_zero_voice_quality_uses_eeg_only(fusion, happy_eeg_probs, sad_voice_probs):
    result = fusion.fuse(happy_eeg_probs, sad_voice_probs, eeg_quality=1.0, voice_quality=0.0)
    assert result["fused_emotion"] == "happy"
    assert result["dominant_modality"] == "eeg"


def test_equal_quality_uses_base_weights(fusion, happy_eeg_probs, happy_voice_probs):
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs, eeg_quality=1.0, voice_quality=1.0)
    weights = result["modality_weights"]
    # With equal quality, effective weights should match base weights
    assert abs(weights["eeg_effective"] - 0.6) < 0.01
    assert abs(weights["voice_effective"] - 0.4) < 0.01


def test_modality_weights_sum_to_one(fusion, happy_eeg_probs, happy_voice_probs):
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs, eeg_quality=0.7, voice_quality=0.3)
    weights = result["modality_weights"]
    total = weights["eeg_effective"] + weights["voice_effective"]
    assert abs(total - 1.0) < 1e-5


# ---- set_weights() -----------------------------------------------------------

def test_set_weights(fusion):
    fusion.set_weights(eeg_weight=0.8, voice_weight=0.2)
    probs = {"happy": 0.5, "sad": 0.1, "angry": 0.1, "fear": 0.1, "surprise": 0.1, "neutral": 0.1}
    result = fusion.fuse(probs, probs)
    assert abs(result["modality_weights"]["eeg_base"] - 0.8) < 1e-6
    assert abs(result["modality_weights"]["voice_base"] - 0.2) < 1e-6


def test_set_weights_normalizes(fusion):
    # Weights that don't sum to 1 should be normalized
    fusion.set_weights(eeg_weight=3.0, voice_weight=2.0)
    probs = {"happy": 0.5, "sad": 0.1, "angry": 0.1, "fear": 0.1, "surprise": 0.1, "neutral": 0.1}
    result = fusion.fuse(probs, probs)
    assert abs(result["modality_weights"]["eeg_base"] - 0.6) < 1e-6
    assert abs(result["modality_weights"]["voice_base"] - 0.4) < 1e-6


# ---- Dominant modality -------------------------------------------------------

def test_dominant_modality_eeg(fusion, happy_eeg_probs, happy_voice_probs):
    # Default EEG weight 0.6 > voice 0.4
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs, eeg_quality=1.0, voice_quality=1.0)
    assert result["dominant_modality"] == "eeg"


def test_dominant_modality_voice(fusion, happy_eeg_probs, happy_voice_probs):
    fusion.set_weights(eeg_weight=0.3, voice_weight=0.7)
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs, eeg_quality=1.0, voice_quality=1.0)
    assert result["dominant_modality"] == "voice"


def test_dominant_modality_quality_override(fusion, happy_eeg_probs, happy_voice_probs):
    # Base weights favor EEG, but voice quality is much higher
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs, eeg_quality=0.1, voice_quality=1.0)
    assert result["dominant_modality"] == "voice"


# ---- Session statistics and history ------------------------------------------

def test_get_session_stats_initial(fusion):
    stats = fusion.get_session_stats()
    assert stats["fusion_count"] == 0
    assert stats["conflict_count"] == 0


def test_get_session_stats_after_fuse(fusion, happy_eeg_probs, happy_voice_probs):
    fusion.fuse(happy_eeg_probs, happy_voice_probs)
    stats = fusion.get_session_stats()
    assert stats["fusion_count"] == 1


def test_get_session_stats_conflict_tracking(fusion, happy_eeg_probs, sad_voice_probs, happy_voice_probs):
    fusion.fuse(happy_eeg_probs, sad_voice_probs)
    fusion.fuse(happy_eeg_probs, happy_voice_probs)
    stats = fusion.get_session_stats()
    assert stats["fusion_count"] == 2
    assert stats["conflict_count"] == 1
    assert 0.0 <= stats["conflict_rate"] <= 1.0


def test_get_history_empty(fusion):
    history = fusion.get_history()
    assert isinstance(history, list)
    assert len(history) == 0


def test_get_history_accumulates(fusion, happy_eeg_probs, happy_voice_probs):
    fusion.fuse(happy_eeg_probs, happy_voice_probs)
    fusion.fuse(happy_eeg_probs, happy_voice_probs)
    history = fusion.get_history()
    assert len(history) == 2


def test_get_history_entry_structure(fusion, happy_eeg_probs, happy_voice_probs):
    fusion.fuse(happy_eeg_probs, happy_voice_probs)
    entry = fusion.get_history()[0]
    assert "fused_emotion" in entry
    assert "confidence" in entry
    assert "agreement_score" in entry
    assert "conflict_detected" in entry


# ---- reset() -----------------------------------------------------------------

def test_reset_clears_history(fusion, happy_eeg_probs, happy_voice_probs):
    fusion.fuse(happy_eeg_probs, happy_voice_probs)
    assert len(fusion.get_history()) == 1
    fusion.reset()
    assert len(fusion.get_history()) == 0


def test_reset_clears_stats(fusion, happy_eeg_probs, happy_voice_probs):
    fusion.fuse(happy_eeg_probs, happy_voice_probs)
    fusion.reset()
    stats = fusion.get_session_stats()
    assert stats["fusion_count"] == 0
    assert stats["conflict_count"] == 0


# ---- Edge cases --------------------------------------------------------------

def test_both_zero_quality(fusion, happy_eeg_probs, happy_voice_probs):
    # Both zero quality -> should still return valid result (fallback to equal)
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs, eeg_quality=0.0, voice_quality=0.0)
    assert result["fused_emotion"] in EMOTIONS_6
    total = sum(result["fused_probabilities"].values())
    assert abs(total - 1.0) < 1e-5


def test_single_emotion_dominance(fusion):
    # One modality is 100% certain
    eeg = {"happy": 1.0, "sad": 0.0, "angry": 0.0, "fear": 0.0, "surprise": 0.0, "neutral": 0.0}
    voice = {"happy": 0.0, "sad": 1.0, "angry": 0.0, "fear": 0.0, "surprise": 0.0, "neutral": 0.0}
    result = fusion.fuse(eeg, voice)
    assert result["fused_emotion"] in EMOTIONS_6
    total = sum(result["fused_probabilities"].values())
    assert abs(total - 1.0) < 1e-5
    assert result["conflict_detected"] is True


def test_uniform_inputs_neutral_or_valid(fusion, uniform_probs):
    result = fusion.fuse(uniform_probs, uniform_probs.copy())
    assert result["fused_emotion"] in EMOTIONS_6
    # Uniform -> high agreement
    assert result["agreement_score"] > 0.9


def test_quality_clipped_to_valid_range(fusion, happy_eeg_probs, happy_voice_probs):
    # Quality values outside [0,1] should be clipped
    result = fusion.fuse(happy_eeg_probs, happy_voice_probs, eeg_quality=2.0, voice_quality=-0.5)
    assert result["fused_emotion"] in EMOTIONS_6
    weights = result["modality_weights"]
    assert 0.0 <= weights["eeg_effective"] <= 1.0
    assert 0.0 <= weights["voice_effective"] <= 1.0


def test_session_stats_has_mean_agreement(fusion, happy_eeg_probs, happy_voice_probs, sad_voice_probs):
    fusion.fuse(happy_eeg_probs, happy_voice_probs)
    fusion.fuse(happy_eeg_probs, sad_voice_probs)
    stats = fusion.get_session_stats()
    assert "mean_agreement" in stats
    assert 0.0 <= stats["mean_agreement"] <= 1.0


def test_session_stats_has_mean_confidence(fusion, happy_eeg_probs, happy_voice_probs):
    fusion.fuse(happy_eeg_probs, happy_voice_probs)
    stats = fusion.get_session_stats()
    assert "mean_confidence" in stats
    assert 0.0 <= stats["mean_confidence"] <= 1.0
