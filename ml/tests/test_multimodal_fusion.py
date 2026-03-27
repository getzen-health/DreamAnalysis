"""Tests for attention-based multimodal fusion (processing/multimodal_fusion.py).

Covers:
- Single modality passthrough
- Equal quality → roughly equal weights
- High quality EEG + low quality voice → EEG dominates
- Valence agreement → higher agreement score
- Valence disagreement → lower agreement, weights shift to higher quality
- Three modalities (EEG + voice + health)
- Fused probabilities sum to ~1.0
- Fused valence is between input valences
- dominant_emotion matches highest fused probability
"""

import pytest
from processing.multimodal_fusion import AttentionFusion, ModalityInput, FusionResult


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_input(
    modality: str = "eeg",
    emotion: str = "happy",
    confidence: float = 0.8,
    signal_quality: float = 0.9,
    valence: float = 0.5,
    arousal: float = 0.6,
) -> ModalityInput:
    """Build a ModalityInput with the given emotion dominant."""
    probs = {e: 0.05 for e in AttentionFusion.EMOTIONS}
    probs[emotion] = 0.75  # make it dominant
    total = sum(probs.values())
    probs = {e: v / total for e, v in probs.items()}
    return ModalityInput(
        probabilities=probs,
        valence=valence,
        arousal=arousal,
        confidence=confidence,
        signal_quality=signal_quality,
        modality=modality,
    )


# ── Single modality ─────────────────────────────────────────────────────

def test_single_modality_passthrough():
    """Single modality input returns its predictions unchanged."""
    fusion = AttentionFusion()
    inp = _make_input("eeg", "happy", confidence=0.9, signal_quality=0.8,
                      valence=0.6, arousal=0.7)
    result = fusion.fuse([inp])

    assert result.dominant_emotion == "happy"
    assert result.valence == pytest.approx(0.6, abs=1e-4)
    assert result.arousal == pytest.approx(0.7, abs=1e-4)
    assert result.confidence == pytest.approx(0.9, abs=1e-4)
    assert result.weights_used == {"eeg": 1.0}
    assert result.agreement_score == 1.0
    # Probabilities should match input
    for e in AttentionFusion.EMOTIONS:
        assert result.probabilities[e] == pytest.approx(inp.probabilities[e], abs=1e-4)


# ── Equal quality → roughly equal weights ────────────────────────────────

def test_equal_quality_equal_weights():
    """Two modalities with identical quality/confidence → near-equal weights."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "happy", confidence=0.8, signal_quality=0.8,
                      valence=0.5, arousal=0.6)
    voice = _make_input("voice", "happy", confidence=0.8, signal_quality=0.8,
                        valence=0.5, arousal=0.6)
    result = fusion.fuse([eeg, voice])

    # With identical scores, softmax should give equal weights
    assert result.weights_used["eeg"] == pytest.approx(0.5, abs=0.01)
    assert result.weights_used["voice"] == pytest.approx(0.5, abs=0.01)


# ── High quality EEG + low quality voice → EEG dominates ────────────────

def test_high_quality_eeg_dominates():
    """High quality EEG + low quality voice → EEG gets higher weight."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "happy", confidence=0.95, signal_quality=0.95,
                      valence=0.7, arousal=0.6)
    voice = _make_input("voice", "sad", confidence=0.3, signal_quality=0.2,
                        valence=-0.5, arousal=0.3)
    result = fusion.fuse([eeg, voice])

    assert result.weights_used["eeg"] > result.weights_used["voice"]
    assert result.weights_used["eeg"] > 0.6  # EEG should dominate substantially
    # Fused valence should lean toward EEG's positive valence
    assert result.valence > 0.0


# ── Valence agreement → higher agreement score ──────────────────────────

def test_valence_agreement_high_score():
    """EEG and voice both positive valence → high agreement score."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "happy", valence=0.6, confidence=0.8,
                      signal_quality=0.8)
    voice = _make_input("voice", "happy", valence=0.4, confidence=0.8,
                        signal_quality=0.8)
    result = fusion.fuse([eeg, voice])

    assert result.agreement_score == 1.0  # Both positive


def test_valence_agreement_both_negative():
    """EEG and voice both negative → full agreement."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "sad", valence=-0.4, confidence=0.7,
                      signal_quality=0.7)
    voice = _make_input("voice", "sad", valence=-0.6, confidence=0.7,
                        signal_quality=0.7)
    result = fusion.fuse([eeg, voice])

    assert result.agreement_score == 1.0


# ── Valence disagreement → lower agreement, weights shift ────────────────

def test_valence_disagreement_low_score():
    """EEG positive, voice negative → agreement < 1.0."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "happy", valence=0.7, confidence=0.8,
                      signal_quality=0.8)
    voice = _make_input("voice", "sad", valence=-0.5, confidence=0.8,
                        signal_quality=0.8)
    result = fusion.fuse([eeg, voice])

    assert result.agreement_score < 1.0
    assert result.agreement_score == pytest.approx(0.5, abs=0.01)


def test_disagreement_shifts_to_higher_quality():
    """When modalities disagree, higher quality modality gets more weight."""
    fusion = AttentionFusion()
    # EEG: high quality, positive
    eeg = _make_input("eeg", "happy", confidence=0.9, signal_quality=0.9,
                      valence=0.7, arousal=0.6)
    # Voice: low quality, negative
    voice = _make_input("voice", "sad", confidence=0.4, signal_quality=0.3,
                        valence=-0.6, arousal=0.3)
    result = fusion.fuse([eeg, voice])

    # EEG should dominate due to higher quality + agreement bonus (it's the majority)
    assert result.weights_used["eeg"] > result.weights_used["voice"]
    # Fused result should lean positive (toward EEG)
    assert result.valence > 0.0


# ── Three modalities (EEG + voice + health) ─────────────────────────────

def test_three_modalities_all_contribute():
    """Three modalities all get non-zero weights."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "happy", confidence=0.8, signal_quality=0.9,
                      valence=0.5, arousal=0.6)
    voice = _make_input("voice", "happy", confidence=0.7, signal_quality=0.7,
                        valence=0.4, arousal=0.5)
    health = _make_input("health", "neutral", confidence=0.6, signal_quality=0.6,
                         valence=0.1, arousal=0.4)
    result = fusion.fuse([eeg, voice, health])

    assert "eeg" in result.weights_used
    assert "voice" in result.weights_used
    assert "health" in result.weights_used
    # All should have positive weight
    assert all(w > 0 for w in result.weights_used.values())
    # Weights should sum to ~1.0
    assert sum(result.weights_used.values()) == pytest.approx(1.0, abs=1e-6)


def test_three_modalities_quality_ordering():
    """Higher quality modalities get higher weight among three inputs."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "happy", confidence=0.95, signal_quality=0.95,
                      valence=0.5, arousal=0.6)
    voice = _make_input("voice", "happy", confidence=0.6, signal_quality=0.5,
                        valence=0.4, arousal=0.5)
    health = _make_input("health", "happy", confidence=0.3, signal_quality=0.3,
                         valence=0.3, arousal=0.4)
    result = fusion.fuse([eeg, voice, health])

    assert result.weights_used["eeg"] > result.weights_used["voice"]
    assert result.weights_used["voice"] > result.weights_used["health"]


# ── Fused probabilities sum to ~1.0 ─────────────────────────────────────

def test_fused_probabilities_sum_to_one():
    """Fused probabilities should sum to approximately 1.0."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "happy", confidence=0.8, signal_quality=0.9,
                      valence=0.5, arousal=0.6)
    voice = _make_input("voice", "sad", confidence=0.6, signal_quality=0.5,
                        valence=-0.3, arousal=0.4)
    result = fusion.fuse([eeg, voice])

    prob_sum = sum(result.probabilities.values())
    assert prob_sum == pytest.approx(1.0, abs=1e-4)


def test_three_modality_probabilities_sum_to_one():
    """Three-modality fused probabilities also sum to 1.0."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "angry", confidence=0.7, signal_quality=0.8)
    voice = _make_input("voice", "fear", confidence=0.5, signal_quality=0.4)
    health = _make_input("health", "neutral", confidence=0.6, signal_quality=0.6)
    result = fusion.fuse([eeg, voice, health])

    prob_sum = sum(result.probabilities.values())
    assert prob_sum == pytest.approx(1.0, abs=1e-4)


# ── Fused valence is between input valences ──────────────────────────────

def test_fused_valence_between_inputs():
    """Fused valence should be a weighted average → between min and max input."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "happy", valence=0.8, confidence=0.7,
                      signal_quality=0.7)
    voice = _make_input("voice", "sad", valence=-0.4, confidence=0.7,
                        signal_quality=0.7)
    result = fusion.fuse([eeg, voice])

    assert result.valence >= -0.4
    assert result.valence <= 0.8


def test_fused_arousal_between_inputs():
    """Fused arousal should fall between min and max of inputs."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "happy", arousal=0.9, confidence=0.8,
                      signal_quality=0.8)
    voice = _make_input("voice", "neutral", arousal=0.2, confidence=0.8,
                        signal_quality=0.8)
    result = fusion.fuse([eeg, voice])

    assert result.arousal >= 0.2
    assert result.arousal <= 0.9


# ── dominant_emotion matches highest fused probability ───────────────────

def test_dominant_emotion_matches_highest_probability():
    """dominant_emotion should be the emotion with max fused probability."""
    fusion = AttentionFusion()
    # Both strongly predict happy
    eeg = _make_input("eeg", "happy", confidence=0.9, signal_quality=0.9)
    voice = _make_input("voice", "happy", confidence=0.9, signal_quality=0.9)
    result = fusion.fuse([eeg, voice])

    max_emotion = max(result.probabilities, key=result.probabilities.get)
    assert result.dominant_emotion == max_emotion
    assert result.dominant_emotion == "happy"


def test_dominant_emotion_reflects_weighted_fusion():
    """When EEG dominates, its emotion should be the dominant one."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "angry", confidence=0.95, signal_quality=0.95,
                      valence=-0.3)
    voice = _make_input("voice", "neutral", confidence=0.2, signal_quality=0.1,
                        valence=0.0)
    result = fusion.fuse([eeg, voice])

    max_emotion = max(result.probabilities, key=result.probabilities.get)
    assert result.dominant_emotion == max_emotion
    # EEG's "angry" should dominate since it has much higher quality
    assert result.dominant_emotion == "angry"


# ── Edge cases ───────────────────────────────────────────────────────────

def test_empty_inputs_raises():
    """Empty input list should raise ValueError."""
    fusion = AttentionFusion()
    with pytest.raises(ValueError, match="At least one"):
        fusion.fuse([])


def test_near_zero_valence_agreement():
    """Near-zero valences should count as agreement."""
    fusion = AttentionFusion()
    eeg = _make_input("eeg", "neutral", valence=0.02, confidence=0.7,
                      signal_quality=0.7)
    voice = _make_input("voice", "neutral", valence=-0.03, confidence=0.7,
                        signal_quality=0.7)
    result = fusion.fuse([eeg, voice])

    assert result.agreement_score == 1.0


def test_custom_component_weights():
    """Custom quality/confidence/agreement weights should be respected."""
    # Quality-only fusion
    fusion = AttentionFusion(quality_weight=1.0, confidence_weight=0.0,
                             agreement_weight=0.0)
    eeg = _make_input("eeg", "happy", confidence=0.1, signal_quality=0.9,
                      valence=0.5)
    voice = _make_input("voice", "happy", confidence=0.9, signal_quality=0.1,
                        valence=0.4)
    result = fusion.fuse([eeg, voice])

    # EEG has much higher signal_quality; should dominate despite low confidence
    assert result.weights_used["eeg"] > result.weights_used["voice"]


def test_confidence_only_fusion():
    """When only confidence matters, high-confidence modality dominates."""
    fusion = AttentionFusion(quality_weight=0.0, confidence_weight=1.0,
                             agreement_weight=0.0)
    eeg = _make_input("eeg", "sad", confidence=0.3, signal_quality=0.9,
                      valence=-0.5)
    voice = _make_input("voice", "happy", confidence=0.95, signal_quality=0.1,
                        valence=0.7)
    result = fusion.fuse([eeg, voice])

    # Voice has much higher confidence; should dominate
    assert result.weights_used["voice"] > result.weights_used["eeg"]
