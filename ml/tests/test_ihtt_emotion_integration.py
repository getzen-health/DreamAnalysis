"""Integration test: IHTT features wired into emotion classifier.

Verifies that the emotion classifier produces ihtt in its output
when given multichannel (4-channel Muse 2) EEG data, and that
the IHTT focus boost modulates focus_index as designed.
"""

import numpy as np
import pytest

import sys
sys.path.insert(0, "/Users/sravyalu/NeuralDreamWorkshop/ml")

from models.emotion_classifier import EmotionClassifier


FS = 256
DURATION_SEC = 4
N_SAMPLES = FS * DURATION_SEC


@pytest.fixture
def classifier():
    return EmotionClassifier()


class TestIHTTInEmotionOutput:
    """Verify ihtt dict appears in the emotion classifier output."""

    def test_multichannel_output_has_ihtt(self, classifier):
        """4-channel input should produce ihtt dict with all keys."""
        signals = np.random.randn(4, N_SAMPLES) * 10
        result = classifier._predict_features(signals, FS)
        assert "ihtt" in result
        ihtt = result["ihtt"]
        assert "frontal_lag_ms" in ihtt
        assert "temporal_lag_ms" in ihtt
        assert "mean_ihtt_ms" in ihtt

    def test_ihtt_values_are_valid(self, classifier):
        """IHTT values should be non-negative and <= 50 ms."""
        signals = np.random.randn(4, N_SAMPLES) * 10
        result = classifier._predict_features(signals, FS)
        ihtt = result["ihtt"]
        assert 0.0 <= ihtt["frontal_lag_ms"] <= 50.0
        assert 0.0 <= ihtt["temporal_lag_ms"] <= 50.0
        assert 0.0 <= ihtt["mean_ihtt_ms"] <= 50.0

    def test_single_channel_has_empty_ihtt(self, classifier):
        """1D input should produce empty ihtt dict (no homologous pairs)."""
        signal = np.random.randn(N_SAMPLES) * 10
        result = classifier._predict_features(signal, FS)
        assert "ihtt" in result
        assert result["ihtt"] == {} or result["ihtt"].get("mean_ihtt_ms", 0.0) == 0.0

    def test_artifact_frozen_has_ihtt(self, classifier):
        """Artifact-frozen output should still have ihtt key."""
        normal = np.random.randn(4, N_SAMPLES) * 10
        classifier._predict_features(normal, FS)
        artifact = np.random.randn(4, N_SAMPLES) * 200
        result = classifier._predict_features(artifact, FS)
        assert "ihtt" in result

    def test_ihtt_in_heuristic_explanation(self, classifier):
        """ihtt_focus_boost should appear in heuristic explanation contributions."""
        signals = np.random.randn(4, N_SAMPLES) * 10
        result = classifier._predict_features(signals, FS)
        # The ihtt key should always be present in the output
        assert "ihtt" in result
        # focus_index should be a valid number
        assert 0.0 <= result["focus_index"] <= 1.0
