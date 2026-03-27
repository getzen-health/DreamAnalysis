"""Tests for the temporal LSTM emotion model wrapper."""

import numpy as np
import pytest
from pathlib import Path
import sys

# Ensure ml/ is on the path (conftest.py also does this)
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.temporal_emotion_model import TemporalEmotionClassifier, TORCH_AVAILABLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier():
    """Classifier with no trained model -- uses heuristic fallback."""
    return TemporalEmotionClassifier(model_path=None, seq_length=5, input_dim=41)


@pytest.fixture
def make_features():
    """Factory that produces random feature vectors of the right shape."""
    def _make(dim: int = 41) -> np.ndarray:
        return np.random.randn(dim).astype(np.float32)
    return _make


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_no_model(self, classifier):
        """Classifier initialises cleanly without a model path."""
        assert classifier.model_loaded is False
        assert classifier.model is None
        assert classifier.seq_length == 5
        assert classifier.input_dim == 41
        assert classifier.buffer == []

    def test_init_nonexistent_model_path(self):
        """Non-existent model path does not crash -- falls back to heuristic."""
        clf = TemporalEmotionClassifier(
            model_path="/tmp/nonexistent_temporal_model.pt",
            seq_length=5,
        )
        assert clf.model_loaded is False

    def test_emotions_list(self, classifier):
        """EMOTIONS class constant has 6 entries."""
        assert len(classifier.EMOTIONS) == 6
        assert "happy" in classifier.EMOTIONS
        assert "neutral" in classifier.EMOTIONS


# ---------------------------------------------------------------------------
# Buffer management
# ---------------------------------------------------------------------------


class TestBuffer:
    def test_add_epoch_grows_buffer(self, classifier, make_features):
        """Adding epochs increases buffer length."""
        assert len(classifier.buffer) == 0
        classifier.add_epoch(make_features())
        assert len(classifier.buffer) == 1
        classifier.add_epoch(make_features())
        assert len(classifier.buffer) == 2

    def test_buffer_caps_at_seq_length(self, classifier, make_features):
        """Buffer never exceeds seq_length."""
        for _ in range(20):
            classifier.add_epoch(make_features())
        assert len(classifier.buffer) == classifier.seq_length

    def test_sliding_window_drops_oldest(self, classifier, make_features):
        """Oldest epoch is dropped when buffer overflows."""
        first = np.ones(41, dtype=np.float32) * 999.0
        classifier.add_epoch(first)
        for _ in range(classifier.seq_length):
            classifier.add_epoch(make_features())
        # first epoch should have been evicted
        assert not np.allclose(classifier.buffer[0], first)

    def test_add_epoch_truncates_long_features(self, classifier):
        """Features longer than input_dim are truncated."""
        long_feats = np.random.randn(100).astype(np.float32)
        classifier.add_epoch(long_feats)
        assert classifier.buffer[0].shape[0] == 41

    def test_add_epoch_pads_short_features(self, classifier):
        """Features shorter than input_dim are zero-padded."""
        short_feats = np.random.randn(10).astype(np.float32)
        classifier.add_epoch(short_feats)
        stored = classifier.buffer[0]
        assert stored.shape[0] == 41
        # First 10 should match, rest should be zero
        np.testing.assert_array_equal(stored[:10], short_feats)
        np.testing.assert_array_equal(stored[10:], np.zeros(31))

    def test_add_epoch_handles_2d_input(self, classifier):
        """2D feature arrays are flattened before buffering."""
        feats_2d = np.random.randn(1, 41).astype(np.float32)
        classifier.add_epoch(feats_2d)
        assert classifier.buffer[0].shape == (41,)

    def test_reset_clears_buffer(self, classifier, make_features):
        """reset() empties the buffer."""
        for _ in range(3):
            classifier.add_epoch(make_features())
        assert len(classifier.buffer) == 3
        classifier.reset()
        assert len(classifier.buffer) == 0


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


class TestIsReady:
    def test_not_ready_before_seq_length(self, classifier, make_features):
        """is_ready() returns False before enough epochs are buffered."""
        for _ in range(classifier.seq_length - 1):
            classifier.add_epoch(make_features())
        assert classifier.is_ready() is False

    def test_ready_at_seq_length(self, classifier, make_features):
        """is_ready() returns True once seq_length epochs are buffered."""
        for _ in range(classifier.seq_length):
            classifier.add_epoch(make_features())
        assert classifier.is_ready() is True

    def test_ready_after_overflow(self, classifier, make_features):
        """is_ready() stays True after buffer overflows."""
        for _ in range(classifier.seq_length + 5):
            classifier.add_epoch(make_features())
        assert classifier.is_ready() is True


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_returns_none_when_not_ready(self, classifier, make_features):
        """predict() returns None before buffer is full."""
        classifier.add_epoch(make_features())
        result = classifier.predict()
        assert result is None

    def test_heuristic_prediction_structure(self, classifier, make_features):
        """Heuristic prediction returns all required keys."""
        for _ in range(classifier.seq_length):
            classifier.add_epoch(make_features())

        result = classifier.predict()
        assert result is not None
        assert result["model_type"] == "temporal-heuristic"
        assert result["emotion"] == "neutral"
        assert result["epochs_used"] == classifier.seq_length
        assert "probabilities" in result
        assert "valence" in result
        assert "arousal" in result
        assert "temporal_features" in result

    def test_heuristic_probabilities_uniform(self, classifier, make_features):
        """Heuristic mode returns uniform probabilities."""
        for _ in range(classifier.seq_length):
            classifier.add_epoch(make_features())

        result = classifier.predict()
        probs = result["probabilities"]
        assert len(probs) == 6
        for emotion in classifier.EMOTIONS:
            assert emotion in probs
            assert abs(probs[emotion] - 1.0 / 6) < 1e-6

    def test_heuristic_valence_arousal_defaults(self, classifier, make_features):
        """Heuristic mode returns valence=0.0 and arousal=0.5."""
        for _ in range(classifier.seq_length):
            classifier.add_epoch(make_features())

        result = classifier.predict()
        assert result["valence"] == 0.0
        assert result["arousal"] == 0.5


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------


class TestTemporalFeatures:
    def test_temporal_features_empty_with_one_epoch(self, classifier, make_features):
        """_compute_temporal_features returns {} with fewer than 2 epochs."""
        classifier.add_epoch(make_features())
        feats = classifier._compute_temporal_features()
        assert feats == {}

    def test_temporal_features_keys(self, classifier, make_features):
        """Temporal features contain all expected keys."""
        for _ in range(classifier.seq_length):
            classifier.add_epoch(make_features())

        feats = classifier._compute_temporal_features()
        assert "mean_stability" in feats
        assert "trend_direction" in feats
        assert "variance" in feats
        assert "epochs_buffered" in feats

    def test_temporal_features_epochs_buffered(self, classifier, make_features):
        """epochs_buffered matches actual buffer length."""
        for i in range(3):
            classifier.add_epoch(make_features())

        feats = classifier._compute_temporal_features()
        assert feats["epochs_buffered"] == 3

    def test_stable_signal_has_low_variance(self):
        """Constant feature vectors produce near-zero variance."""
        clf = TemporalEmotionClassifier(seq_length=5, input_dim=10)
        constant = np.ones(10, dtype=np.float32) * 3.0
        for _ in range(5):
            clf.add_epoch(constant)

        feats = clf._compute_temporal_features()
        assert feats["variance"] < 1e-6
        assert feats["trend_direction"] == pytest.approx(0.0, abs=1e-6)

    def test_increasing_signal_positive_trend(self):
        """Monotonically increasing features produce positive trend."""
        clf = TemporalEmotionClassifier(seq_length=5, input_dim=10)
        for i in range(5):
            clf.add_epoch(np.ones(10, dtype=np.float32) * float(i))

        feats = clf._compute_temporal_features()
        assert feats["trend_direction"] > 0


# ---------------------------------------------------------------------------
# LSTM model path (only if torch is available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestLSTMModel:
    def test_model_forward_shape(self):
        """TemporalEEGNet forward pass produces correct output shapes."""
        from models.temporal_emotion_model import TemporalEEGNet
        import torch

        net = TemporalEEGNet(input_dim=41, hidden_dim=64, num_classes=6)
        x = torch.randn(2, 10, 41)  # batch=2, seq=10, features=41
        logits, valence, arousal = net(x)

        assert logits.shape == (2, 6)
        assert valence.shape == (2, 1)
        assert arousal.shape == (2, 1)

    def test_valence_range(self):
        """Valence output is bounded by tanh to [-1, 1]."""
        from models.temporal_emotion_model import TemporalEEGNet
        import torch

        net = TemporalEEGNet(input_dim=41)
        x = torch.randn(5, 10, 41)
        _, valence, _ = net(x)
        assert (valence >= -1.0).all()
        assert (valence <= 1.0).all()

    def test_arousal_range(self):
        """Arousal output is bounded by sigmoid to [0, 1]."""
        from models.temporal_emotion_model import TemporalEEGNet
        import torch

        net = TemporalEEGNet(input_dim=41)
        x = torch.randn(5, 10, 41)
        _, _, arousal = net(x)
        assert (arousal >= 0.0).all()
        assert (arousal <= 1.0).all()

    def test_classifier_with_saved_model(self, tmp_path):
        """Classifier loads a real saved model and produces LSTM predictions."""
        from models.temporal_emotion_model import TemporalEEGNet
        import torch

        # Save a randomly initialized model
        net = TemporalEEGNet(input_dim=10, hidden_dim=16, num_classes=6, num_layers=1)
        model_file = tmp_path / "temporal_test.pt"
        torch.save(net.state_dict(), model_file)

        # Load it via the classifier — architecture params must match saved model
        clf = TemporalEmotionClassifier(
            model_path=str(model_file),
            seq_length=3,
            input_dim=10,
            hidden_dim=16,
            num_layers=1,
        )
        assert clf.model_loaded is True

        for _ in range(3):
            clf.add_epoch(np.random.randn(10).astype(np.float32))

        result = clf.predict()
        assert result is not None
        assert result["model_type"] == "temporal-lstm"
        assert result["emotion"] in clf.EMOTIONS
        assert -1.0 <= result["valence"] <= 1.0
        assert 0.0 <= result["arousal"] <= 1.0
        # Probabilities should sum to ~1.0
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-4
