"""Tests for TSception wiring into EmotionClassifier fallback chain."""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add ml/ to path so imports work (mirrors conftest.py pattern)
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTSceptionWiring:
    """Verify TSception is properly wired into EmotionClassifier."""

    def setup_method(self):
        from models.emotion_classifier import EmotionClassifier
        self.clf = EmotionClassifier()

    # ── Test 1: attribute exists after init ───────────────────────────────────

    def test_tsception_attribute_initialized(self):
        """EmotionClassifier must expose _tsception after __init__ (None or loaded)."""
        assert hasattr(self.clf, "_tsception"), (
            "_tsception attribute missing from EmotionClassifier.__init__"
        )

    # ── Test 2: long epoch falls through without crashing ────────────────────

    def test_long_epoch_no_crash_with_tsception_disabled(self):
        """With mega_lgbm disabled and epoch >= 1024 samples, predict() must not raise.

        Either tsception or feature heuristics handles it — any valid result is fine.
        """
        # Disable all trained models so path reaches tsception / heuristics
        self.clf.mega_lgbm_model = None
        self.clf.lgbm_muse_model = None
        self.clf.sklearn_model = None
        self.clf.onnx_session = None
        self.clf.model_type = "feature-based"
        self.clf._reve_foundation = None
        self.clf._reve = None
        self.clf._eegnet = None

        eeg = np.random.randn(4, 1024) * 20  # 4 channels, 4 sec @ 256 Hz
        result = self.clf.predict(eeg, fs=256.0)

        assert isinstance(result, dict), "predict() must return a dict"
        assert "emotion" in result, "result must contain 'emotion' key"
        assert "model_type" in result, "result must contain 'model_type' key"

    def test_long_epoch_with_mock_tsception_uses_tsception(self):
        """When _tsception is available and epoch >= 1024, predict() uses it."""
        # Disable all higher-priority models
        self.clf.mega_lgbm_model = None
        self.clf.lgbm_muse_model = None
        self.clf.sklearn_model = None
        self.clf.onnx_session = None
        self.clf.model_type = "feature-based"
        self.clf._reve_foundation = None
        self.clf._reve = None
        self.clf._eegnet = None

        # Install a mock TSception that returns a known result
        mock_tsception = MagicMock()
        mock_tsception.predict.return_value = {
            "emotion": "neutral",
            "emotion_index": 1,
            "probabilities": {"negative": 0.1, "neutral": 0.6, "positive": 0.3},
            "valence": 0.2,
            "confidence": 0.6,
            "model_type": "tsception",
        }
        self.clf._tsception = mock_tsception

        eeg = np.random.randn(4, 1024) * 20
        result = self.clf.predict(eeg, fs=256.0)

        assert result["model_type"] == "tsception", (
            f"Expected model_type='tsception', got '{result['model_type']}'"
        )
        mock_tsception.predict.assert_called_once()

    # ── Test 3: short epoch skips TSception ───────────────────────────────────

    def test_short_epoch_skips_tsception(self):
        """Epoch < 1024 samples must NOT produce model_type='tsception'.

        TSception needs >= 4 sec (1024 samples @ 256 Hz) for valid temporal conv.
        A short epoch should fall through to feature heuristics instead.
        """
        # Disable all higher-priority models
        self.clf.mega_lgbm_model = None
        self.clf.lgbm_muse_model = None
        self.clf.sklearn_model = None
        self.clf.onnx_session = None
        self.clf.model_type = "feature-based"
        self.clf._reve_foundation = None
        self.clf._reve = None
        self.clf._eegnet = None

        # Install mock TSception so we can confirm it's NOT called
        mock_tsception = MagicMock()
        mock_tsception.predict.return_value = {
            "emotion": "neutral",
            "emotion_index": 1,
            "probabilities": {"negative": 0.1, "neutral": 0.6, "positive": 0.3},
            "valence": 0.0,
            "confidence": 0.6,
            "model_type": "tsception",
        }
        self.clf._tsception = mock_tsception

        # Short epoch: 512 samples = 2 sec @ 256 Hz (below 1024 threshold)
        eeg = np.random.randn(4, 512) * 20
        result = self.clf.predict(eeg, fs=256.0)

        assert result.get("model_type") != "tsception", (
            "TSception must not activate on epochs shorter than 1024 samples"
        )
        mock_tsception.predict.assert_not_called()
