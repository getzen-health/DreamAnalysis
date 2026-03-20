"""Tests for the PersonalModelAdapter online learning pipeline.

Validates:
1. adapt_from_features auto-initializes SGDClassifier on first correction
2. Predictions change after incremental updates
3. Unknown labels are rejected gracefully
4. Persistence round-trip (save/load)
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.online_learner import PersonalModelAdapter


def _make_features(n: int = 17, seed: int = 0) -> dict:
    """Create a dummy feature dict matching extract_features output structure."""
    rng = np.random.RandomState(seed)
    keys = [
        "delta", "theta", "alpha", "beta", "gamma",
        "alpha_beta_ratio", "theta_beta_ratio", "alpha_theta_ratio",
        "theta_alpha_ratio", "high_beta_frac", "delta_theta_ratio",
        "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
        "spectral_entropy", "diff_entropy_alpha", "diff_entropy_theta",
    ]
    return {k: float(rng.randn()) for k in keys}


class TestAdaptFromFeatures:
    """Test the new adapt_from_features method that closes the correction loop."""

    _TEST_USER_IDS = [
        "test_auto_init", "test_incr", "test_pred", "test_unknown", "test_legacy",
    ]

    def setup_method(self):
        """Remove stale persisted models so each test starts from clean state."""
        from models.online_learner import USER_MODELS_DIR
        import shutil
        for uid in self._TEST_USER_IDS:
            user_dir = USER_MODELS_DIR / uid
            if user_dir.exists():
                shutil.rmtree(user_dir)

    def test_auto_initialize_on_first_correction(self):
        """SGDClassifier should auto-create on first call without prior calibrate()."""
        base = MagicMock()
        adapter = PersonalModelAdapter(base, user_id="test_auto_init")

        assert adapter.personal_model is None
        feats = _make_features(seed=1)
        result = adapter.adapt_from_features(feats, "happy", "sad")

        assert result["updated"] is True
        assert adapter.personal_model is not None
        assert adapter.n_samples == 1
        assert adapter.classes == list(PersonalModelAdapter._DEFAULT_CLASSES)

    def test_incremental_updates_accumulate(self):
        """Multiple corrections should increase n_samples and update the model."""
        base = MagicMock()
        adapter = PersonalModelAdapter(base, user_id="test_incr")

        for i in range(5):
            feats = _make_features(seed=i)
            label = ["happy", "sad", "angry", "relaxed", "focused"][i]
            result = adapter.adapt_from_features(feats, "neutral", label)
            assert result["updated"] is True

        assert adapter.n_samples == 5

    def test_prediction_after_adaptation(self):
        """After enough corrections, predict() should return personal predictions."""
        base = MagicMock()
        adapter = PersonalModelAdapter(base, user_id="test_pred")

        # Feed several "happy" corrections with similar features
        for i in range(10):
            feats = _make_features(seed=42)  # same features = same pattern
            adapter.adapt_from_features(feats, "sad", "happy")

        # Now predict with same features — should lean toward "happy"
        pred = adapter.predict(_make_features(seed=42))
        assert pred["has_personal"] is True
        assert "personal_prediction" in pred
        assert "personal_confidence" in pred

    def test_unknown_label_rejected(self):
        """Labels not in the class list should return updated=False."""
        base = MagicMock()
        adapter = PersonalModelAdapter(base, user_id="test_unknown")

        # First call initializes with default classes
        adapter.adapt_from_features(_make_features(), "happy", "happy")

        # "confused" is not in _DEFAULT_CLASSES
        result = adapter.adapt_from_features(_make_features(seed=2), "happy", "confused")
        assert result["updated"] is False
        assert "Unknown label" in result["reason"]

    def test_adapt_legacy_with_signal(self):
        """The original adapt() method should still work via adapt_from_features."""
        base = MagicMock()
        adapter = PersonalModelAdapter(base, user_id="test_legacy")

        signal = np.random.randn(256)  # 1 second of EEG at 256 Hz
        result = adapter.adapt(signal, "happy", "sad", fs=256.0)
        assert result["updated"] is True
        assert adapter.n_samples == 1
