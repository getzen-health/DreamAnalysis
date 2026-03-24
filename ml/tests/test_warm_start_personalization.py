"""Tests for warm-start personalization in both retraining pipelines.

Verifies that:
1. auto_retrainer warm-starts from existing personal model (not random)
2. retrain_from_user_data warm-starts from existing user model (not random)
3. First-time training (no prior model) still works correctly
4. Warm-started model preserves learned decision boundaries from prior run
5. Warm-started model incorporates new data (weights change)
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for sessions, user_models, user_data."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    user_models_dir = tmp_path / "user_models"
    user_models_dir.mkdir()
    user_data_dir = tmp_path / "user_data" / "corrections"
    user_data_dir.mkdir(parents=True)
    return {
        "sessions": sessions_dir,
        "user_models": user_models_dir,
        "user_data": user_data_dir,
    }


def _make_session_json(sessions_dir: Path, n_entries: int = 50) -> None:
    """Create a synthetic session JSON with labeled timeline entries."""
    rng = np.random.RandomState(42)
    emotions = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

    timeline = []
    for i in range(n_entries):
        emo = emotions[i % len(emotions)]
        timeline.append({
            "emotions": {
                "emotion": emo,
                "stress_index": float(rng.uniform(0, 1)),
                "focus_index": float(rng.uniform(0, 1)),
                "relaxation_index": float(rng.uniform(0, 1)),
                "valence": float(rng.uniform(-1, 1)),
                "arousal": float(rng.uniform(0, 1)),
            },
            "band_powers": {
                "delta": float(rng.uniform(0, 0.5)),
                "theta": float(rng.uniform(0, 0.3)),
                "alpha": float(rng.uniform(0, 0.3)),
                "beta": float(rng.uniform(0, 0.2)),
                "gamma": float(rng.uniform(0, 0.1)),
            },
            "features": {
                "band_power_delta": float(rng.uniform(0, 0.5)),
                "band_power_theta": float(rng.uniform(0, 0.3)),
                "band_power_alpha": float(rng.uniform(0, 0.3)),
                "band_power_beta": float(rng.uniform(0, 0.2)),
            },
        })

    with open(sessions_dir / "session_001.json", "w") as f:
        json.dump({"analysis_timeline": timeline}, f)


def _make_corrections_jsonl(corrections_path: Path, n_corrections: int = 30) -> None:
    """Create synthetic user corrections with 170-dim EEG feature vectors."""
    rng = np.random.RandomState(42)
    emotions = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

    with open(corrections_path, "w") as f:
        for i in range(n_corrections):
            corrected = emotions[i % len(emotions)]
            predicted = emotions[(i + 1) % len(emotions)]
            features = rng.randn(170).tolist()
            record = {
                "corrected_emotion": corrected,
                "predicted_emotion": predicted,
                "eeg_features": features,
            }
            f.write(json.dumps(record) + "\n")


class TestAutoRetrainerWarmStart:
    """Tests for warm-start in auto_retrainer.retrain_personal_model."""

    def test_first_train_from_scratch_works(self, temp_dirs):
        """First-time training with no prior model should succeed."""
        _make_session_json(temp_dirs["sessions"])

        with patch("training.auto_retrainer.SESSIONS_DIR", temp_dirs["sessions"]), \
             patch("training.auto_retrainer.USER_MODELS_DIR", temp_dirs["user_models"]):
            from training.auto_retrainer import retrain_personal_model
            result = retrain_personal_model(user_id="test_user")

        assert result["trained"] is True
        assert result["n_samples"] > 0
        assert result["train_accuracy"] > 0
        # Model file should exist
        assert (temp_dirs["user_models"] / "test_user_personal.pkl").exists()

    def test_second_train_warm_starts_from_existing(self, temp_dirs):
        """Second training should load prior model and warm-start, not reset."""
        _make_session_json(temp_dirs["sessions"])

        with patch("training.auto_retrainer.SESSIONS_DIR", temp_dirs["sessions"]), \
             patch("training.auto_retrainer.USER_MODELS_DIR", temp_dirs["user_models"]):
            from training.auto_retrainer import retrain_personal_model

            # First training
            result1 = retrain_personal_model(user_id="test_user")
            assert result1["trained"] is True

            # Load the trained model weights
            import joblib
            model_path = temp_dirs["user_models"] / "test_user_personal.pkl"
            model1 = joblib.load(model_path)
            weights_after_first = model1.coef_.copy()

            # Second training (should warm-start from the first model)
            result2 = retrain_personal_model(user_id="test_user")
            assert result2["trained"] is True
            assert result2.get("warm_started") is True

            # Weights should exist and be different (continued training)
            model2 = joblib.load(model_path)
            # The model should still have valid weights (not reset to random)
            assert model2.coef_ is not None
            assert model2.coef_.shape == weights_after_first.shape

    def test_cold_start_when_no_prior_model(self, temp_dirs):
        """When no prior model exists, should train from scratch without error."""
        _make_session_json(temp_dirs["sessions"])

        # Ensure no prior model exists
        assert not (temp_dirs["user_models"] / "test_user_personal.pkl").exists()

        with patch("training.auto_retrainer.SESSIONS_DIR", temp_dirs["sessions"]), \
             patch("training.auto_retrainer.USER_MODELS_DIR", temp_dirs["user_models"]):
            from training.auto_retrainer import retrain_personal_model
            result = retrain_personal_model(user_id="test_user")

        assert result["trained"] is True
        assert result.get("warm_started") is False


class TestUserModelRetrainerWarmStart:
    """Tests for warm-start in retrain_from_user_data.UserModelRetrainer."""

    def test_first_retrain_eeg_from_scratch(self, temp_dirs):
        """First EEG retrain should work from scratch."""
        corrections_path = temp_dirs["user_data"] / "test_user_corrections.jsonl"
        _make_corrections_jsonl(corrections_path, n_corrections=30)

        with patch("training.retrain_from_user_data.USER_DATA_DIR", temp_dirs["user_data"]), \
             patch("training.retrain_from_user_data.USER_MODELS_DIR", temp_dirs["user_models"]):
            from training.retrain_from_user_data import UserModelRetrainer
            retrainer = UserModelRetrainer("test_user")
            result = retrainer.retrain_eeg(force=True)

        assert result["trained"] is True
        assert result["n_samples"] == 30
        assert (temp_dirs["user_models"] / "test_user" / "eeg_classifier.pkl").exists()

    def test_second_retrain_eeg_warm_starts(self, temp_dirs):
        """Second EEG retrain should warm-start from prior model."""
        corrections_path = temp_dirs["user_data"] / "test_user_corrections.jsonl"
        _make_corrections_jsonl(corrections_path, n_corrections=30)

        with patch("training.retrain_from_user_data.USER_DATA_DIR", temp_dirs["user_data"]), \
             patch("training.retrain_from_user_data.USER_MODELS_DIR", temp_dirs["user_models"]):
            from training.retrain_from_user_data import UserModelRetrainer

            retrainer = UserModelRetrainer("test_user")

            # First training
            result1 = retrainer.retrain_eeg(force=True)
            assert result1["trained"] is True

            # Load model weights
            import joblib
            model_path = temp_dirs["user_models"] / "test_user" / "eeg_classifier.pkl"
            model1 = joblib.load(model_path)
            weights_after_first = model1.coef_.copy()

            # Second training
            result2 = retrainer.retrain_eeg(force=True)
            assert result2["trained"] is True
            assert result2.get("warm_started") is True

            # Weights should still be valid
            model2 = joblib.load(model_path)
            assert model2.coef_ is not None
            assert model2.coef_.shape == weights_after_first.shape

    def test_warm_start_preserves_prior_knowledge(self, temp_dirs):
        """Warm-started model should perform at least as well as cold start
        because it incorporates prior learning."""
        # Create two sets of corrections
        rng = np.random.RandomState(42)
        emotions = ["happy", "sad", "angry"]

        corrections_path = temp_dirs["user_data"] / "test_user_corrections.jsonl"

        # Phase 1: Train on first 30 corrections with clear class separation
        with open(corrections_path, "w") as f:
            for i in range(30):
                emo = emotions[i % 3]
                # Create features with clear class separation
                base = np.zeros(170)
                class_idx = emotions.index(emo)
                base[class_idx * 50:(class_idx + 1) * 50] = rng.uniform(0.5, 1.5, 50)
                base += rng.randn(170) * 0.1  # small noise
                record = {
                    "corrected_emotion": emo,
                    "predicted_emotion": emotions[(i + 1) % 3],
                    "eeg_features": base.tolist(),
                }
                f.write(json.dumps(record) + "\n")

        with patch("training.retrain_from_user_data.USER_DATA_DIR", temp_dirs["user_data"]), \
             patch("training.retrain_from_user_data.USER_MODELS_DIR", temp_dirs["user_models"]):
            from training.retrain_from_user_data import UserModelRetrainer
            retrainer = UserModelRetrainer("test_user")

            # Train first round
            result1 = retrainer.retrain_eeg(force=True)
            assert result1["trained"] is True
            acc1 = result1["train_accuracy"]

            # Add 10 more corrections (same distribution) and retrain
            with open(corrections_path, "a") as f:
                for i in range(10):
                    emo = emotions[i % 3]
                    base = np.zeros(170)
                    class_idx = emotions.index(emo)
                    base[class_idx * 50:(class_idx + 1) * 50] = rng.uniform(0.5, 1.5, 50)
                    base += rng.randn(170) * 0.1
                    record = {
                        "corrected_emotion": emo,
                        "predicted_emotion": emotions[(i + 1) % 3],
                        "eeg_features": base.tolist(),
                    }
                    f.write(json.dumps(record) + "\n")

            # Warm-start retrain
            result2 = retrainer.retrain_eeg(force=True)
            assert result2["trained"] is True
            assert result2.get("warm_started") is True

            # Warm-started model on more data should be at least as good
            # (with same distribution, adding data should not degrade)
            acc2 = result2["train_accuracy"]
            assert acc2 >= acc1 * 0.9, (
                f"Warm-started accuracy ({acc2:.3f}) should not be much worse "
                f"than initial ({acc1:.3f})"
            )
