"""Tests for UserModelRetrainer — per-user model fine-tuning from corrections."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Ensure ml/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.retrain_from_user_data import (
    AUTO_RETRAIN_INITIAL,
    AUTO_RETRAIN_INCREMENT,
    MIN_CORRECTIONS_EEG,
    MIN_CORRECTIONS_VOICE,
    UserModelRetrainer,
)


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    """Redirect user_data and user_models to temp directories."""
    import training.retrain_from_user_data as mod

    corrections_dir = tmp_path / "corrections"
    corrections_dir.mkdir()
    models_dir = tmp_path / "user_models"
    models_dir.mkdir()

    monkeypatch.setattr(mod, "USER_DATA_DIR", corrections_dir)
    monkeypatch.setattr(mod, "USER_MODELS_DIR", models_dir)

    return tmp_path


def _write_corrections(tmp_path, user_id: str, corrections: list) -> None:
    """Write correction records to the JSONL file."""
    corrections_dir = tmp_path / "corrections"
    corrections_dir.mkdir(exist_ok=True)
    path = corrections_dir / f"{user_id}_corrections.jsonl"
    with open(path, "w") as f:
        for c in corrections:
            f.write(json.dumps(c) + "\n")


def _make_eeg_correction(emotion: str, idx: int) -> dict:
    """Create a correction record with 85-dim EEG features."""
    return {
        "corrected_emotion": emotion,
        "predicted_emotion": "neutral",
        "eeg_features": list(np.random.randn(85).astype(float)),
        "created_at": f"2026-03-23T{10 + idx}:00:00Z",
    }


def _make_voice_correction(emotion: str, idx: int) -> dict:
    """Create a correction record with voice features."""
    return {
        "corrected_emotion": emotion,
        "predicted_emotion": "neutral",
        "voice_features": list(np.random.randn(40).astype(float)),
        "created_at": f"2026-03-23T{10 + idx}:00:00Z",
    }


class TestLoadCorrections:
    def test_empty_user(self, tmp_dirs):
        """Returns empty list when no corrections exist."""
        retrainer = UserModelRetrainer("nonexistent")
        assert retrainer.load_corrections() == []

    def test_load_written_corrections(self, tmp_dirs):
        """Loads corrections from JSONL file."""
        corrections = [_make_eeg_correction("happy", i) for i in range(3)]
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        loaded = retrainer.load_corrections()
        assert len(loaded) == 3
        assert loaded[0]["corrected_emotion"] == "happy"


class TestShouldRetrain:
    def test_below_threshold(self, tmp_dirs):
        """No retrain when below AUTO_RETRAIN_INITIAL (5)."""
        corrections = [_make_eeg_correction("happy", i) for i in range(3)]
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        assert retrainer.should_retrain() is False

    def test_at_initial_threshold(self, tmp_dirs):
        """Triggers retrain at AUTO_RETRAIN_INITIAL corrections."""
        corrections = [
            _make_eeg_correction("happy" if i % 2 == 0 else "sad", i)
            for i in range(AUTO_RETRAIN_INITIAL)
        ]
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        assert retrainer.should_retrain() is True

    def test_no_retrain_after_recent(self, tmp_dirs):
        """No retrain when not enough new corrections since last retrain."""
        corrections = [
            _make_eeg_correction("happy" if i % 2 == 0 else "sad", i)
            for i in range(AUTO_RETRAIN_INITIAL + 3)
        ]
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        # Simulate a retrain having happened at AUTO_RETRAIN_INITIAL
        retrainer._update_retrain_meta(AUTO_RETRAIN_INITIAL)

        # Only 3 new — need AUTO_RETRAIN_INCREMENT (5)
        assert retrainer.should_retrain() is False


class TestRetrainEEG:
    def test_not_enough_corrections(self, tmp_dirs):
        """Returns not-trained when below MIN_CORRECTIONS_EEG."""
        corrections = [_make_eeg_correction("happy", i) for i in range(3)]
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        result = retrainer.retrain_eeg()
        assert result["trained"] is False
        assert "Need" in result["reason"]

    def test_successful_retrain(self, tmp_dirs):
        """Trains EEG model from enough corrections with >= 2 classes."""
        corrections = []
        for i in range(MIN_CORRECTIONS_EEG):
            emotion = "happy" if i % 2 == 0 else "sad"
            corrections.append(_make_eeg_correction(emotion, i))
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        result = retrainer.retrain_eeg()

        assert result["trained"] is True
        assert result["modality"] == "eeg"
        assert result["n_samples"] == MIN_CORRECTIONS_EEG
        assert "happy" in result["classes"]
        assert "sad" in result["classes"]
        assert 0.0 <= result["train_accuracy"] <= 1.0

        # Verify model files were saved
        user_dir = tmp_dirs / "user_models" / "user1"
        assert (user_dir / "eeg_classifier.pkl").exists()
        assert (user_dir / "eeg_scaler.pkl").exists()
        assert (user_dir / "eeg_meta.json").exists()

    def test_single_class_rejected(self, tmp_dirs):
        """Cannot train with only one emotion class."""
        corrections = [_make_eeg_correction("happy", i) for i in range(MIN_CORRECTIONS_EEG)]
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        result = retrainer.retrain_eeg()
        assert result["trained"] is False
        assert "2 emotion classes" in result["reason"]


class TestRetrainVoice:
    def test_successful_voice_retrain(self, tmp_dirs):
        """Trains voice model from enough corrections."""
        corrections = []
        for i in range(MIN_CORRECTIONS_VOICE):
            emotion = "happy" if i % 2 == 0 else "angry"
            corrections.append(_make_voice_correction(emotion, i))
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        result = retrainer.retrain_voice()

        assert result["trained"] is True
        assert result["modality"] == "voice"
        assert result["n_samples"] == MIN_CORRECTIONS_VOICE

        user_dir = tmp_dirs / "user_models" / "user1"
        assert (user_dir / "voice_classifier.pkl").exists()
        assert (user_dir / "voice_scaler.pkl").exists()


class TestRetrainAll:
    def test_retrain_both_modalities(self, tmp_dirs):
        """retrain_all trains both EEG and voice when data is available."""
        corrections = []
        for i in range(MIN_CORRECTIONS_EEG):
            c = _make_eeg_correction("happy" if i % 2 == 0 else "sad", i)
            c["voice_features"] = list(np.random.randn(40).astype(float))
            corrections.append(c)
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        result = retrainer.retrain_all()

        assert result["any_trained"] is True
        assert result["eeg"]["trained"] is True
        assert result["voice"]["trained"] is True


class TestModelLoading:
    def test_load_nonexistent_model(self, tmp_dirs):
        """Returns None when no model has been trained."""
        retrainer = UserModelRetrainer("nomodel")
        assert retrainer.load_eeg_model() is None
        assert retrainer.load_voice_model() is None

    def test_load_trained_model(self, tmp_dirs):
        """Can load a previously trained EEG model."""
        corrections = []
        for i in range(MIN_CORRECTIONS_EEG):
            corrections.append(_make_eeg_correction("happy" if i % 2 == 0 else "sad", i))
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        retrainer.retrain_eeg()

        # Load it back
        loaded = retrainer.load_eeg_model()
        assert loaded is not None
        model, scaler, meta = loaded
        assert model is not None
        assert scaler is not None
        assert "classes" in meta
        assert meta["n_samples"] == MIN_CORRECTIONS_EEG


class TestGetStatus:
    def test_status_empty_user(self, tmp_dirs):
        """Status for a user with no corrections."""
        retrainer = UserModelRetrainer("empty_user")
        status = retrainer.get_status()
        assert status["user_id"] == "empty_user"
        assert status["total_corrections"] == 0
        assert status["eeg_model_exists"] is False
        assert status["voice_model_exists"] is False
        assert status["should_retrain"] is False
