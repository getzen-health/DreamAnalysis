"""Tests for per-user ONNX export and model serving endpoints."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Ensure ml/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.retrain_from_user_data import (
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


def _train_eeg_model(tmp_path, user_id: str = "user1") -> UserModelRetrainer:
    """Helper: train an EEG model so we can test export."""
    corrections = []
    for i in range(MIN_CORRECTIONS_EEG):
        emotion = "happy" if i % 2 == 0 else "sad"
        corrections.append(_make_eeg_correction(emotion, i))
    _write_corrections(tmp_path, user_id, corrections)

    retrainer = UserModelRetrainer(user_id)
    result = retrainer.retrain_eeg()
    assert result["trained"] is True
    return retrainer


def _train_voice_model(tmp_path, user_id: str = "user1") -> UserModelRetrainer:
    """Helper: train a voice model so we can test export."""
    corrections = []
    for i in range(MIN_CORRECTIONS_VOICE):
        emotion = "happy" if i % 2 == 0 else "angry"
        corrections.append(_make_voice_correction(emotion, i))
    _write_corrections(tmp_path, user_id, corrections)

    retrainer = UserModelRetrainer(user_id)
    result = retrainer.retrain_voice()
    assert result["trained"] is True
    return retrainer


class TestEEGOnnxExport:
    def test_export_creates_valid_onnx_file(self, tmp_dirs):
        """_export_eeg_onnx creates a valid ONNX file after training."""
        retrainer = _train_eeg_model(tmp_dirs)
        onnx_path = retrainer.export_eeg_onnx()

        assert onnx_path is not None
        assert Path(onnx_path).exists()
        assert Path(onnx_path).stat().st_size > 0
        assert Path(onnx_path).name == "eeg_emotion_user.onnx"

    def test_onnx_loadable_by_onnxruntime(self, tmp_dirs):
        """Exported ONNX file can be loaded by onnxruntime."""
        onnxruntime = pytest.importorskip("onnxruntime")

        retrainer = _train_eeg_model(tmp_dirs)
        onnx_path = retrainer.export_eeg_onnx()
        assert onnx_path is not None

        session = onnxruntime.InferenceSession(onnx_path)
        assert session is not None
        assert len(session.get_inputs()) > 0
        assert len(session.get_outputs()) > 0

    def test_onnx_inference_correct_output_shape(self, tmp_dirs):
        """ONNX inference produces output with correct shape (batch, n_classes)."""
        onnxruntime = pytest.importorskip("onnxruntime")

        retrainer = _train_eeg_model(tmp_dirs)
        onnx_path = retrainer.export_eeg_onnx()
        assert onnx_path is not None

        session = onnxruntime.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        # Run inference with a batch of 3 samples, 170-dim features (padded from 85)
        test_input = np.random.randn(3, 170).astype(np.float32)
        outputs = session.run(None, {input_name: test_input})

        # skl2onnx produces 2 outputs: class labels (int64) + probabilities (map or array)
        # The first output is predicted class labels, shape (batch,)
        assert len(outputs) >= 1
        labels = outputs[0]
        assert labels.shape[0] == 3  # batch size
        # Each label should be a valid class index (0 or 1 for happy/sad)
        assert all(l in (0, 1) for l in labels)

    def test_export_returns_none_without_model(self, tmp_dirs):
        """Returns None when no trained model exists."""
        retrainer = UserModelRetrainer("no_model_user")
        result = retrainer.export_eeg_onnx()
        assert result is None

    def test_retrain_auto_exports_onnx(self, tmp_dirs):
        """retrain_eeg() automatically exports ONNX and reports it."""
        corrections = []
        for i in range(MIN_CORRECTIONS_EEG):
            emotion = "happy" if i % 2 == 0 else "sad"
            corrections.append(_make_eeg_correction(emotion, i))
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        result = retrainer.retrain_eeg()

        assert result["trained"] is True
        assert result["onnx_exported"] is True

        onnx_path = tmp_dirs / "user_models" / "user1" / "eeg_emotion_user.onnx"
        assert onnx_path.exists()


class TestVoiceOnnxExport:
    def test_voice_export_creates_onnx(self, tmp_dirs):
        """Voice ONNX export creates a valid file."""
        retrainer = _train_voice_model(tmp_dirs)
        onnx_path = retrainer.export_voice_onnx()

        assert onnx_path is not None
        assert Path(onnx_path).exists()
        assert Path(onnx_path).name == "voice_emotion_user.onnx"

    def test_voice_retrain_auto_exports(self, tmp_dirs):
        """retrain_voice() automatically exports ONNX."""
        corrections = []
        for i in range(MIN_CORRECTIONS_VOICE):
            emotion = "happy" if i % 2 == 0 else "angry"
            corrections.append(_make_voice_correction(emotion, i))
        _write_corrections(tmp_dirs, "user1", corrections)

        retrainer = UserModelRetrainer("user1")
        result = retrainer.retrain_voice()

        assert result["trained"] is True
        assert result["onnx_exported"] is True


class TestVersionEndpoint:
    def test_version_no_models(self, tmp_dirs, monkeypatch):
        """Version endpoint returns not-available when no models exist."""
        import asyncio
        import api.routes.training_sync as sync_mod
        monkeypatch.setattr(sync_mod, "USER_MODELS_DIR", tmp_dirs / "user_models")

        from api.routes.training_sync import get_model_version
        result = asyncio.get_event_loop().run_until_complete(
            get_model_version("nonexistent_user")
        )

        assert result["eeg_available"] is False
        assert result["voice_available"] is False
        assert result["eeg_updated"] is None
        assert result["voice_updated"] is None

    def test_version_with_eeg_model(self, tmp_dirs, monkeypatch):
        """Version endpoint returns available when EEG ONNX exists."""
        import asyncio
        import api.routes.training_sync as sync_mod
        monkeypatch.setattr(sync_mod, "USER_MODELS_DIR", tmp_dirs / "user_models")

        # Train and export
        _train_eeg_model(tmp_dirs)
        retrainer = UserModelRetrainer("user1")
        retrainer.export_eeg_onnx()

        from api.routes.training_sync import get_model_version
        result = asyncio.get_event_loop().run_until_complete(
            get_model_version("user1")
        )

        assert result["eeg_available"] is True
        assert result["eeg_updated"] is not None
        assert isinstance(result["eeg_updated"], float)

    def test_model_download_returns_file(self, tmp_dirs, monkeypatch):
        """Download endpoint returns FileResponse for existing model."""
        import asyncio
        import api.routes.training_sync as sync_mod
        monkeypatch.setattr(sync_mod, "USER_MODELS_DIR", tmp_dirs / "user_models")

        _train_eeg_model(tmp_dirs)
        retrainer = UserModelRetrainer("user1")
        retrainer.export_eeg_onnx()

        from api.routes.training_sync import get_user_eeg_onnx
        result = asyncio.get_event_loop().run_until_complete(
            get_user_eeg_onnx("user1")
        )

        # FileResponse is returned for existing models
        from fastapi.responses import FileResponse as FR
        assert isinstance(result, FR)

    def test_model_download_missing_returns_json(self, tmp_dirs, monkeypatch):
        """Download endpoint returns JSON error when no model exists."""
        import asyncio
        import api.routes.training_sync as sync_mod
        monkeypatch.setattr(sync_mod, "USER_MODELS_DIR", tmp_dirs / "user_models")

        from api.routes.training_sync import get_user_eeg_onnx
        result = asyncio.get_event_loop().run_until_complete(
            get_user_eeg_onnx("no_model_user")
        )

        assert isinstance(result, dict)
        assert result["status"] == "generic"
