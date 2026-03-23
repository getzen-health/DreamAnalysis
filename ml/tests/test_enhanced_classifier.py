"""Tests for the enhanced emotion classifier combining SSL + hand-crafted features.

Verifies:
1. EnhancedEmotionClassifier forward pass produces (batch, 6) output
2. With pretrained encoder frozen, only classifier params have gradients
3. ONNX export succeeds
4. ONNX inference matches PyTorch (within tolerance)
5. predict_emotion returns correct format
6. Save/load roundtrip preserves predictions
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")

from models.eeg_emotion_enhanced import EMOTIONS, EnhancedEmotionClassifier
from models.eeg_ssl import EEGContrastiveEncoder
from processing.emotion_features_enhanced import ENHANCED_FEATURE_DIM


@pytest.fixture
def encoder():
    """Fresh (untrained) SSL encoder."""
    return EEGContrastiveEncoder(n_channels=4, n_samples=1024, embed_dim=64, proj_dim=32)


@pytest.fixture
def model(encoder):
    """Enhanced classifier with frozen encoder."""
    return EnhancedEmotionClassifier(
        n_classes=6,
        pretrained_encoder=encoder,
        include_temporal=True,
        freeze_encoder=True,
    )


@pytest.fixture
def model_no_temporal(encoder):
    """Enhanced classifier without temporal features."""
    return EnhancedEmotionClassifier(
        n_classes=6,
        pretrained_encoder=encoder,
        include_temporal=False,
        freeze_encoder=True,
    )


@pytest.fixture
def batch_data():
    """Batch of 4 samples with raw EEG + hand-crafted features (with temporal)."""
    raw_eeg = torch.randn(4, 4, 1024)
    features = torch.randn(4, ENHANCED_FEATURE_DIM * 2)  # 106 dims (53 + 53 temporal)
    return raw_eeg, features


@pytest.fixture
def batch_data_no_temporal():
    """Batch with features but no temporal deltas."""
    raw_eeg = torch.randn(4, 4, 1024)
    features = torch.randn(4, ENHANCED_FEATURE_DIM)  # 53 dims only
    return raw_eeg, features


class TestEnhancedEmotionClassifier:
    """Tests for the main classifier."""

    def test_forward_output_shape(self, model, batch_data):
        """Forward pass should produce (batch, n_classes) logits."""
        raw_eeg, features = batch_data
        logits = model(raw_eeg, features)
        assert logits.shape == (4, 6)

    def test_forward_no_temporal(self, model_no_temporal, batch_data_no_temporal):
        """Model without temporal features should accept 53-dim input."""
        raw_eeg, features = batch_data_no_temporal
        logits = model_no_temporal(raw_eeg, features)
        assert logits.shape == (4, 6)

    def test_output_is_finite(self, model, batch_data):
        """All outputs must be finite."""
        logits = model(*batch_data)
        assert torch.all(torch.isfinite(logits))

    def test_frozen_encoder_no_gradients(self, model, batch_data):
        """When encoder is frozen, encoder params should not have gradients."""
        logits = model(*batch_data)
        loss = logits.sum()
        loss.backward()

        # Encoder params should have no gradient
        for name, param in model.encoder.named_parameters():
            assert param.grad is None or torch.all(param.grad == 0), (
                f"Encoder param '{name}' has non-zero gradient despite being frozen"
            )

        # Classifier params should have gradients
        has_grad = False
        for param in model.classifier.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_grad = True
                break
        assert has_grad, "Classifier head has no gradients"

    def test_unfrozen_encoder_has_gradients(self, encoder):
        """With freeze_encoder=False, encoder should have gradients."""
        model = EnhancedEmotionClassifier(
            n_classes=6,
            pretrained_encoder=encoder,
            include_temporal=True,
            freeze_encoder=False,
        )
        raw_eeg = torch.randn(2, 4, 1024)
        features = torch.randn(2, ENHANCED_FEATURE_DIM * 2)

        logits = model(raw_eeg, features)
        loss = logits.sum()
        loss.backward()

        encoder_has_grad = any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in model.encoder.parameters()
        )
        assert encoder_has_grad, "Encoder should have gradients when not frozen"

    def test_single_sample(self, model):
        """Should work with batch_size=1 in eval mode (inference).

        Note: BatchNorm1d requires batch_size > 1 in training mode.
        Single-sample inference is the real-world use case (one EEG epoch at a time).
        """
        model.eval()
        raw_eeg = torch.randn(1, 4, 1024)
        features = torch.randn(1, ENHANCED_FEATURE_DIM * 2)
        with torch.no_grad():
            logits = model(raw_eeg, features)
        assert logits.shape == (1, 6)

    def test_predict_proba_shape(self, model, batch_data):
        """predict_proba should return (batch, n_classes) numpy array summing to 1."""
        probs = model.predict_proba(*batch_data)
        assert probs.shape == (4, 6)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0)

    def test_predict_emotion_format(self, model):
        """predict_emotion should return correctly structured dict."""
        raw_eeg = torch.randn(1, 4, 1024)
        features = torch.randn(1, ENHANCED_FEATURE_DIM * 2)
        result = model.predict_emotion(raw_eeg, features)

        assert "emotion" in result
        assert result["emotion"] in EMOTIONS
        assert "probabilities" in result
        assert set(result["probabilities"].keys()) == set(EMOTIONS)
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
        assert result["model_type"] == "enhanced_ssl_handcrafted"


class TestONNXExport:
    """Tests for ONNX export and inference."""

    def test_onnx_export_succeeds(self, model):
        """ONNX export should produce a valid .onnx file."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            model.export_onnx(f.name)
            assert Path(f.name).stat().st_size > 0
            Path(f.name).unlink()

    def test_onnx_export_no_temporal(self, model_no_temporal):
        """ONNX export should work for model without temporal features."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            model_no_temporal.export_onnx(f.name)
            assert Path(f.name).stat().st_size > 0
            Path(f.name).unlink()

    def test_onnx_inference_matches_pytorch(self, model):
        """ONNX inference should match PyTorch within floating-point tolerance."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        raw_eeg = torch.randn(1, 4, 1024)
        features = torch.randn(1, ENHANCED_FEATURE_DIM * 2)

        # PyTorch prediction
        model.eval()
        with torch.no_grad():
            pt_logits = model(raw_eeg, features).numpy()

        # ONNX prediction
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            model.export_onnx(f.name)
            sess = ort.InferenceSession(f.name)
            onnx_logits = sess.run(
                None,
                {
                    "raw_eeg": raw_eeg.numpy(),
                    "handcrafted_features": features.numpy(),
                },
            )[0]
            Path(f.name).unlink()

        np.testing.assert_allclose(pt_logits, onnx_logits, atol=1e-4, rtol=1e-4)


class TestSerialization:
    """Tests for save/load."""

    def test_save_load_roundtrip(self, model, batch_data):
        """Save and load should preserve predictions."""
        model.eval()
        with torch.no_grad():
            logits_before = model(*batch_data)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            model.save(f.name)
            loaded = EnhancedEmotionClassifier.load(f.name)

        loaded.eval()
        with torch.no_grad():
            logits_after = loaded(*batch_data)

        torch.testing.assert_close(logits_before, logits_after)
        Path(f.name).unlink()

    def test_loaded_model_config(self, model):
        """Loaded model should have correct config."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            model.save(f.name)
            loaded = EnhancedEmotionClassifier.load(f.name)

        assert loaded.n_classes == model.n_classes
        assert loaded.include_temporal == model.include_temporal
        Path(f.name).unlink()
