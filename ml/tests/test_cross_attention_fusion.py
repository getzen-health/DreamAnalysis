"""Tests for cross-modal attention fusion (EEG + Voice).

Tests the CrossModalFusion PyTorch module and its ONNX export.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# 1. Forward pass produces valid output shape (batch, 6)
# ---------------------------------------------------------------------------

class TestCrossModalFusionForward:
    def test_single_sample_output_shape(self):
        """Forward pass with batch_size=1 produces (1, 6) logits."""
        import torch
        from models.cross_attention_fusion import CrossModalFusion

        model = CrossModalFusion()
        model.eval()
        eeg_probs = torch.randn(1, 6)
        voice_probs = torch.randn(1, 5)
        out = model(eeg_probs, voice_probs)
        assert out.shape == (1, 6), f"Expected (1, 6), got {out.shape}"

    def test_batch_output_shape(self):
        """Forward pass with batch_size=8 produces (8, 6) logits."""
        import torch
        from models.cross_attention_fusion import CrossModalFusion

        model = CrossModalFusion()
        model.eval()
        eeg_probs = torch.randn(8, 6)
        voice_probs = torch.randn(8, 5)
        out = model(eeg_probs, voice_probs)
        assert out.shape == (8, 6), f"Expected (8, 6), got {out.shape}"

    def test_custom_dimensions(self):
        """Module works with custom eeg_dim, voice_dim, hidden_dim."""
        import torch
        from models.cross_attention_fusion import CrossModalFusion

        model = CrossModalFusion(eeg_dim=8, voice_dim=4, hidden_dim=16, n_heads=2, n_classes=4)
        model.eval()
        eeg = torch.randn(4, 8)
        voice = torch.randn(4, 4)
        out = model(eeg, voice)
        assert out.shape == (4, 4)


# ---------------------------------------------------------------------------
# 2. Output sums approximately to 1 after softmax
# ---------------------------------------------------------------------------

class TestSoftmaxNormalization:
    def test_softmax_sums_to_one(self):
        """After softmax, output probabilities sum to ~1."""
        import torch
        from models.cross_attention_fusion import CrossModalFusion

        model = CrossModalFusion()
        model.eval()
        eeg = torch.rand(4, 6)
        voice = torch.rand(4, 5)
        logits = model(eeg, voice)
        probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        for i in range(4):
            assert abs(sums[i].item() - 1.0) < 1e-5, (
                f"Sample {i} probabilities sum to {sums[i].item()}, expected ~1.0"
            )


# ---------------------------------------------------------------------------
# 3. When EEG is very confident, output follows EEG
# ---------------------------------------------------------------------------

class TestConfidenceFollowing:
    def test_follows_confident_eeg(self):
        """When EEG is peaked on class 0 and voice is uniform, output favors class 0."""
        import torch
        from models.cross_attention_fusion import CrossModalFusion

        torch.manual_seed(42)
        model = CrossModalFusion()
        # Train briefly so the model learns to pass through confident inputs
        model = _train_confidence_following_model(model, n_epochs=100)
        model.eval()

        # EEG: very confident on class 0 (happy)
        eeg = torch.tensor([[0.90, 0.02, 0.02, 0.02, 0.02, 0.02]])
        # Voice: uniform (uninformative)
        voice = torch.tensor([[0.20, 0.20, 0.20, 0.20, 0.20]])
        logits = model(eeg, voice)
        probs = torch.softmax(logits, dim=-1)
        # Output should lean toward class 0
        assert probs[0, 0].item() > probs[0].mean().item(), (
            "Output should favor class 0 when EEG is confident on class 0"
        )

    def test_follows_confident_voice(self):
        """When voice is peaked on class 2 and EEG is uniform, output favors class 2."""
        import torch
        from models.cross_attention_fusion import CrossModalFusion

        torch.manual_seed(42)
        model = CrossModalFusion()
        model = _train_confidence_following_model(model, n_epochs=100)
        model.eval()

        # EEG: uniform (uninformative)
        eeg = torch.tensor([[1/6] * 6])
        # Voice: very confident on class 2 (angry) -- pad to 5 dims
        voice = torch.tensor([[0.05, 0.05, 0.80, 0.05, 0.05]])
        logits = model(eeg, voice)
        probs = torch.softmax(logits, dim=-1)
        # Output should lean toward class 2
        assert probs[0, 2].item() > probs[0].mean().item(), (
            "Output should favor class 2 when voice is confident on class 2"
        )


# ---------------------------------------------------------------------------
# 4. ONNX export succeeds
# ---------------------------------------------------------------------------

class TestOnnxExport:
    def test_onnx_export_creates_file(self, tmp_path):
        """ONNX export produces a valid file."""
        import torch
        from models.cross_attention_fusion import CrossModalFusion

        model = CrossModalFusion()
        model.eval()
        onnx_path = tmp_path / "test_fusion.onnx"

        dummy_eeg = torch.randn(1, 6)
        dummy_voice = torch.randn(1, 5)
        from models.cross_attention_fusion import export_to_onnx
        export_to_onnx(model, str(onnx_path))
        assert onnx_path.exists(), "ONNX file was not created"
        assert onnx_path.stat().st_size > 0, "ONNX file is empty"


# ---------------------------------------------------------------------------
# 5. ONNX inference matches PyTorch output
# ---------------------------------------------------------------------------

class TestOnnxInferenceMatchesPyTorch:
    def test_onnx_output_matches_pytorch(self, tmp_path):
        """ONNX Runtime inference produces same output as PyTorch."""
        import torch
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        from models.cross_attention_fusion import CrossModalFusion

        model = CrossModalFusion()
        model.eval()
        onnx_path = tmp_path / "test_fusion.onnx"

        dummy_eeg = torch.randn(1, 6)
        dummy_voice = torch.randn(1, 5)

        # PyTorch inference
        with torch.no_grad():
            pt_out = model(dummy_eeg, dummy_voice).numpy()

        # Export to ONNX
        from models.cross_attention_fusion import export_to_onnx
        export_to_onnx(model, str(onnx_path))

        # ONNX inference
        session = ort.InferenceSession(str(onnx_path))
        ort_out = session.run(
            ["fused_logits"],
            {
                "eeg_probs": dummy_eeg.numpy(),
                "voice_probs": dummy_voice.numpy(),
            },
        )[0]

        np.testing.assert_allclose(pt_out, ort_out, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 6. fuse_cross_attention helper function works correctly
# ---------------------------------------------------------------------------

class TestFuseCrossAttentionHelper:
    def test_returns_valid_result_dict(self):
        """fuse_cross_attention returns dict with expected keys."""
        from models.cross_attention_fusion import fuse_cross_attention

        eeg_probs = {"happy": 0.6, "sad": 0.1, "angry": 0.05, "fear": 0.05, "surprise": 0.1, "neutral": 0.1}
        voice_probs = {"happy": 0.3, "sad": 0.2, "angry": 0.1, "fear": 0.1, "surprise": 0.3}
        result = fuse_cross_attention(eeg_probs, voice_probs)

        assert "emotion" in result
        assert "probabilities" in result
        assert "confidence" in result
        assert "model_type" in result
        assert result["model_type"] == "cross_attention_fusion"
        assert len(result["probabilities"]) == 6
        # Probabilities should sum to ~1
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_returns_fallback_when_model_missing(self):
        """When no trained model exists, falls back to weighted average."""
        from models.cross_attention_fusion import fuse_cross_attention

        eeg_probs = {"happy": 0.8, "sad": 0.05, "angry": 0.05, "fear": 0.03, "surprise": 0.02, "neutral": 0.05}
        voice_probs = {"happy": 0.7, "sad": 0.1, "angry": 0.05, "fear": 0.05, "surprise": 0.1}
        result = fuse_cross_attention(eeg_probs, voice_probs)

        # Even with fallback, should produce valid output
        assert result["emotion"] in ["happy", "sad", "angry", "fear", "surprise", "neutral"]
        assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_confidence_following_model(model, n_epochs=100):
    """Train the fusion model with knowledge distillation so it learns
    to follow the more confident modality."""
    import torch
    import torch.nn as nn

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()

    rng = np.random.default_rng(42)
    for _ in range(n_epochs):
        batch_size = 32
        # Generate synthetic training data
        eeg = rng.dirichlet(np.ones(6) * 2, batch_size).astype(np.float32)
        voice_5 = rng.dirichlet(np.ones(5) * 2, batch_size).astype(np.float32)

        # Teacher labels: argmax of the more confident prediction
        labels = []
        for i in range(batch_size):
            eeg_conf = eeg[i].max()
            voice_conf = voice_5[i].max()
            if eeg_conf > voice_conf:
                labels.append(eeg[i].argmax())
            else:
                # Voice has 5 classes, map to 6-class index (cap at 5)
                labels.append(min(voice_5[i].argmax(), 5))

        eeg_t = torch.from_numpy(eeg)
        voice_t = torch.from_numpy(voice_5)
        labels_t = torch.tensor(labels, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(eeg_t, voice_t)
        loss = criterion(logits, labels_t)
        loss.backward()
        optimizer.step()

    return model
