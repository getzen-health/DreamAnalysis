"""Tests for self-supervised contrastive EEG encoder.

Verifies:
1. Encoder produces 64-dim output
2. Projector produces 32-dim output
3. Augmentations produce different but similar signals
4. NT-Xent loss is positive and finite
5. Forward pass works with batch of 4-channel 1024-sample inputs
6. encode() returns same shape as encoder forward
7. Save/load roundtrip preserves weights
8. Pretraining data generation produces correct shape
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

from models.eeg_ssl import EEGContrastiveEncoder, augment_eeg, nt_xent_loss
from training.pretrain_eeg_ssl import generate_pretraining_data


@pytest.fixture
def encoder():
    """Fresh EEGContrastiveEncoder with default params."""
    return EEGContrastiveEncoder(n_channels=4, n_samples=1024, embed_dim=64, proj_dim=32)


@pytest.fixture
def batch_eeg():
    """Batch of 8 x 4-channel x 1024-sample EEG tensors."""
    return torch.randn(8, 4, 1024)


class TestEEGContrastiveEncoder:
    """Tests for the encoder architecture."""

    def test_encoder_output_dim(self, encoder, batch_eeg):
        """Encoder must produce (batch, 64) representation."""
        h = encoder.encode(batch_eeg)
        assert h.shape == (8, 64)

    def test_projector_output_dim(self, encoder, batch_eeg):
        """Full forward must produce (batch, 32) projection."""
        h, z = encoder(batch_eeg)
        assert h.shape == (8, 64)
        assert z.shape == (8, 32)

    def test_encode_matches_forward_representation(self, encoder, batch_eeg):
        """encode() should return the same representation as forward()[0]."""
        encoder.eval()
        with torch.no_grad():
            h_encode = encoder.encode(batch_eeg)
            h_forward, _ = encoder(batch_eeg)
        torch.testing.assert_close(h_encode, h_forward)

    def test_forward_single_sample(self, encoder):
        """Forward pass should work with batch_size=1."""
        x = torch.randn(1, 4, 1024)
        h, z = encoder(x)
        assert h.shape == (1, 64)
        assert z.shape == (1, 32)

    def test_outputs_are_finite(self, encoder, batch_eeg):
        """All outputs must be finite."""
        h, z = encoder(batch_eeg)
        assert torch.all(torch.isfinite(h))
        assert torch.all(torch.isfinite(z))

    def test_different_sample_lengths(self):
        """Encoder should work with different n_samples via AdaptiveAvgPool1d."""
        enc = EEGContrastiveEncoder(n_channels=4, n_samples=512, embed_dim=64)
        x = torch.randn(4, 4, 512)
        h = enc.encode(x)
        assert h.shape == (4, 64)

    def test_save_load_roundtrip(self, encoder, batch_eeg):
        """Save and load should preserve weights exactly."""
        encoder.eval()
        with torch.no_grad():
            h_before, z_before = encoder(batch_eeg)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            encoder.save(f.name)
            loaded = EEGContrastiveEncoder.load(f.name)

        loaded.eval()
        with torch.no_grad():
            h_after, z_after = loaded(batch_eeg)

        torch.testing.assert_close(h_before, h_after)
        torch.testing.assert_close(z_before, z_after)
        Path(f.name).unlink()


class TestAugmentations:
    """Tests for EEG augmentations."""

    def test_augmentation_produces_different_signal(self):
        """Augmented signal should differ from original."""
        import random

        x = torch.randn(4, 1024)
        rng = random.Random(42)
        aug = augment_eeg(x, rng)

        # Should not be identical (extremely unlikely with noise + scaling)
        assert not torch.allclose(x, aug, atol=1e-6), "Augmented signal is identical to original"

    def test_augmentation_preserves_shape(self):
        """Augmentation must not change tensor shape."""
        x = torch.randn(4, 1024)
        aug = augment_eeg(x)
        assert aug.shape == x.shape

    def test_augmentation_produces_finite_values(self):
        """Augmented signal must be finite."""
        x = torch.randn(4, 1024)
        aug = augment_eeg(x)
        assert torch.all(torch.isfinite(aug))

    def test_two_augmentations_differ(self):
        """Two independent augmentations of the same signal should differ."""
        import random

        x = torch.randn(4, 1024)
        aug1 = augment_eeg(x, random.Random(1))
        aug2 = augment_eeg(x, random.Random(2))
        assert not torch.allclose(aug1, aug2, atol=1e-6)


class TestNTXentLoss:
    """Tests for NT-Xent contrastive loss."""

    def test_loss_is_positive(self):
        """NT-Xent loss should be positive."""
        z1 = torch.randn(16, 32)
        z2 = torch.randn(16, 32)
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        assert loss.item() > 0

    def test_loss_is_finite(self):
        """NT-Xent loss must be finite."""
        z1 = torch.randn(16, 32)
        z2 = torch.randn(16, 32)
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        assert torch.isfinite(loss)

    def test_identical_pairs_have_lower_loss(self):
        """Identical positive pairs should have lower loss than random pairs."""
        z1 = torch.randn(16, 32)
        # Identical pair (should be easy for the loss)
        loss_identical = nt_xent_loss(z1, z1.clone(), temperature=0.5)
        # Random pair (should be harder)
        z2 = torch.randn(16, 32)
        loss_random = nt_xent_loss(z1, z2, temperature=0.5)
        assert loss_identical.item() < loss_random.item()

    def test_loss_with_small_batch(self):
        """Loss should work with batch_size=2 (minimum)."""
        z1 = torch.randn(2, 32)
        z2 = torch.randn(2, 32)
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        assert torch.isfinite(loss)


class TestPretrainingData:
    """Tests for synthetic pretraining data generation."""

    def test_data_shape(self):
        """Generated data must have correct shape."""
        data = generate_pretraining_data(n_epochs=100, n_channels=4, n_samples=1024)
        assert data.shape == (100, 4, 1024)

    def test_data_is_finite(self):
        """All generated data must be finite."""
        data = generate_pretraining_data(n_epochs=50)
        assert np.all(np.isfinite(data))

    def test_data_has_variance(self):
        """Generated data should not be constant."""
        data = generate_pretraining_data(n_epochs=50)
        assert np.std(data) > 0.1

    def test_different_seeds_produce_different_data(self):
        """Different seeds should produce different data."""
        d1 = generate_pretraining_data(n_epochs=10, seed=1)
        d2 = generate_pretraining_data(n_epochs=10, seed=2)
        assert not np.allclose(d1, d2)

    def test_reproducible_with_same_seed(self):
        """Same seed should produce identical data."""
        d1 = generate_pretraining_data(n_epochs=10, seed=42)
        d2 = generate_pretraining_data(n_epochs=10, seed=42)
        np.testing.assert_array_equal(d1, d2)
