"""Tests for self-supervised contrastive EEG pretraining.

Covers EEGAugmentor augmentations, EEGContrastivePretrainer lifecycle,
and extract_features / pretrain / pretrain_step outputs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure ml root and training/ are importable
_ML_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ML_ROOT), str(_ML_ROOT / "training")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.self_supervised_pretrain import (
    EEGAugmentor,
    EEGContrastivePretrainer,
    TORCH_AVAILABLE,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(0)
EPOCHS_BATCH = RNG.standard_normal((8, 4, 256)).astype(np.float32) * 10.0
SINGLE_EPOCH = RNG.standard_normal((4, 256)).astype(np.float32) * 10.0
SINGLE_1D = RNG.standard_normal(256).astype(np.float32) * 10.0


# ─── EEGAugmentor ────────────────────────────────────────────────────────────

class TestTemporalJitter:
    def test_preserves_shape_2d(self):
        aug = EEGAugmentor()
        result = aug.temporal_jitter(SINGLE_EPOCH)
        assert result.shape == SINGLE_EPOCH.shape

    def test_preserves_shape_1d(self):
        aug = EEGAugmentor()
        result = aug.temporal_jitter(SINGLE_1D)
        assert result.shape == SINGLE_1D.shape

    def test_shifts_data(self):
        """With sufficient max_shift, at least one run should differ from original."""
        aug = EEGAugmentor()
        any_diff = any(
            not np.allclose(aug.temporal_jitter(SINGLE_1D, max_shift_ms=50), SINGLE_1D)
            for _ in range(30)
        )
        assert any_diff


class TestChannelDropout:
    def test_preserves_shape(self):
        aug = EEGAugmentor()
        result = aug.channel_dropout(SINGLE_EPOCH, p_drop=0.5)
        assert result.shape == SINGLE_EPOCH.shape

    def test_zeros_channel_sometimes(self):
        """With p_drop=1.0, every channel should be zeroed."""
        aug = EEGAugmentor()
        result = aug.channel_dropout(SINGLE_EPOCH.copy(), p_drop=1.0)
        assert np.all(result == 0.0)

    def test_no_mutation_of_input(self):
        aug = EEGAugmentor()
        original = SINGLE_EPOCH.copy()
        aug.channel_dropout(SINGLE_EPOCH, p_drop=1.0)
        np.testing.assert_array_equal(SINGLE_EPOCH, original)


class TestGaussianNoise:
    def test_output_differs_from_input(self):
        aug = EEGAugmentor()
        result = aug.gaussian_noise(SINGLE_EPOCH, snr_db=20.0)
        assert not np.allclose(result, SINGLE_EPOCH)

    def test_preserves_shape_2d(self):
        aug = EEGAugmentor()
        result = aug.gaussian_noise(SINGLE_EPOCH)
        assert result.shape == SINGLE_EPOCH.shape

    def test_preserves_shape_1d(self):
        aug = EEGAugmentor()
        result = aug.gaussian_noise(SINGLE_1D)
        assert result.shape == SINGLE_1D.shape


class TestFrequencyMasking:
    def test_preserves_shape_2d(self):
        aug = EEGAugmentor()
        result = aug.frequency_masking(SINGLE_EPOCH)
        assert result.shape == SINGLE_EPOCH.shape

    def test_preserves_shape_1d(self):
        aug = EEGAugmentor()
        result = aug.frequency_masking(SINGLE_1D)
        assert result.shape == SINGLE_1D.shape

    def test_output_differs_from_input(self):
        """Masking a band should change at least some values."""
        aug = EEGAugmentor()
        result = aug.frequency_masking(SINGLE_EPOCH)
        # Not all-identical after masking
        assert not np.allclose(result, SINGLE_EPOCH)


class TestAugment:
    def test_preserves_shape_2d(self):
        aug = EEGAugmentor()
        result = aug.augment(SINGLE_EPOCH)
        assert result.shape == SINGLE_EPOCH.shape

    def test_preserves_shape_1d(self):
        aug = EEGAugmentor()
        result = aug.augment(SINGLE_1D)
        assert result.shape == SINGLE_1D.shape

    def test_no_mutation_of_input(self):
        aug = EEGAugmentor()
        original = SINGLE_EPOCH.copy()
        aug.augment(SINGLE_EPOCH)
        np.testing.assert_array_equal(SINGLE_EPOCH, original)


# ─── EEGContrastivePretrainer ────────────────────────────────────────────────

class TestPretrainerInstantiation:
    def test_instantiates_without_error(self):
        pretrainer = EEGContrastivePretrainer()
        assert pretrainer is not None

    def test_default_temperature(self):
        pretrainer = EEGContrastivePretrainer()
        assert pretrainer.temperature == 0.1

    def test_custom_temperature(self):
        pretrainer = EEGContrastivePretrainer(temperature=0.5)
        assert pretrainer.temperature == 0.5


class TestGetPretrainerInfo:
    def test_returns_all_required_keys(self):
        pretrainer = EEGContrastivePretrainer()
        info = pretrainer.get_pretrainer_info()
        required_keys = {
            "architecture",
            "n_params",
            "torch_available",
            "temperature",
            "augmentations",
            "reference",
        }
        assert required_keys.issubset(info.keys())

    def test_augmentations_list_nonempty(self):
        pretrainer = EEGContrastivePretrainer()
        info = pretrainer.get_pretrainer_info()
        assert len(info["augmentations"]) > 0

    def test_torch_available_is_bool(self):
        pretrainer = EEGContrastivePretrainer()
        info = pretrainer.get_pretrainer_info()
        assert isinstance(info["torch_available"], bool)


class TestPretrainStep:
    def test_returns_torch_available_key(self):
        pretrainer = EEGContrastivePretrainer()
        result = pretrainer.pretrain_step(EPOCHS_BATCH)
        assert "torch_available" in result

    def test_does_not_crash_with_batch_of_4(self):
        pretrainer = EEGContrastivePretrainer()
        small_batch = EPOCHS_BATCH[:4]
        result = pretrainer.pretrain_step(small_batch)
        assert isinstance(result, dict)

    def test_n_samples_matches_input(self):
        pretrainer = EEGContrastivePretrainer()
        result = pretrainer.pretrain_step(EPOCHS_BATCH)
        assert result["n_samples"] == len(EPOCHS_BATCH)

    def test_loss_is_float_or_none(self):
        pretrainer = EEGContrastivePretrainer()
        result = pretrainer.pretrain_step(EPOCHS_BATCH)
        loss = result.get("loss")
        assert loss is None or isinstance(loss, float)


class TestExtractFeatures:
    def test_2d_input_returns_1d_output(self):
        pretrainer = EEGContrastivePretrainer()
        embedding = pretrainer.extract_features(SINGLE_EPOCH)
        assert embedding.ndim == 1

    def test_returns_array_of_positive_length(self):
        pretrainer = EEGContrastivePretrainer()
        embedding = pretrainer.extract_features(SINGLE_EPOCH)
        assert len(embedding) > 0

    def test_batch_input_returns_batch_output(self):
        pretrainer = EEGContrastivePretrainer()
        batch = EPOCHS_BATCH[:3]
        embeddings = pretrainer.extract_features(batch)
        assert embeddings.shape[0] == 3

    def test_embed_dim_matches_encoder(self):
        pretrainer = EEGContrastivePretrainer(embed_dim=64)
        embedding = pretrainer.extract_features(SINGLE_EPOCH)
        assert len(embedding) == 64


class TestPretrain:
    def test_returns_n_epochs_trained(self):
        pretrainer = EEGContrastivePretrainer()
        result = pretrainer.pretrain(EPOCHS_BATCH, n_epochs=2, batch_size=4)
        assert result["n_epochs_trained"] == 2 or result["n_epochs_trained"] == 0

    def test_n_samples_matches_dataset(self):
        pretrainer = EEGContrastivePretrainer()
        result = pretrainer.pretrain(EPOCHS_BATCH, n_epochs=1, batch_size=4)
        assert result["n_samples"] == len(EPOCHS_BATCH)

    def test_final_loss_is_float_or_none(self):
        pretrainer = EEGContrastivePretrainer()
        result = pretrainer.pretrain(EPOCHS_BATCH, n_epochs=2, batch_size=4)
        final_loss = result.get("final_loss")
        assert final_loss is None or isinstance(final_loss, float)

    def test_torch_available_key_present(self):
        pretrainer = EEGContrastivePretrainer()
        result = pretrainer.pretrain(EPOCHS_BATCH, n_epochs=1, batch_size=4)
        assert "torch_available" in result


# ─── NT-Xent loss (only when PyTorch available) ───────────────────────────────

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestNTXentLoss:
    def test_returns_scalar(self):
        import torch
        pretrainer = EEGContrastivePretrainer()
        z1 = torch.randn(4, 64)
        z2 = torch.randn(4, 64)
        loss = pretrainer.nt_xent_loss(z1, z2)
        assert loss.ndim == 0  # scalar tensor

    def test_loss_is_positive(self):
        import torch
        pretrainer = EEGContrastivePretrainer()
        z1 = torch.randn(8, 64)
        z2 = torch.randn(8, 64)
        loss = pretrainer.nt_xent_loss(z1, z2)
        assert float(loss.item()) > 0
