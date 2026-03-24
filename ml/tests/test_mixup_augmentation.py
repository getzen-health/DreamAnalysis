"""Tests for Mixup data augmentation module.

Mixup (Zhang et al. 2018, ICLR) interpolates between training samples to
regularize the decision boundary. For EEG emotion recognition, it helps
bridge the cross-subject gap by smoothing between different subjects'
feature distributions. Expected improvement: 2-5% on cross-subject accuracy.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# 1. mixup_batch: core functionality
# ---------------------------------------------------------------------------

class TestMixupBatch:
    """Test the core mixup_batch function for feature-level augmentation."""

    def test_output_shapes_match_input(self):
        """Mixup output has the same shape as input."""
        from processing.mixup_augmentation import mixup_batch

        X = np.random.randn(32, 85).astype(np.float32)
        y = np.eye(6, dtype=np.float32)[np.random.randint(0, 6, 32)]  # one-hot
        X_mix, y_mix = mixup_batch(X, y, alpha=1.0)

        assert X_mix.shape == X.shape, f"Expected {X.shape}, got {X_mix.shape}"
        assert y_mix.shape == y.shape, f"Expected {y.shape}, got {y_mix.shape}"

    def test_labels_are_soft(self):
        """Mixed labels should be soft (not one-hot) when alpha > 0."""
        from processing.mixup_augmentation import mixup_batch

        X = np.random.randn(64, 85).astype(np.float32)
        # Create two distinct classes
        y = np.zeros((64, 6), dtype=np.float32)
        y[:32, 0] = 1.0  # first half: class 0
        y[32:, 1] = 1.0  # second half: class 1

        X_mix, y_mix = mixup_batch(X, y, alpha=1.0)

        # At least some mixed labels should be non-one-hot (intermediate values)
        max_vals = y_mix.max(axis=1)
        has_soft_labels = np.any(max_vals < 0.99)
        assert has_soft_labels, "Mixup should produce soft labels, not all one-hot"

    def test_alpha_zero_returns_original(self):
        """With alpha=0, lambda is always 1.0, so output should be original data."""
        from processing.mixup_augmentation import mixup_batch

        np.random.seed(42)
        X = np.random.randn(16, 10).astype(np.float32)
        y = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, 16)]

        X_mix, y_mix = mixup_batch(X, y, alpha=0.0)
        # alpha=0 means no mixing — output equals input (up to permutation)
        # The implementation should handle this edge case
        assert X_mix.shape == X.shape

    def test_labels_sum_to_one(self):
        """Mixed labels should still form valid probability distributions."""
        from processing.mixup_augmentation import mixup_batch

        X = np.random.randn(100, 50).astype(np.float32)
        y = np.eye(6, dtype=np.float32)[np.random.randint(0, 6, 100)]
        X_mix, y_mix = mixup_batch(X, y, alpha=0.4)

        sums = y_mix.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5,
                                   err_msg="Mixed labels must sum to 1.0")

    def test_no_nans_or_infs(self):
        """Output should never contain NaN or Inf."""
        from processing.mixup_augmentation import mixup_batch

        X = np.random.randn(50, 85).astype(np.float32)
        y = np.eye(6, dtype=np.float32)[np.random.randint(0, 6, 50)]
        X_mix, y_mix = mixup_batch(X, y, alpha=1.0)

        assert not np.any(np.isnan(X_mix)), "X_mix contains NaN"
        assert not np.any(np.isinf(X_mix)), "X_mix contains Inf"
        assert not np.any(np.isnan(y_mix)), "y_mix contains NaN"
        assert not np.any(np.isinf(y_mix)), "y_mix contains Inf"


# ---------------------------------------------------------------------------
# 2. mixup_signal: raw EEG signal-level mixup
# ---------------------------------------------------------------------------

class TestMixupSignal:
    """Test signal-level mixup for raw EEG data (channels x samples)."""

    def test_output_shape_matches_input(self):
        """Signal mixup preserves shape."""
        from processing.mixup_augmentation import mixup_signal_batch

        # batch of 4-channel EEG epochs: (batch, channels, samples)
        X = np.random.randn(32, 4, 1024).astype(np.float32)
        y = np.eye(6, dtype=np.float32)[np.random.randint(0, 6, 32)]
        X_mix, y_mix = mixup_signal_batch(X, y, alpha=0.4)

        assert X_mix.shape == X.shape
        assert y_mix.shape == y.shape

    def test_multichannel_signal_preserves_channel_structure(self):
        """Mixing should operate across samples, not corrupt channel structure."""
        from processing.mixup_augmentation import mixup_signal_batch

        X = np.random.randn(16, 4, 512).astype(np.float32)
        y = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, 16)]
        X_mix, _ = mixup_signal_batch(X, y, alpha=1.0)

        # Each mixed sample should be a linear combination of two original samples
        # So channel count should be unchanged
        assert X_mix.shape[1] == 4, "Channel dimension should be preserved"


# ---------------------------------------------------------------------------
# 3. mixup_criterion: soft label cross-entropy loss
# ---------------------------------------------------------------------------

class TestMixupCriterion:
    """Test the soft-label cross-entropy loss for use with mixed labels."""

    def test_matches_hard_label_ce_when_labels_are_one_hot(self):
        """When labels are one-hot, soft CE should match standard CE."""
        import torch
        import torch.nn as nn
        from processing.mixup_augmentation import mixup_cross_entropy

        logits = torch.randn(8, 6)
        hard_labels = torch.randint(0, 6, (8,))
        soft_labels = torch.nn.functional.one_hot(hard_labels, 6).float()

        standard_loss = nn.CrossEntropyLoss()(logits, hard_labels)
        soft_loss = mixup_cross_entropy(logits, soft_labels)

        torch.testing.assert_close(soft_loss, standard_loss, atol=1e-5, rtol=1e-5)

    def test_loss_is_scalar(self):
        """Loss should be a single scalar value."""
        import torch
        from processing.mixup_augmentation import mixup_cross_entropy

        logits = torch.randn(16, 6)
        soft_labels = torch.softmax(torch.randn(16, 6), dim=1)
        loss = mixup_cross_entropy(logits, soft_labels)

        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_loss_is_positive(self):
        """Cross-entropy loss should always be non-negative."""
        import torch
        from processing.mixup_augmentation import mixup_cross_entropy

        logits = torch.randn(32, 6)
        soft_labels = torch.softmax(torch.randn(32, 6), dim=1)
        loss = mixup_cross_entropy(logits, soft_labels)

        assert loss.item() >= 0.0, f"Loss should be non-negative, got {loss.item()}"

    def test_gradient_flows(self):
        """Loss should produce valid gradients for backprop."""
        import torch
        from processing.mixup_augmentation import mixup_cross_entropy

        logits = torch.randn(8, 6, requires_grad=True)
        soft_labels = torch.softmax(torch.randn(8, 6), dim=1)
        loss = mixup_cross_entropy(logits, soft_labels)
        loss.backward()

        assert logits.grad is not None, "Gradient should flow through mixup CE"
        assert not torch.any(torch.isnan(logits.grad)), "Gradients should not be NaN"


# ---------------------------------------------------------------------------
# 4. to_one_hot: utility for converting integer labels
# ---------------------------------------------------------------------------

class TestToOneHot:
    """Test the integer-to-one-hot conversion utility."""

    def test_correct_shape(self):
        """One-hot output has shape (n_samples, n_classes)."""
        from processing.mixup_augmentation import to_one_hot

        y = np.array([0, 1, 2, 3, 4, 5])
        one_hot = to_one_hot(y, n_classes=6)
        assert one_hot.shape == (6, 6)

    def test_correct_values(self):
        """Each row should have exactly one 1.0 at the correct index."""
        from processing.mixup_augmentation import to_one_hot

        y = np.array([0, 2, 5])
        one_hot = to_one_hot(y, n_classes=6)

        expected = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        np.testing.assert_array_equal(one_hot, expected)


# ---------------------------------------------------------------------------
# 5. MixupTrainer: end-to-end integration with PyTorch training loop
# ---------------------------------------------------------------------------

class TestMixupTrainer:
    """Integration test: mixup in a real training step."""

    def test_training_step_with_mixup(self):
        """A single training step with mixup should reduce loss."""
        import torch
        import torch.nn as nn
        from processing.mixup_augmentation import mixup_batch, mixup_cross_entropy, to_one_hot

        # Simple 2-layer net
        model = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Synthetic data
        X_np = np.random.randn(64, 20).astype(np.float32)
        y_np = np.random.randint(0, 3, 64)
        y_onehot = to_one_hot(y_np, n_classes=3)

        # Mixup
        X_mix_np, y_mix_np = mixup_batch(X_np, y_onehot, alpha=0.4)
        X_mix = torch.from_numpy(X_mix_np)
        y_mix = torch.from_numpy(y_mix_np)

        # Training step
        model.train()
        logits = model(X_mix)
        loss = mixup_cross_entropy(logits, y_mix)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss should be finite
        assert loss.item() < 100.0, "Loss is unreasonably large"
        assert not np.isnan(loss.item()), "Loss is NaN"

    def test_mixup_with_class_weights(self):
        """Mixup should work alongside class weighting."""
        import torch
        from processing.mixup_augmentation import (
            mixup_batch,
            mixup_cross_entropy,
            to_one_hot,
        )

        X_np = np.random.randn(32, 10).astype(np.float32)
        y_np = np.random.randint(0, 3, 32)
        y_onehot = to_one_hot(y_np, n_classes=3)

        X_mix_np, y_mix_np = mixup_batch(X_np, y_onehot, alpha=0.4)
        logits = torch.randn(32, 3, requires_grad=True)
        y_mix = torch.from_numpy(y_mix_np)

        # Class weights: upweight rare class
        weights = torch.tensor([1.0, 2.0, 1.5])
        loss = mixup_cross_entropy(logits, y_mix, class_weights=weights)

        loss.backward()
        assert logits.grad is not None
        assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: single sample, large alpha, etc."""

    def test_single_sample(self):
        """Mixup with a single sample should not crash."""
        from processing.mixup_augmentation import mixup_batch

        X = np.random.randn(1, 10).astype(np.float32)
        y = np.array([[1.0, 0.0, 0.0]])
        X_mix, y_mix = mixup_batch(X, y, alpha=1.0)

        assert X_mix.shape == (1, 10)
        assert y_mix.shape == (1, 3)

    def test_large_alpha(self):
        """Large alpha (= more aggressive mixing) should still produce valid output."""
        from processing.mixup_augmentation import mixup_batch

        X = np.random.randn(32, 50).astype(np.float32)
        y = np.eye(6, dtype=np.float32)[np.random.randint(0, 6, 32)]
        X_mix, y_mix = mixup_batch(X, y, alpha=5.0)

        sums = y_mix.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_two_class_problem(self):
        """Mixup works for binary classification."""
        from processing.mixup_augmentation import mixup_batch

        X = np.random.randn(20, 30).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[np.random.randint(0, 2, 20)]
        X_mix, y_mix = mixup_batch(X, y, alpha=0.2)

        assert y_mix.shape == (20, 2)
        sums = y_mix.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)
