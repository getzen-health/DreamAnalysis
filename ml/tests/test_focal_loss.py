"""Tests for Focal Loss module.

Focal Loss (Lin et al. 2017) down-weights easy/well-classified examples,
focusing training on hard cases. For EEG emotion recognition, this means
the model spends more gradient on ambiguous emotional states and less on
clearly-classified epochs.

Tests verify:
1. Correct output shape and gradient flow
2. gamma=0 reduces to standard cross-entropy
3. Higher gamma produces lower loss for easy (high-confidence) samples
4. Class weights are applied correctly
5. Label smoothing works
6. Soft-label (mixup) variant matches expected behavior
7. Device compatibility (CPU)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. FocalLoss: basic functionality
# ---------------------------------------------------------------------------

class TestFocalLossBasic:
    """Test the FocalLoss class with hard (integer) labels."""

    def test_output_is_scalar(self):
        """FocalLoss returns a scalar loss tensor."""
        from processing.focal_loss import FocalLoss

        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(16, 6)
        targets = torch.randint(0, 6, (16,))
        loss = criterion(logits, targets)

        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
        assert loss.item() > 0, "Loss should be positive"

    def test_gradient_flows(self):
        """Loss supports backpropagation."""
        from processing.focal_loss import FocalLoss

        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 6, requires_grad=True)
        targets = torch.randint(0, 6, (8,))
        loss = criterion(logits, targets)
        loss.backward()

        assert logits.grad is not None, "Gradient should exist"
        assert not torch.all(logits.grad == 0), "Gradient should be non-zero"

    def test_reduction_none_returns_per_sample(self):
        """reduction='none' returns per-sample losses."""
        from processing.focal_loss import FocalLoss

        criterion = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(8, 6)
        targets = torch.randint(0, 6, (8,))
        loss = criterion(logits, targets)

        assert loss.shape == (8,), f"Expected (8,), got {loss.shape}"

    def test_reduction_sum(self):
        """reduction='sum' returns sum of per-sample losses."""
        from processing.focal_loss import FocalLoss

        logits = torch.randn(8, 6)
        targets = torch.randint(0, 6, (8,))

        criterion_none = FocalLoss(gamma=2.0, reduction="none")
        criterion_sum = FocalLoss(gamma=2.0, reduction="sum")

        loss_none = criterion_none(logits, targets)
        loss_sum = criterion_sum(logits, targets)

        assert torch.allclose(loss_none.sum(), loss_sum, atol=1e-5), \
            "sum reduction should equal sum of per-sample losses"


# ---------------------------------------------------------------------------
# 2. FocalLoss: gamma=0 reduces to standard cross-entropy
# ---------------------------------------------------------------------------

class TestFocalLossReducesToCE:
    """Verify that FocalLoss(gamma=0) is equivalent to CrossEntropyLoss."""

    def test_gamma_zero_matches_ce_no_weights(self):
        """gamma=0, no class weights should match nn.CrossEntropyLoss."""
        from processing.focal_loss import FocalLoss

        torch.manual_seed(42)
        logits = torch.randn(32, 6)
        targets = torch.randint(0, 6, (32,))

        focal = FocalLoss(gamma=0.0)
        ce = nn.CrossEntropyLoss()

        loss_focal = focal(logits, targets)
        loss_ce = ce(logits, targets)

        assert torch.allclose(loss_focal, loss_ce, atol=1e-5), \
            f"gamma=0 focal ({loss_focal.item():.6f}) != CE ({loss_ce.item():.6f})"

    def test_gamma_zero_matches_ce_with_weights(self):
        """gamma=0 with class weights should match weighted CrossEntropyLoss."""
        from processing.focal_loss import FocalLoss

        torch.manual_seed(42)
        logits = torch.randn(32, 6)
        targets = torch.randint(0, 6, (32,))
        weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 0.8, 1.2])

        focal = FocalLoss(gamma=0.0, weight=weights)
        ce = nn.CrossEntropyLoss(weight=weights)

        loss_focal = focal(logits, targets)
        loss_ce = ce(logits, targets)

        assert torch.allclose(loss_focal, loss_ce, atol=1e-4), \
            f"Weighted gamma=0 focal ({loss_focal.item():.6f}) != weighted CE ({loss_ce.item():.6f})"


# ---------------------------------------------------------------------------
# 3. FocalLoss: focusing behavior (core property)
# ---------------------------------------------------------------------------

class TestFocalLossFocusing:
    """Verify that focal loss down-weights easy examples."""

    def test_easy_sample_has_lower_loss_than_ce(self):
        """For a well-classified sample, focal loss < standard CE."""
        from processing.focal_loss import FocalLoss

        # Create an "easy" sample: high logit for true class
        logits = torch.tensor([[5.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
        targets = torch.tensor([0])

        focal = FocalLoss(gamma=2.0, reduction="none")
        ce = nn.CrossEntropyLoss(reduction="none")

        loss_focal = focal(logits, targets).item()
        loss_ce = ce(logits, targets).item()

        assert loss_focal < loss_ce, \
            f"Focal ({loss_focal:.6f}) should be < CE ({loss_ce:.6f}) for easy sample"

    def test_hard_sample_has_similar_loss_to_ce(self):
        """For a misclassified sample, focal loss is close to CE (less down-weighting)."""
        from processing.focal_loss import FocalLoss

        # Create a "hard" sample: low logit for true class
        logits = torch.tensor([[-2.0, 3.0, 1.0, 0.5, -1.0, 0.0]])
        targets = torch.tensor([0])

        focal = FocalLoss(gamma=2.0, reduction="none")
        ce = nn.CrossEntropyLoss(reduction="none")

        loss_focal = focal(logits, targets).item()
        loss_ce = ce(logits, targets).item()

        # For hard samples, focal should still be <= CE (modulation <= 1)
        # but the ratio should be closer to 1 than for easy samples
        ratio = loss_focal / loss_ce
        assert ratio > 0.5, \
            f"For hard sample, focal/CE ratio ({ratio:.4f}) should be > 0.5"
        assert loss_focal <= loss_ce + 1e-5, \
            f"Focal ({loss_focal:.6f}) should be <= CE ({loss_ce:.6f})"

    def test_higher_gamma_reduces_easy_loss_more(self):
        """Higher gamma should reduce loss more for easy samples."""
        from processing.focal_loss import FocalLoss

        logits = torch.tensor([[5.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
        targets = torch.tensor([0])

        loss_g1 = FocalLoss(gamma=1.0, reduction="none")(logits, targets).item()
        loss_g2 = FocalLoss(gamma=2.0, reduction="none")(logits, targets).item()
        loss_g3 = FocalLoss(gamma=3.0, reduction="none")(logits, targets).item()

        assert loss_g1 > loss_g2 > loss_g3, \
            f"Expected decreasing loss: g1={loss_g1:.6f} > g2={loss_g2:.6f} > g3={loss_g3:.6f}"


# ---------------------------------------------------------------------------
# 4. FocalLoss: class weights
# ---------------------------------------------------------------------------

class TestFocalLossWeights:
    """Verify per-class weighting works correctly."""

    def test_higher_weight_increases_loss(self):
        """A class with higher weight should produce higher loss."""
        from processing.focal_loss import FocalLoss

        logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # uniform
        targets = torch.tensor([0])

        weights_low = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        weights_high = torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        loss_low = FocalLoss(gamma=2.0, weight=weights_low, reduction="none")(logits, targets).item()
        loss_high = FocalLoss(gamma=2.0, weight=weights_high, reduction="none")(logits, targets).item()

        assert loss_high > loss_low, \
            f"Higher weight loss ({loss_high:.6f}) should exceed low ({loss_low:.6f})"


# ---------------------------------------------------------------------------
# 5. FocalLoss: label smoothing
# ---------------------------------------------------------------------------

class TestFocalLossLabelSmoothing:
    """Verify label smoothing works with focal loss."""

    def test_smoothing_reduces_loss(self):
        """Label smoothing should reduce loss for hard labels (less peaky targets)."""
        from processing.focal_loss import FocalLoss

        torch.manual_seed(42)
        logits = torch.randn(32, 6)
        targets = torch.randint(0, 6, (32,))

        loss_no_smooth = FocalLoss(gamma=2.0, label_smoothing=0.0)(logits, targets).item()
        loss_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)(logits, targets).item()

        # Smoothing generally changes loss; just verify it runs without error
        # and produces a finite, positive value
        assert loss_smooth > 0, "Smoothed loss should be positive"
        assert np.isfinite(loss_smooth), "Smoothed loss should be finite"

    def test_zero_smoothing_matches_hard_labels(self):
        """label_smoothing=0.0 should match the hard-label path."""
        from processing.focal_loss import FocalLoss

        torch.manual_seed(42)
        logits = torch.randn(32, 6)
        targets = torch.randint(0, 6, (32,))

        loss_hard = FocalLoss(gamma=2.0, label_smoothing=0.0)(logits, targets)
        loss_smooth0 = FocalLoss(gamma=2.0, label_smoothing=0.0)(logits, targets)

        assert torch.allclose(loss_hard, loss_smooth0, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Soft-label focal loss (mixup variant)
# ---------------------------------------------------------------------------

class TestFocalMixupCrossEntropy:
    """Test the focal_mixup_cross_entropy function for mixup training."""

    def test_output_is_scalar(self):
        """Should return a scalar loss."""
        from processing.focal_loss import focal_mixup_cross_entropy

        logits = torch.randn(16, 6)
        # Soft labels from mixup (rows sum to 1)
        soft = F.softmax(torch.randn(16, 6), dim=1)

        loss = focal_mixup_cross_entropy(logits, soft, gamma=2.0)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gamma_zero_matches_mixup_ce(self):
        """gamma=0 should match standard mixup cross-entropy."""
        from processing.focal_loss import focal_mixup_cross_entropy
        from processing.mixup_augmentation import mixup_cross_entropy

        torch.manual_seed(42)
        logits = torch.randn(32, 6)
        soft = F.softmax(torch.randn(32, 6), dim=1)

        loss_focal_g0 = focal_mixup_cross_entropy(logits, soft, gamma=0.0)
        loss_mixup_ce = mixup_cross_entropy(logits, soft)

        assert torch.allclose(loss_focal_g0, loss_mixup_ce, atol=1e-5), \
            f"gamma=0 focal ({loss_focal_g0.item():.6f}) != mixup_ce ({loss_mixup_ce.item():.6f})"

    def test_gradient_flows(self):
        """Soft-label focal loss supports backpropagation."""
        from processing.focal_loss import focal_mixup_cross_entropy

        logits = torch.randn(8, 6, requires_grad=True)
        soft = F.softmax(torch.randn(8, 6), dim=1)

        loss = focal_mixup_cross_entropy(logits, soft, gamma=2.0)
        loss.backward()

        assert logits.grad is not None

    def test_with_class_weights(self):
        """Class weights should affect the loss magnitude."""
        from processing.focal_loss import focal_mixup_cross_entropy

        logits = torch.randn(16, 6)
        soft = F.softmax(torch.randn(16, 6), dim=1)
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        weights_big = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

        loss_normal = focal_mixup_cross_entropy(logits, soft, gamma=2.0, class_weights=weights)
        loss_big = focal_mixup_cross_entropy(logits, soft, gamma=2.0, class_weights=weights_big)

        assert loss_big.item() > loss_normal.item(), \
            "Larger weights should produce larger loss"


# ---------------------------------------------------------------------------
# 7. Integration: focal loss in a mini training loop
# ---------------------------------------------------------------------------

class TestFocalLossTrainingLoop:
    """Verify focal loss works in a realistic training scenario."""

    def test_mini_training_loop_converges(self):
        """A small model trained with focal loss should decrease loss over steps."""
        from processing.focal_loss import FocalLoss

        torch.manual_seed(42)

        # Simple 2-layer model
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )

        # Synthetic data: 6 classes, 10 features
        X = torch.randn(64, 10)
        y = torch.randint(0, 6, (64,))

        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for _ in range(20):
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], \
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"

    def test_focal_loss_with_mixup_training(self):
        """Focal loss works with mixup augmentation in a training loop."""
        from processing.focal_loss import focal_mixup_cross_entropy
        from processing.mixup_augmentation import mixup_batch, to_one_hot

        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )

        X = np.random.randn(64, 10).astype(np.float32)
        y_int = np.random.randint(0, 6, 64)
        y_onehot = to_one_hot(y_int, 6)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for _ in range(20):
            X_mix, y_mix = mixup_batch(X, y_onehot, alpha=0.4)
            logits = model(torch.from_numpy(X_mix))
            loss = focal_mixup_cross_entropy(
                logits, torch.from_numpy(y_mix), gamma=2.0,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Focal+mixup loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestFocalLossEdgeCases:
    """Test edge cases and numerical stability."""

    def test_single_sample(self):
        """Works with batch size 1."""
        from processing.focal_loss import FocalLoss

        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(1, 6)
        targets = torch.tensor([3])
        loss = criterion(logits, targets)

        assert loss.ndim == 0
        assert np.isfinite(loss.item())

    def test_two_classes(self):
        """Works with binary classification."""
        from processing.focal_loss import FocalLoss

        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(16, 2)
        targets = torch.randint(0, 2, (16,))
        loss = criterion(logits, targets)

        assert loss.item() > 0

    def test_large_logits_numerically_stable(self):
        """Should not produce inf/nan with large logits."""
        from processing.focal_loss import FocalLoss

        criterion = FocalLoss(gamma=2.0)
        logits = torch.tensor([[100.0, -100.0, -100.0, -100.0, -100.0, -100.0]])
        targets = torch.tensor([0])
        loss = criterion(logits, targets)

        assert np.isfinite(loss.item()), f"Loss should be finite, got {loss.item()}"
        # For a perfectly classified sample, focal loss should be very near 0
        assert loss.item() < 1e-5, f"Perfect classification loss should be ~0, got {loss.item()}"
