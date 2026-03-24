"""Mixup Data Augmentation for EEG Training.

Implements Mixup (Zhang et al. 2018, ICLR) for EEG emotion classification.
Creates virtual training examples by linearly interpolating between pairs of
training samples and their labels. This regularizes the decision boundary and
reduces overfitting, especially for small EEG datasets with cross-subject
variability.

Expected improvement: 2-5% on cross-subject emotion accuracy (based on
2024 EEG-BCI literature). Works for both feature-level and raw-signal data.

Usage in training loop:
    from processing.mixup_augmentation import (
        mixup_batch, mixup_cross_entropy, to_one_hot,
    )

    y_onehot = to_one_hot(y_int, n_classes=6)
    for xb, yb in dataloader:
        xb_mix, yb_mix = mixup_batch(xb.numpy(), yb.numpy(), alpha=0.4)
        logits = model(torch.from_numpy(xb_mix))
        loss = mixup_cross_entropy(logits, torch.from_numpy(yb_mix))
        loss.backward()

References:
    Zhang et al. (2018). "mixup: Beyond Empirical Risk Minimization." ICLR.
    Luo et al. (2024). "Mixup augmentation for EEG-based emotion recognition."
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utility: integer labels -> one-hot
# ---------------------------------------------------------------------------

def to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer label vector to one-hot encoded matrix.

    Args:
        y: Integer labels, shape (n_samples,). Values in [0, n_classes).
        n_classes: Number of classes.

    Returns:
        One-hot matrix, shape (n_samples, n_classes), dtype float32.
    """
    y = np.asarray(y, dtype=np.int64)
    one_hot = np.zeros((len(y), n_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot


# ---------------------------------------------------------------------------
# Core: feature-level mixup
# ---------------------------------------------------------------------------

def mixup_batch(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Mixup augmentation to a batch of feature vectors.

    For each sample i, draws a random partner j and mixing coefficient
    lambda ~ Beta(alpha, alpha), then computes:
        X_mix[i] = lambda * X[i] + (1 - lambda) * X[j]
        y_mix[i] = lambda * y[i] + (1 - lambda) * y[j]

    Args:
        X: Feature matrix, shape (batch, n_features). Float.
        y: One-hot label matrix, shape (batch, n_classes). Float.
           Must be one-hot or soft probability distribution (rows sum to 1).
        alpha: Beta distribution parameter. Higher = more aggressive mixing.
               alpha=0 disables mixing (lambda=1 always).
               alpha=0.2-0.4 recommended for EEG (conservative).
               alpha=1.0 = uniform mixing.

    Returns:
        (X_mixed, y_mixed): Same shapes as input, with interpolated values.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    batch_size = X.shape[0]

    if batch_size <= 1:
        # Nothing to mix with a single sample
        return X.copy(), y.copy()

    # Draw mixing coefficient from Beta distribution
    if alpha <= 0.0:
        lam = 1.0
    else:
        lam = float(np.random.beta(alpha, alpha))

    # Ensure lambda >= 0.5 so the first sample dominates
    # This is the standard trick from the mixup paper
    lam = max(lam, 1.0 - lam)

    # Random permutation for pairing
    indices = np.random.permutation(batch_size)

    X_mixed = lam * X + (1.0 - lam) * X[indices]
    y_mixed = lam * y + (1.0 - lam) * y[indices]

    return X_mixed, y_mixed


# ---------------------------------------------------------------------------
# Signal-level mixup (raw EEG: batch x channels x samples)
# ---------------------------------------------------------------------------

def mixup_signal_batch(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Mixup to raw EEG signal batches.

    Same algorithm as mixup_batch but operates on 3D signal data
    (batch, channels, samples) rather than 2D feature matrices.

    The mixing is applied uniformly across all channels and time points
    for each sample pair, preserving the temporal and spatial structure.

    Args:
        X: Raw EEG signals, shape (batch, n_channels, n_samples). Float.
        y: One-hot labels, shape (batch, n_classes). Float.
        alpha: Beta distribution parameter (see mixup_batch).

    Returns:
        (X_mixed, y_mixed): Same shapes as input.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    batch_size = X.shape[0]

    if batch_size <= 1:
        return X.copy(), y.copy()

    if alpha <= 0.0:
        lam = 1.0
    else:
        lam = float(np.random.beta(alpha, alpha))

    lam = max(lam, 1.0 - lam)

    indices = np.random.permutation(batch_size)

    X_mixed = lam * X + (1.0 - lam) * X[indices]
    y_mixed = lam * y + (1.0 - lam) * y[indices]

    return X_mixed, y_mixed


# ---------------------------------------------------------------------------
# Loss: soft-label cross-entropy (required for mixup)
# ---------------------------------------------------------------------------

def mixup_cross_entropy(
    logits: "torch.Tensor",
    soft_labels: "torch.Tensor",
    class_weights: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Cross-entropy loss with soft (non-integer) labels.

    Standard nn.CrossEntropyLoss requires integer labels. Mixup produces
    soft probability distributions as targets. This function computes:
        loss = -sum(soft_labels * log_softmax(logits), dim=1).mean()

    Optionally applies per-class weighting for imbalanced datasets.

    Args:
        logits: Raw model output, shape (batch, n_classes). NOT softmaxed.
        soft_labels: Target distributions, shape (batch, n_classes).
                     Rows should sum to 1.0.
        class_weights: Optional per-class weights, shape (n_classes,).
                       Scales the loss contribution of each class.

    Returns:
        Scalar loss tensor with gradient attached.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for mixup_cross_entropy")

    log_probs = F.log_softmax(logits, dim=1)

    if class_weights is not None:
        # Expand weights to (1, n_classes) for broadcasting
        w = class_weights.unsqueeze(0)  # (1, n_classes)
        loss_per_sample = -torch.sum(w * soft_labels * log_probs, dim=1)
    else:
        loss_per_sample = -torch.sum(soft_labels * log_probs, dim=1)

    return loss_per_sample.mean()
