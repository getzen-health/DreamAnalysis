"""DFF-Net: Domain adaptation with Few-shot Fine-tuning for cross-subject EEG
emotion recognition (#400).

Addresses the cross-subject transfer problem in EEG-based emotion recognition
by combining domain adaptation with few-shot fine-tuning:

1. Source domain: labeled EEG data from existing subjects
2. Target domain: new user with very few labeled samples (5-shot per class)

Architecture (conceptual)
-------------------------
Shared Feature Extractor:
  Input: (batch, 4, 1024) -- 4 channels x 4 sec @ 256 Hz
  Conv1d(4, 32, 64, stride=16) + BN + ReLU  -> (batch, 32, 64)
  Conv1d(32, 64, 16, stride=4)  + BN + ReLU -> (batch, 64, 16)
  AdaptiveAvgPool1d(1) -> Flatten            -> (batch, 64)

Domain Classifier (adversarial):
  Linear(64, 32) + ReLU
  Linear(32, 2)  -- source vs target domain

Emotion Classifier:
  Linear(64, 32) + ReLU
  Linear(32, n_classes)

Domain alignment via:
  - MMD (Maximum Mean Discrepancy) with RBF kernel
  - Or domain-adversarial training (GRL -- gradient reversal layer)

Few-shot adaptation:
  After domain alignment, fine-tune on k-shot per class from new user.
  Default: k=5 (5 labeled samples per emotion class).

References:
  Ganin et al. (2016) -- Domain-Adversarial Training of Neural Networks
  Gretton et al. (2012) -- A Kernel Two-Sample Test (MMD)
  Li et al. (2019) -- Multisource Transfer Learning for Cross-Subject EEG
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch

log = logging.getLogger(__name__)

# -- Emotion classes -----------------------------------------------------------

EMOTION_CLASSES = ["happy", "neutral", "sad", "angry", "fear", "calm"]

# -- Configuration dataclasses ------------------------------------------------


@dataclass
class DFFNetConfig:
    """Architecture and training configuration for DFF-Net."""

    # Feature extractor
    n_channels: int = 4
    n_samples: int = 1024
    fs: float = 256.0
    feature_dim: int = 64

    # Domain classifier
    domain_hidden: int = 32
    n_domains: int = 2  # source + target

    # Emotion classifier
    emotion_hidden: int = 32
    n_classes: int = 6

    # Domain adaptation
    adaptation_method: str = "mmd"  # "mmd" or "adversarial"
    mmd_kernel: str = "rbf"
    mmd_bandwidth: float = 1.0
    lambda_domain: float = 0.1  # weight for domain loss

    # Few-shot fine-tuning
    k_shot: int = 5  # samples per class
    finetune_lr: float = 0.001
    finetune_epochs: int = 20

    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100


@dataclass
class DomainAdaptationResult:
    """Result of domain discrepancy evaluation."""

    mmd_score: float = 0.0
    domain_accuracy: float = 0.5
    alignment_score: float = 0.0
    source_mean_norm: float = 0.0
    target_mean_norm: float = 0.0
    n_source_samples: int = 0
    n_target_samples: int = 0
    feature_dim: int = 0
    evaluated_at: float = 0.0


@dataclass
class FewShotResult:
    """Result of few-shot adaptation."""

    adapted: bool = False
    k_shot: int = 5
    classes_seen: List[str] = field(default_factory=list)
    samples_per_class: Dict[str, int] = field(default_factory=dict)
    pre_adaptation_accuracy: float = 0.0
    post_adaptation_accuracy: float = 0.0
    improvement: float = 0.0
    adapted_at: float = 0.0


# -- Helper functions ---------------------------------------------------------


def _extract_band_powers(signal: np.ndarray, fs: float) -> np.ndarray:
    """Extract band powers from a single-channel EEG signal.

    Returns 5-element array: [delta, theta, alpha, beta, gamma].
    """
    bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
    nperseg = min(len(signal), int(fs * 2))
    if nperseg < 4:
        return np.zeros(5)
    f, psd = welch(signal, fs=fs, nperseg=nperseg)
    powers = []
    for flo, fhi in bands:
        idx = np.logical_and(f >= flo, f <= fhi)
        band_power = np.sum(psd[idx]) if idx.any() else 0.0
        powers.append(float(band_power))
    return np.array(powers, dtype=np.float64)


def _extract_features(signals: np.ndarray, fs: float) -> np.ndarray:
    """Extract feature vector from multichannel EEG.

    Args:
        signals: (n_channels, n_samples) or (n_samples,)
        fs: sampling frequency

    Returns:
        Feature vector of shape (n_channels * 5,) -- 5 bands per channel.
        Padded to 20 features (4 channels * 5 bands) if fewer channels.
    """
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    n_ch = min(signals.shape[0], 4)
    feats = []
    for ch in range(n_ch):
        feats.extend(_extract_band_powers(signals[ch], fs).tolist())
    # Pad to 20 features if fewer channels
    while len(feats) < 20:
        feats.append(0.0)
    return np.array(feats[:20], dtype=np.float64)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0) -> float:
    """Compute RBF (Gaussian) kernel between two vectors.

    k(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))
    """
    diff = x - y
    return float(np.exp(-np.dot(diff, diff) / (2.0 * bandwidth ** 2)))


# -- Core functions -----------------------------------------------------------


def create_dff_net_config(
    n_channels: int = 4,
    n_classes: int = 6,
    adaptation_method: str = "mmd",
    k_shot: int = 5,
    **kwargs,
) -> DFFNetConfig:
    """Create a DFF-Net configuration.

    Args:
        n_channels: Number of EEG channels (default 4 for Muse 2).
        n_classes: Number of emotion classes.
        adaptation_method: "mmd" or "adversarial".
        k_shot: Number of labeled samples per class for few-shot.
        **kwargs: Additional config overrides.

    Returns:
        DFFNetConfig instance.
    """
    config = DFFNetConfig(
        n_channels=n_channels,
        n_classes=n_classes,
        adaptation_method=adaptation_method,
        k_shot=k_shot,
    )
    for key, val in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, val)
    return config


def compute_mmd(
    source_features: np.ndarray,
    target_features: np.ndarray,
    kernel: str = "rbf",
    bandwidth: float = 1.0,
) -> float:
    """Compute Maximum Mean Discrepancy between source and target distributions.

    MMD^2 = E[k(x_s, x_s')] + E[k(x_t, x_t')] - 2 * E[k(x_s, x_t)]

    Uses an unbiased estimator with O(n*m) complexity.

    Args:
        source_features: (n_source, d) feature matrix from source domain.
        target_features: (n_target, d) feature matrix from target domain.
        kernel: Kernel type ("rbf" or "linear").
        bandwidth: RBF kernel bandwidth (ignored for linear).

    Returns:
        MMD^2 estimate (non-negative float). Lower = more aligned.
    """
    source_features = np.atleast_2d(source_features)
    target_features = np.atleast_2d(target_features)

    n_s = source_features.shape[0]
    n_t = target_features.shape[0]

    if n_s < 2 or n_t < 2:
        return 0.0

    if kernel == "linear":
        # Linear MMD: compare means directly
        diff = source_features.mean(axis=0) - target_features.mean(axis=0)
        return float(np.dot(diff, diff))

    # RBF kernel MMD
    # E[k(x_s, x_s')]
    ss_sum = 0.0
    ss_count = 0
    for i in range(n_s):
        for j in range(i + 1, n_s):
            ss_sum += _rbf_kernel(source_features[i], source_features[j], bandwidth)
            ss_count += 1
    ss_mean = ss_sum / max(ss_count, 1)

    # E[k(x_t, x_t')]
    tt_sum = 0.0
    tt_count = 0
    for i in range(n_t):
        for j in range(i + 1, n_t):
            tt_sum += _rbf_kernel(target_features[i], target_features[j], bandwidth)
            tt_count += 1
    tt_mean = tt_sum / max(tt_count, 1)

    # E[k(x_s, x_t)]
    st_sum = 0.0
    for i in range(n_s):
        for j in range(n_t):
            st_sum += _rbf_kernel(source_features[i], target_features[j], bandwidth)
    st_mean = st_sum / (n_s * n_t)

    mmd_sq = ss_mean + tt_mean - 2.0 * st_mean
    return float(max(mmd_sq, 0.0))  # clamp small negatives from estimation noise


def compute_domain_discrepancy(
    source_features: np.ndarray,
    target_features: np.ndarray,
    config: Optional[DFFNetConfig] = None,
) -> DomainAdaptationResult:
    """Evaluate domain discrepancy between source and target feature sets.

    Computes MMD, a simple domain classification accuracy proxy, and
    overall alignment score.

    Args:
        source_features: (n_source, d) feature matrix.
        target_features: (n_target, d) feature matrix.
        config: Optional DFFNetConfig for kernel parameters.

    Returns:
        DomainAdaptationResult with all metrics.
    """
    if config is None:
        config = DFFNetConfig()

    source_features = np.atleast_2d(source_features)
    target_features = np.atleast_2d(target_features)

    mmd = compute_mmd(
        source_features,
        target_features,
        kernel=config.mmd_kernel,
        bandwidth=config.mmd_bandwidth,
    )

    # Domain classification accuracy proxy:
    # How well can we separate source vs target by nearest-centroid?
    s_mean = source_features.mean(axis=0)
    t_mean = target_features.mean(axis=0)

    correct = 0
    total = 0
    for feat in source_features:
        ds = np.linalg.norm(feat - s_mean)
        dt = np.linalg.norm(feat - t_mean)
        if ds <= dt:
            correct += 1
        total += 1
    for feat in target_features:
        ds = np.linalg.norm(feat - s_mean)
        dt = np.linalg.norm(feat - t_mean)
        if dt <= ds:
            correct += 1
        total += 1
    domain_acc = correct / max(total, 1)

    # Alignment score: 1.0 = perfectly aligned, 0.0 = maximally different
    # Uses MMD decay: alignment = exp(-mmd * 5)
    alignment = float(np.exp(-mmd * 5.0))

    return DomainAdaptationResult(
        mmd_score=mmd,
        domain_accuracy=domain_acc,
        alignment_score=alignment,
        source_mean_norm=float(np.linalg.norm(s_mean)),
        target_mean_norm=float(np.linalg.norm(t_mean)),
        n_source_samples=source_features.shape[0],
        n_target_samples=target_features.shape[0],
        feature_dim=source_features.shape[1],
        evaluated_at=time.time(),
    )


def setup_few_shot_adaptation(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    config: Optional[DFFNetConfig] = None,
) -> FewShotResult:
    """Setup and evaluate few-shot adaptation from source to target domain.

    Uses nearest-centroid classification with prototype refinement.
    Source prototypes are computed from all source data, then refined
    using the k-shot target samples (exponential moving average).

    Args:
        source_features: (n_source, d) source feature matrix.
        source_labels: (n_source,) integer labels for source.
        target_features: (n_target, d) target feature matrix (k-shot labeled).
        target_labels: (n_target,) integer labels for target.
        config: Optional DFFNetConfig.

    Returns:
        FewShotResult with adaptation metrics.
    """
    if config is None:
        config = DFFNetConfig()

    source_features = np.atleast_2d(source_features)
    target_features = np.atleast_2d(target_features)
    source_labels = np.asarray(source_labels).ravel()
    target_labels = np.asarray(target_labels).ravel()

    # Build source prototypes (class centroids)
    unique_labels = np.unique(np.concatenate([source_labels, target_labels]))
    source_prototypes: Dict[int, np.ndarray] = {}
    for label in unique_labels:
        mask = source_labels == label
        if mask.any():
            source_prototypes[label] = source_features[mask].mean(axis=0)
        else:
            # No source samples for this class; use overall source mean
            source_prototypes[label] = source_features.mean(axis=0)

    # Pre-adaptation accuracy on target using source prototypes
    pre_correct = 0
    for feat, label in zip(target_features, target_labels):
        distances = {
            lbl: np.linalg.norm(feat - proto)
            for lbl, proto in source_prototypes.items()
        }
        pred = min(distances, key=lambda k: distances[k])
        if pred == label:
            pre_correct += 1
    pre_acc = pre_correct / max(len(target_labels), 1)

    # Adapt prototypes using target k-shot samples (EMA, alpha=0.3)
    adapted_prototypes = {k: v.copy() for k, v in source_prototypes.items()}
    alpha_adapt = 0.3
    for feat, label in zip(target_features, target_labels):
        if label in adapted_prototypes:
            old = adapted_prototypes[label]
            adapted_prototypes[label] = (1 - alpha_adapt) * old + alpha_adapt * feat

    # Post-adaptation accuracy on target
    post_correct = 0
    for feat, label in zip(target_features, target_labels):
        distances = {
            lbl: np.linalg.norm(feat - proto)
            for lbl, proto in adapted_prototypes.items()
        }
        pred = min(distances, key=lambda k: distances[k])
        if pred == label:
            post_correct += 1
    post_acc = post_correct / max(len(target_labels), 1)

    # Compute per-class sample counts
    classes_seen = []
    samples_per_class: Dict[str, int] = {}
    for label in unique_labels:
        cls_name = EMOTION_CLASSES[label] if label < len(EMOTION_CLASSES) else f"class_{label}"
        classes_seen.append(cls_name)
        samples_per_class[cls_name] = int(np.sum(target_labels == label))

    return FewShotResult(
        adapted=True,
        k_shot=config.k_shot,
        classes_seen=classes_seen,
        samples_per_class=samples_per_class,
        pre_adaptation_accuracy=pre_acc,
        post_adaptation_accuracy=post_acc,
        improvement=post_acc - pre_acc,
        adapted_at=time.time(),
    )


def evaluate_cross_subject(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    config: Optional[DFFNetConfig] = None,
) -> Dict:
    """Full cross-subject evaluation pipeline.

    Combines domain discrepancy analysis with few-shot adaptation evaluation.

    Args:
        source_features: (n_source, d) source domain features.
        source_labels: (n_source,) source labels.
        target_features: (n_target, d) target domain features.
        target_labels: (n_target,) target labels.
        config: Optional DFFNetConfig.

    Returns:
        Dict with domain_adaptation and few_shot results plus summary.
    """
    if config is None:
        config = DFFNetConfig()

    da_result = compute_domain_discrepancy(source_features, target_features, config)
    fs_result = setup_few_shot_adaptation(
        source_features, source_labels, target_features, target_labels, config
    )

    return {
        "domain_adaptation": asdict(da_result),
        "few_shot": asdict(fs_result),
        "summary": {
            "mmd_score": da_result.mmd_score,
            "alignment_score": da_result.alignment_score,
            "pre_adaptation_accuracy": fs_result.pre_adaptation_accuracy,
            "post_adaptation_accuracy": fs_result.post_adaptation_accuracy,
            "improvement": fs_result.improvement,
            "n_source": da_result.n_source_samples,
            "n_target": da_result.n_target_samples,
        },
    }


def config_to_dict(config: DFFNetConfig) -> Dict:
    """Convert DFFNetConfig to a JSON-serializable dict."""
    return asdict(config)
