"""CSCL: Cross-Subject Contrastive Learning for EEG emotion recognition (#401).

Implements SimCLR-style contrastive learning adapted for cross-subject EEG
emotion recognition. The core idea: learn subject-invariant representations
by pulling same-emotion samples together and pushing different-emotion samples
apart in an embedding space, regardless of which subject they came from.

Contrastive learning framework
------------------------------
1. EEG augmentations create two views of the same trial (positive pair).
2. Cross-subject samples with the same emotion label serve as additional
   positive pairs.
3. Different-emotion samples serve as negative pairs.
4. NT-Xent loss (Normalized Temperature-scaled Cross-Entropy) trains the
   projection head to map all positives close and negatives far apart.

EEG augmentations
-----------------
  temporal_crop     : random crop of 80-100% signal length, zero-pad back
  amplitude_jitter  : multiply by Uniform(0.8, 1.2) per channel
  channel_dropout   : zero out one random channel with probability 0.25
  frequency_perturb : add band-limited noise in a random frequency band

Projection head (MLP)
---------------------
  Linear(feature_dim, 64) + ReLU
  Linear(64, projection_dim)  -- maps to contrastive embedding space

After contrastive pre-training, the feature extractor produces subject-invariant
embeddings usable for downstream emotion classification without per-subject
fine-tuning.

References:
  Chen et al. (2020) -- A Simple Framework for Contrastive Learning (SimCLR)
  Shen et al. (2022) -- Contrastive Learning of Subject-Invariant EEG Representations
  Mohsenvand et al. (2020) -- Contrastive Representation Learning for EEG
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
class CSCLConfig:
    """Configuration for cross-subject contrastive learning."""

    # Input
    n_channels: int = 4
    n_samples: int = 1024
    fs: float = 256.0
    feature_dim: int = 20  # 5 bands x 4 channels

    # Projection head
    projection_hidden: int = 64
    projection_dim: int = 32  # contrastive embedding dimension

    # Contrastive loss
    temperature: float = 0.07  # NT-Xent temperature (SimCLR default: 0.07)
    n_negatives: int = 16  # negatives per positive pair

    # Augmentation probabilities
    aug_temporal_crop: float = 0.8
    aug_amplitude_jitter: float = 0.8
    aug_channel_dropout: float = 0.25
    aug_frequency_perturb: float = 0.5

    # Augmentation parameters
    crop_ratio_min: float = 0.8
    jitter_range: Tuple[float, float] = (0.8, 1.2)
    perturb_amplitude: float = 0.1

    # Training
    n_classes: int = 6
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 200


@dataclass
class ContrastivePair:
    """A pair of augmented views for contrastive learning."""

    anchor: np.ndarray = field(default_factory=lambda: np.zeros(20))
    positive: np.ndarray = field(default_factory=lambda: np.zeros(20))
    label: int = 0
    subject_id: str = ""
    augmentations_applied: List[str] = field(default_factory=list)


@dataclass
class RepresentationQualityResult:
    """Result of representation quality evaluation."""

    alignment_score: float = 0.0      # positive pair cosine similarity
    uniformity_score: float = 0.0     # negative pair spread
    silhouette_score: float = 0.0     # cluster quality
    cross_subject_consistency: float = 0.0
    n_samples: int = 0
    n_classes: int = 0
    embedding_dim: int = 0
    evaluated_at: float = 0.0


# -- Feature extraction -------------------------------------------------------


def _band_power(signal: np.ndarray, fs: float, flo: float, fhi: float) -> float:
    """Compute band power via Welch PSD."""
    nperseg = min(len(signal), int(fs * 2))
    if nperseg < 4:
        return 0.0
    f, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = np.logical_and(f >= flo, f <= fhi)
    if not idx.any():
        return 0.0
    return float(np.sum(psd[idx]))


def _extract_features(signals: np.ndarray, fs: float) -> np.ndarray:
    """Extract band-power features from multichannel EEG.

    Returns 20-dim vector: 5 bands x 4 channels.
    """
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
    n_ch = min(signals.shape[0], 4)
    feats = []
    for ch in range(n_ch):
        for flo, fhi in bands:
            feats.append(_band_power(signals[ch], fs, flo, fhi))
    while len(feats) < 20:
        feats.append(0.0)
    return np.array(feats[:20], dtype=np.float64)


# -- EEG augmentations --------------------------------------------------------


def _augment_temporal_crop(
    signals: np.ndarray,
    crop_ratio_min: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Random temporal crop and zero-pad back to original length."""
    if rng is None:
        rng = np.random.default_rng()
    n_samples = signals.shape[-1]
    crop_len = int(n_samples * rng.uniform(crop_ratio_min, 1.0))
    start = rng.integers(0, max(n_samples - crop_len, 1) + 1)
    cropped = signals[..., start:start + crop_len]
    result = np.zeros_like(signals)
    result[..., :cropped.shape[-1]] = cropped
    return result


def _augment_amplitude_jitter(
    signals: np.ndarray,
    jitter_range: Tuple[float, float] = (0.8, 1.2),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Multiply each channel by a random scale factor."""
    if rng is None:
        rng = np.random.default_rng()
    if signals.ndim == 1:
        scale = rng.uniform(*jitter_range)
        return signals * scale
    result = signals.copy()
    for ch in range(signals.shape[0]):
        scale = rng.uniform(*jitter_range)
        result[ch] *= scale
    return result


def _augment_channel_dropout(
    signals: np.ndarray,
    prob: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Zero out one random channel with given probability."""
    if rng is None:
        rng = np.random.default_rng()
    if signals.ndim == 1:
        return signals  # can't drop channel from 1D
    if rng.random() > prob:
        return signals.copy()
    result = signals.copy()
    ch = rng.integers(0, signals.shape[0])
    result[ch] = 0.0
    return result


def _augment_frequency_perturb(
    signals: np.ndarray,
    amplitude: float = 0.1,
    fs: float = 256.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add band-limited noise in a random frequency band."""
    if rng is None:
        rng = np.random.default_rng()
    n_samples = signals.shape[-1]
    t = np.arange(n_samples) / fs
    # Random center frequency between 2 and 40 Hz
    freq = rng.uniform(2.0, 40.0)
    noise = amplitude * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    if signals.ndim == 1:
        return signals + noise
    result = signals.copy()
    # Add noise to a random channel
    ch = rng.integers(0, signals.shape[0])
    result[ch] += noise
    return result


def _apply_augmentations(
    signals: np.ndarray,
    config: CSCLConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Apply random augmentations to EEG signals.

    Returns augmented signals and list of augmentation names applied.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = signals.copy()
    applied: List[str] = []

    if rng.random() < config.aug_temporal_crop:
        result = _augment_temporal_crop(result, config.crop_ratio_min, rng)
        applied.append("temporal_crop")

    if rng.random() < config.aug_amplitude_jitter:
        result = _augment_amplitude_jitter(result, config.jitter_range, rng)
        applied.append("amplitude_jitter")

    if rng.random() < config.aug_channel_dropout:
        result = _augment_channel_dropout(result, 1.0, rng)  # prob=1.0 since we already checked
        applied.append("channel_dropout")

    if rng.random() < config.aug_frequency_perturb:
        result = _augment_frequency_perturb(result, config.perturb_amplitude, config.fs, rng)
        applied.append("frequency_perturb")

    return result, applied


# -- Contrastive learning functions -------------------------------------------


def compute_nt_xent_loss(
    anchor: np.ndarray,
    positive: np.ndarray,
    negatives: np.ndarray,
    temperature: float = 0.07,
) -> float:
    """Compute NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    L = -log( exp(sim(z_i, z_j) / tau) / sum_k(exp(sim(z_i, z_k) / tau)) )

    where z_j is the positive pair and z_k are all other samples (including positive).

    Args:
        anchor: (d,) anchor embedding.
        positive: (d,) positive pair embedding.
        negatives: (n_neg, d) negative embeddings.
        temperature: Temperature scaling factor (tau).

    Returns:
        NT-Xent loss value (float). Lower = better alignment.
    """
    anchor = np.asarray(anchor, dtype=np.float64).ravel()
    positive = np.asarray(positive, dtype=np.float64).ravel()
    negatives = np.atleast_2d(negatives)

    # Normalize to unit sphere
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(n, 1e-8)

    anchor_n = _norm(anchor[np.newaxis, :])[0]
    positive_n = _norm(positive[np.newaxis, :])[0]
    negatives_n = _norm(negatives)

    # Cosine similarities
    sim_pos = float(np.dot(anchor_n, positive_n)) / temperature
    sim_negs = negatives_n @ anchor_n / temperature

    # Log-sum-exp for numerical stability
    all_sims = np.concatenate([[sim_pos], sim_negs])
    max_sim = np.max(all_sims)
    log_sum_exp = max_sim + np.log(np.sum(np.exp(all_sims - max_sim)))

    loss = -sim_pos + log_sum_exp
    return float(loss)


def generate_positive_pair(
    signals: np.ndarray,
    config: Optional[CSCLConfig] = None,
    seed: Optional[int] = None,
) -> ContrastivePair:
    """Generate a positive pair by applying different augmentations to the same trial.

    Args:
        signals: (n_channels, n_samples) or (n_samples,) raw EEG.
        config: CSCL configuration.
        seed: Random seed for reproducibility.

    Returns:
        ContrastivePair with anchor and positive views (as feature vectors).
    """
    if config is None:
        config = CSCLConfig()

    rng = np.random.default_rng(seed)

    # View 1: apply augmentations
    view1, augs1 = _apply_augmentations(signals, config, rng)
    # View 2: apply different augmentations
    view2, augs2 = _apply_augmentations(signals, config, rng)

    # Extract features from both views
    feat1 = _extract_features(view1, config.fs)
    feat2 = _extract_features(view2, config.fs)

    return ContrastivePair(
        anchor=feat1,
        positive=feat2,
        augmentations_applied=augs1 + augs2,
    )


def generate_negative_pairs(
    anchor_features: np.ndarray,
    anchor_label: int,
    all_features: np.ndarray,
    all_labels: np.ndarray,
    n_negatives: int = 16,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample negative pairs (different-emotion samples).

    Args:
        anchor_features: (d,) anchor feature vector (for reference, not used in sampling).
        anchor_label: Emotion label of anchor.
        all_features: (n, d) pool of all features to sample from.
        all_labels: (n,) labels for the feature pool.
        n_negatives: Number of negatives to sample.
        seed: Random seed.

    Returns:
        (n_negatives, d) matrix of negative feature vectors.
    """
    rng = np.random.default_rng(seed)
    all_features = np.atleast_2d(all_features)
    all_labels = np.asarray(all_labels).ravel()

    # Indices of different-emotion samples
    neg_mask = all_labels != anchor_label
    neg_indices = np.where(neg_mask)[0]

    if len(neg_indices) == 0:
        # No negatives available; return random noise
        d = anchor_features.shape[-1] if anchor_features.ndim > 0 else 20
        return rng.standard_normal((n_negatives, d))

    # Sample with replacement if fewer negatives than requested
    chosen = rng.choice(neg_indices, size=n_negatives, replace=len(neg_indices) < n_negatives)
    return all_features[chosen]


def create_cscl_config(
    n_channels: int = 4,
    temperature: float = 0.07,
    projection_dim: int = 32,
    n_classes: int = 6,
    **kwargs,
) -> CSCLConfig:
    """Create a CSCL training configuration.

    Args:
        n_channels: Number of EEG channels.
        temperature: NT-Xent temperature parameter.
        projection_dim: Contrastive embedding dimension.
        n_classes: Number of emotion classes.
        **kwargs: Additional overrides.

    Returns:
        CSCLConfig instance.
    """
    config = CSCLConfig(
        n_channels=n_channels,
        temperature=temperature,
        projection_dim=projection_dim,
        n_classes=n_classes,
    )
    for key, val in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, val)
    return config


def evaluate_representation_quality(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: Optional[np.ndarray] = None,
) -> RepresentationQualityResult:
    """Evaluate quality of learned representations.

    Measures:
    - Alignment: mean cosine similarity of positive pairs (same-label).
    - Uniformity: negative of mean pairwise distance (more spread = better).
    - Silhouette: cluster separability score.
    - Cross-subject consistency: same-class similarity across subjects.

    Args:
        features: (n, d) feature/embedding matrix.
        labels: (n,) integer class labels.
        subject_ids: (n,) optional subject identifiers for cross-subject metric.

    Returns:
        RepresentationQualityResult with all metrics.
    """
    features = np.atleast_2d(features)
    labels = np.asarray(labels).ravel()
    n = features.shape[0]
    d = features.shape[1]

    if n < 2:
        return RepresentationQualityResult(
            n_samples=n,
            n_classes=len(np.unique(labels)),
            embedding_dim=d,
            evaluated_at=time.time(),
        )

    # Normalize features for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    feat_normed = features / norms

    # Alignment: mean cosine similarity of same-label pairs
    alignment_sum = 0.0
    alignment_count = 0
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        class_feats = feat_normed[mask]
        nc = class_feats.shape[0]
        if nc < 2:
            continue
        # Pairwise cosine similarities within class
        sim_matrix = class_feats @ class_feats.T
        # Sum upper triangle (exclude diagonal)
        for i in range(nc):
            for j in range(i + 1, nc):
                alignment_sum += sim_matrix[i, j]
                alignment_count += 1
    alignment = alignment_sum / max(alignment_count, 1)

    # Uniformity: mean pairwise distance (using cosine distance)
    # Higher spread (lower cosine similarity across all pairs) = better uniformity
    all_sim = feat_normed @ feat_normed.T
    n_pairs = 0
    sim_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                sim_sum += all_sim[i, j]
                n_pairs += 1
    uniformity = 1.0 - (sim_sum / max(n_pairs, 1))  # higher = more uniform

    # Silhouette score (simplified)
    silhouette_scores = []
    for i in range(n):
        own_label = labels[i]
        # Mean intra-cluster distance
        same_mask = labels == own_label
        same_mask[i] = False
        if not same_mask.any():
            continue
        a_i = np.mean(np.linalg.norm(features[same_mask] - features[i], axis=1))
        # Mean nearest-cluster distance
        b_i = float("inf")
        for lbl in unique_labels:
            if lbl == own_label:
                continue
            other_mask = labels == lbl
            if not other_mask.any():
                continue
            mean_dist = np.mean(np.linalg.norm(features[other_mask] - features[i], axis=1))
            b_i = min(b_i, mean_dist)
        if b_i == float("inf"):
            continue
        s_i = (b_i - a_i) / max(a_i, b_i, 1e-8)
        silhouette_scores.append(s_i)
    silhouette = float(np.mean(silhouette_scores)) if silhouette_scores else 0.0

    # Cross-subject consistency
    cross_subj = 0.0
    if subject_ids is not None:
        subject_ids = np.asarray(subject_ids).ravel()
        unique_subjects = np.unique(subject_ids)
        if len(unique_subjects) >= 2:
            cross_sim_sum = 0.0
            cross_count = 0
            for lbl in unique_labels:
                lbl_mask = labels == lbl
                lbl_feats = feat_normed[lbl_mask]
                lbl_subjects = subject_ids[lbl_mask]
                for i in range(len(lbl_feats)):
                    for j in range(i + 1, len(lbl_feats)):
                        if lbl_subjects[i] != lbl_subjects[j]:
                            cross_sim_sum += float(np.dot(lbl_feats[i], lbl_feats[j]))
                            cross_count += 1
            cross_subj = cross_sim_sum / max(cross_count, 1)

    return RepresentationQualityResult(
        alignment_score=float(np.clip(alignment, -1, 1)),
        uniformity_score=float(np.clip(uniformity, 0, 2)),
        silhouette_score=float(np.clip(silhouette, -1, 1)),
        cross_subject_consistency=float(np.clip(cross_subj, -1, 1)),
        n_samples=n,
        n_classes=len(unique_labels),
        embedding_dim=d,
        evaluated_at=time.time(),
    )


def config_to_dict(config: CSCLConfig) -> Dict:
    """Convert CSCLConfig to a JSON-serializable dict."""
    return asdict(config)
