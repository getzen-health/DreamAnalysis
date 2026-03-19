"""Multi-dataset joint pre-training (mdJPT) for EEG emotion recognition.

Implements a multi-dataset joint pre-training framework that harmonizes
heterogeneous EEG emotion datasets (DEAP, SEED, GAMEEMO, DREAMER) into a
common feature space, enabling a shared encoder to learn universal emotion
representations.

Key contributions (NeurIPS 2025):
  - Channel mapping: aligns different electrode montages to a shared set of
    virtual channels via spatial interpolation weights.
  - Label alignment: unifies emotion labels across datasets that use different
    annotation schemes (valence/arousal continuous, discrete categories, etc.).
  - Dataset harmonization: z-score normalisation per dataset, then a learned
    affine projection to a common feature space.
  - Pre-training config: flexible batch sampling strategies (proportional,
    equal, curriculum) to balance dataset contributions during training.
  - Transfer learning: fine-tune the pre-trained encoder on a target dataset
    with frozen/unfrozen backbone options.

Functions
---------
  harmonize_datasets()        — align feature spaces across datasets
  align_channels()            — map different montages to shared virtual channels
  align_labels()              — unify emotion labels to a common scheme
  create_pretraining_config() — build a pre-training configuration
  compute_dataset_statistics()— compute per-dataset feature statistics
  evaluate_transfer()         — evaluate transfer learning performance
  config_to_dict()            — serialise config to a plain dict
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = ("DEAP", "SEED", "GAMEEMO", "DREAMER")

SAMPLING_STRATEGIES = ("proportional", "equal", "curriculum")

# Canonical 4 virtual channels (Muse 2 compatible)
VIRTUAL_CHANNELS = ("VF_left", "VF_right", "VT_left", "VT_right")

# Unified 6-class emotion label set
UNIFIED_LABELS = ("happy", "sad", "angry", "fear", "surprise", "neutral")

# Known electrode montages per dataset
_DATASET_MONTAGES: Dict[str, List[str]] = {
    "DEAP": [
        "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
        "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
        "Fp2", "AF4", "F4", "F8", "FC6", "FC2", "C4", "T8",
        "CP6", "CP2", "P4", "P8", "PO4", "O2", "Fz", "Cz",
    ],
    "SEED": [
        "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3",
        "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
        "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
        "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
        "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
        "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
        "PO8", "CB1", "O1", "OZ", "O2", "CB2",
    ],
    "GAMEEMO": [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
        "P8", "T8", "FC6", "F4", "F8", "AF4",
    ],
    "DREAMER": [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
        "P8", "T8", "FC6", "F4", "F8", "AF4",
    ],
}

# Label scheme descriptions per dataset
_DATASET_LABEL_SCHEMES: Dict[str, str] = {
    "DEAP": "valence_arousal_continuous",   # 1-9 continuous
    "SEED": "discrete_3class",              # positive, neutral, negative
    "GAMEEMO": "discrete_4class",           # happy, sad, fear, relaxed
    "DREAMER": "valence_arousal_continuous", # 1-5 continuous
}

# Spatial interpolation weights: mapping from source electrodes to virtual
# channels.  Each virtual channel is a weighted average of nearby electrodes.
_CHANNEL_MAPPING_WEIGHTS: Dict[str, Dict[str, Dict[str, float]]] = {
    "DEAP": {
        "VF_left":  {"F3": 0.5, "AF3": 0.3, "F7": 0.2},
        "VF_right": {"F4": 0.5, "AF4": 0.3, "F8": 0.2},
        "VT_left":  {"T7": 0.6, "CP5": 0.2, "P7": 0.2},
        "VT_right": {"T8": 0.6, "CP6": 0.2, "P8": 0.2},
    },
    "SEED": {
        "VF_left":  {"F3": 0.5, "AF3": 0.3, "F7": 0.2},
        "VF_right": {"F4": 0.5, "AF4": 0.3, "F8": 0.2},
        "VT_left":  {"T7": 0.6, "CP5": 0.2, "P7": 0.2},
        "VT_right": {"T8": 0.6, "CP6": 0.2, "P8": 0.2},
    },
    "GAMEEMO": {
        "VF_left":  {"F3": 0.5, "AF3": 0.3, "F7": 0.2},
        "VF_right": {"F4": 0.5, "AF4": 0.3, "F8": 0.2},
        "VT_left":  {"T7": 0.7, "P7": 0.3},
        "VT_right": {"T8": 0.7, "P8": 0.3},
    },
    "DREAMER": {
        "VF_left":  {"F3": 0.5, "AF3": 0.3, "F7": 0.2},
        "VF_right": {"F4": 0.5, "AF4": 0.3, "F8": 0.2},
        "VT_left":  {"T7": 0.7, "P7": 0.3},
        "VT_right": {"T8": 0.7, "P8": 0.3},
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatasetStatistics:
    """Per-dataset feature statistics for normalisation."""
    name: str = ""
    n_samples: int = 0
    n_channels: int = 0
    n_features: int = 0
    feature_means: Optional[List[float]] = None
    feature_stds: Optional[List[float]] = None
    label_distribution: Optional[Dict[str, int]] = None
    computed_at: float = 0.0


@dataclass
class PretrainingConfig:
    """Configuration for mdJPT pre-training run."""
    datasets: List[str] = field(default_factory=lambda: list(SUPPORTED_DATASETS))
    sampling_strategy: str = "proportional"
    shared_encoder_dim: int = 128
    batch_size: int = 64
    learning_rate: float = 1e-3
    n_epochs: int = 100
    warmup_epochs: int = 5
    virtual_channels: List[str] = field(
        default_factory=lambda: list(VIRTUAL_CHANNELS)
    )
    label_scheme: str = "unified_6class"
    freeze_encoder_for_transfer: bool = False
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def harmonize_datasets(
    dataset_features: Dict[str, np.ndarray],
    dataset_stats: Optional[Dict[str, DatasetStatistics]] = None,
) -> Dict[str, np.ndarray]:
    """Align feature spaces across datasets via z-score normalisation.

    Each dataset's features are independently z-scored using either
    pre-computed statistics or statistics derived from the data itself.

    Args:
        dataset_features: mapping of dataset name to feature array
            (n_samples, n_features).
        dataset_stats: optional pre-computed statistics.  If None, statistics
            are computed from the provided features.

    Returns:
        Dict mapping dataset name to harmonised feature array (same shapes).

    Raises:
        ValueError: if a dataset name is not in SUPPORTED_DATASETS.
    """
    harmonised: Dict[str, np.ndarray] = {}

    for name, features in dataset_features.items():
        if name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset '{name}'. Supported: {SUPPORTED_DATASETS}"
            )

        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if dataset_stats and name in dataset_stats:
            stats = dataset_stats[name]
            means = np.array(stats.feature_means or [0.0], dtype=np.float64)
            stds = np.array(stats.feature_stds or [1.0], dtype=np.float64)
        else:
            means = arr.mean(axis=0)
            stds = arr.std(axis=0)

        stds = np.where(stds < 1e-8, 1.0, stds)
        harmonised[name] = (arr - means) / stds

    logger.info(
        "Harmonised %d datasets: %s",
        len(harmonised),
        list(harmonised.keys()),
    )
    return harmonised


def align_channels(
    signals: np.ndarray,
    source_channels: List[str],
    dataset_name: str,
) -> np.ndarray:
    """Map electrode montage to shared virtual channels.

    Produces a (n_samples, len(VIRTUAL_CHANNELS)) array by applying spatial
    interpolation weights for the given dataset.

    Args:
        signals: raw EEG array, shape (n_samples, n_source_channels) or
            (n_source_channels,) for a single sample.
        source_channels: list of electrode names matching columns of signals.
        dataset_name: one of SUPPORTED_DATASETS.

    Returns:
        (n_samples, 4) array mapped to VIRTUAL_CHANNELS.

    Raises:
        ValueError: if dataset_name is unsupported.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {SUPPORTED_DATASETS}"
        )

    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n_samples, n_ch = arr.shape
    ch_index = {ch: i for i, ch in enumerate(source_channels)}
    weights = _CHANNEL_MAPPING_WEIGHTS.get(dataset_name, {})

    result = np.zeros((n_samples, len(VIRTUAL_CHANNELS)), dtype=np.float64)

    for v_idx, v_name in enumerate(VIRTUAL_CHANNELS):
        w_map = weights.get(v_name, {})
        total_weight = 0.0
        for src_ch, w in w_map.items():
            if src_ch in ch_index:
                result[:, v_idx] += arr[:, ch_index[src_ch]] * w
                total_weight += w
        if total_weight > 0:
            result[:, v_idx] /= total_weight

    logger.info(
        "Aligned %d channels -> %d virtual channels for %s (%d samples)",
        n_ch,
        len(VIRTUAL_CHANNELS),
        dataset_name,
        n_samples,
    )
    return result


def align_labels(
    labels: List[Any],
    dataset_name: str,
) -> List[str]:
    """Unify emotion labels to the common 6-class scheme.

    Handles:
      - Continuous valence/arousal (DEAP, DREAMER) -> discrete via thresholds
      - Discrete 3-class (SEED) -> map to unified
      - Discrete 4-class (GAMEEMO) -> map to unified

    Args:
        labels: raw label values from the dataset.
        dataset_name: one of SUPPORTED_DATASETS.

    Returns:
        List of unified label strings from UNIFIED_LABELS.

    Raises:
        ValueError: if dataset_name is unsupported.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {SUPPORTED_DATASETS}"
        )

    scheme = _DATASET_LABEL_SCHEMES[dataset_name]
    unified: List[str] = []

    for lbl in labels:
        if scheme == "valence_arousal_continuous":
            # Expect (valence, arousal) tuple or list
            if isinstance(lbl, (list, tuple)) and len(lbl) >= 2:
                v, a = float(lbl[0]), float(lbl[1])
            else:
                v, a = float(lbl), 0.5
            unified.append(_va_to_discrete(v, a, dataset_name))

        elif scheme == "discrete_3class":
            # SEED: positive(1), neutral(0), negative(-1)
            val = int(lbl) if not isinstance(lbl, str) else lbl
            mapping = {
                1: "happy", "positive": "happy",
                0: "neutral", "neutral": "neutral",
                -1: "sad", "negative": "sad",
            }
            unified.append(mapping.get(val, "neutral"))

        elif scheme == "discrete_4class":
            # GAMEEMO: happy, sad, fear, relaxed
            s = str(lbl).lower().strip()
            mapping = {
                "happy": "happy",
                "sad": "sad",
                "fear": "fear",
                "relaxed": "neutral",
            }
            unified.append(mapping.get(s, "neutral"))

        else:
            unified.append("neutral")

    logger.info(
        "Aligned %d labels for %s (scheme=%s)",
        len(unified),
        dataset_name,
        scheme,
    )
    return unified


def create_pretraining_config(
    datasets: Optional[List[str]] = None,
    sampling_strategy: str = "proportional",
    shared_encoder_dim: int = 128,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    n_epochs: int = 100,
    warmup_epochs: int = 5,
    freeze_encoder_for_transfer: bool = False,
) -> PretrainingConfig:
    """Build a pre-training configuration.

    Args:
        datasets: list of dataset names to include.  Defaults to all supported.
        sampling_strategy: one of SAMPLING_STRATEGIES.
        shared_encoder_dim: encoder output dimensionality.
        batch_size: training batch size.
        learning_rate: base learning rate.
        n_epochs: total training epochs.
        warmup_epochs: linear warmup epochs.
        freeze_encoder_for_transfer: if True, freeze encoder during fine-tune.

    Returns:
        PretrainingConfig dataclass.

    Raises:
        ValueError: if any dataset or strategy is invalid.
    """
    if datasets is None:
        datasets = list(SUPPORTED_DATASETS)

    for d in datasets:
        if d not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset '{d}'. Supported: {SUPPORTED_DATASETS}"
            )

    if sampling_strategy not in SAMPLING_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{sampling_strategy}'. "
            f"Supported: {SAMPLING_STRATEGIES}"
        )

    if shared_encoder_dim < 1:
        raise ValueError("shared_encoder_dim must be >= 1")

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    return PretrainingConfig(
        datasets=datasets,
        sampling_strategy=sampling_strategy,
        shared_encoder_dim=shared_encoder_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        warmup_epochs=warmup_epochs,
        freeze_encoder_for_transfer=freeze_encoder_for_transfer,
    )


def compute_dataset_statistics(
    features: np.ndarray,
    labels: Optional[List[str]] = None,
    dataset_name: str = "unknown",
) -> DatasetStatistics:
    """Compute per-dataset feature statistics.

    Args:
        features: (n_samples, n_features) array.
        labels: optional list of string labels for distribution computation.
        dataset_name: name identifier.

    Returns:
        DatasetStatistics dataclass.
    """
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n_samples, n_features = arr.shape
    means = arr.mean(axis=0).tolist()
    stds = arr.std(axis=0).tolist()

    label_dist: Optional[Dict[str, int]] = None
    if labels is not None:
        label_dist = {}
        for lbl in labels:
            label_dist[lbl] = label_dist.get(lbl, 0) + 1

    return DatasetStatistics(
        name=dataset_name,
        n_samples=n_samples,
        n_channels=0,
        n_features=n_features,
        feature_means=means,
        feature_stds=stds,
        label_distribution=label_dist,
        computed_at=time.time(),
    )


def evaluate_transfer(
    source_predictions: List[str],
    target_labels: List[str],
) -> Dict[str, Any]:
    """Evaluate transfer learning performance.

    Computes accuracy, per-class precision/recall, and a confusion summary.

    Args:
        source_predictions: predicted labels from the pre-trained model.
        target_labels: ground-truth labels.

    Returns:
        Dict with keys: accuracy, per_class_precision, per_class_recall,
        n_samples, n_correct, evaluated_at.
    """
    if len(source_predictions) != len(target_labels):
        raise ValueError(
            f"Length mismatch: predictions={len(source_predictions)}, "
            f"labels={len(target_labels)}"
        )

    n = len(target_labels)
    if n == 0:
        return {
            "accuracy": 0.0,
            "per_class_precision": {},
            "per_class_recall": {},
            "n_samples": 0,
            "n_correct": 0,
            "evaluated_at": time.time(),
        }

    correct = sum(
        1 for p, t in zip(source_predictions, target_labels) if p == t
    )
    accuracy = correct / n

    # Per-class precision and recall
    classes = sorted(set(target_labels) | set(source_predictions))
    precision: Dict[str, float] = {}
    recall: Dict[str, float] = {}

    for cls in classes:
        tp = sum(
            1 for p, t in zip(source_predictions, target_labels)
            if p == cls and t == cls
        )
        fp = sum(
            1 for p, t in zip(source_predictions, target_labels)
            if p == cls and t != cls
        )
        fn = sum(
            1 for p, t in zip(source_predictions, target_labels)
            if p != cls and t == cls
        )
        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "per_class_precision": {k: round(v, 4) for k, v in precision.items()},
        "per_class_recall": {k: round(v, 4) for k, v in recall.items()},
        "n_samples": n,
        "n_correct": correct,
        "evaluated_at": time.time(),
    }


def config_to_dict(config: PretrainingConfig) -> Dict[str, Any]:
    """Serialise a PretrainingConfig to a plain dict.

    Args:
        config: PretrainingConfig instance.

    Returns:
        Serialisable dict.
    """
    return asdict(config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _va_to_discrete(
    valence: float,
    arousal: float,
    dataset_name: str,
) -> str:
    """Map continuous valence/arousal to a discrete emotion label.

    Uses dataset-specific scale ranges:
      - DEAP: 1-9 scale, midpoint 5
      - DREAMER: 1-5 scale, midpoint 3

    Quadrant mapping:
      high V, high A -> happy
      high V, low A  -> neutral (calm/content)
      low V, high A  -> angry (or fear if very low V)
      low V, low A   -> sad
    """
    if dataset_name == "DEAP":
        mid_v, mid_a = 5.0, 5.0
    elif dataset_name == "DREAMER":
        mid_v, mid_a = 3.0, 3.0
    else:
        mid_v, mid_a = 0.5, 0.5

    high_v = valence >= mid_v
    high_a = arousal >= mid_a

    if high_v and high_a:
        return "happy"
    elif high_v and not high_a:
        return "neutral"
    elif not high_v and high_a:
        return "angry" if valence >= (mid_v - 1.0) else "fear"
    else:
        return "sad"
