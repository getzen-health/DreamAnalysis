"""Self-supervised EEG pretraining for few-shot calibration.

Self-supervised pretraining framework for 4-channel EEG that learns
generalizable representations from unlabeled data, then fine-tunes
with only 5-10 labeled samples per emotion class.

Pretext tasks:
  - Masked signal prediction: mask 15% of timepoints, predict from context
  - Contrastive learning: positive=same trial augmented, negative=different trial
  - Temporal order prediction: is segment A before or after segment B?

Few-shot calibration:
  After pretraining, compute optimal config for fine-tuning with 5-10
  labeled samples per emotion class.

Data augmentation for pretraining:
  Time shift, amplitude scaling, channel dropout, additive noise.

Functions:
  create_pretext_task()
  generate_masked_sample()
  generate_contrastive_pair()
  generate_temporal_order_task()
  compute_few_shot_config()
  pretraining_config_to_dict()

GitHub issue: #387
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---- Constants ---------------------------------------------------------------

DEFAULT_MASK_RATIO = 0.15
DEFAULT_N_CHANNELS = 4
DEFAULT_FS = 256
SUPPORTED_TASKS = ("masked_prediction", "contrastive", "temporal_order")
EMOTION_CLASSES = ("happy", "sad", "angry", "fear", "surprise", "neutral")


# ---- Data augmentation helpers for pretraining --------------------------------

def _time_shift(signal: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    """Shift signal in time by a random amount."""
    shift = rng.integers(-max_shift, max_shift + 1)
    return np.roll(signal, shift, axis=-1)


def _amplitude_scale(signal: np.ndarray, scale_range: Tuple[float, float],
                     rng: np.random.Generator) -> np.ndarray:
    """Scale signal amplitude by a random factor."""
    factor = rng.uniform(*scale_range)
    return signal * factor


def _channel_dropout(signal: np.ndarray, p_drop: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Zero out random channels."""
    if signal.ndim < 2:
        return signal
    out = signal.copy()
    n_channels = out.shape[0]
    for ch in range(n_channels):
        if rng.random() < p_drop:
            out[ch, :] = 0.0
    return out


def _additive_noise(signal: np.ndarray, snr_db: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise at a specified SNR."""
    sig_power = np.mean(signal ** 2)
    if sig_power < 1e-10:
        return signal.copy()
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(signal.shape) * np.sqrt(noise_power)
    return signal + noise


def _augment_eeg(
    signal: np.ndarray,
    rng: np.random.Generator,
    time_shift_samples: int = 25,
    amplitude_range: Tuple[float, float] = (0.8, 1.2),
    channel_drop_prob: float = 0.1,
    noise_snr_db: float = 25.0,
) -> np.ndarray:
    """Apply a random combination of augmentations for pretraining."""
    out = signal.copy()
    # Time shift
    out = _time_shift(out, time_shift_samples, rng)
    # Amplitude scaling
    out = _amplitude_scale(out, amplitude_range, rng)
    # Channel dropout
    out = _channel_dropout(out, channel_drop_prob, rng)
    # Additive noise
    out = _additive_noise(out, noise_snr_db, rng)
    return out


# ---- Pretext task functions ---------------------------------------------------

def generate_masked_sample(
    signal: np.ndarray,
    mask_ratio: float = DEFAULT_MASK_RATIO,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a masked signal prediction sample.

    Masks a fraction of timepoints and returns the masked signal,
    the mask itself, and the original signal as the target.

    Args:
        signal: EEG signal, shape (n_channels, n_samples) or (n_samples,).
        mask_ratio: Fraction of timepoints to mask (default 0.15).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            masked_signal: signal with masked timepoints set to 0
            mask: boolean array indicating masked positions
            target: original signal (prediction target)
            mask_ratio: actual mask ratio applied
            n_masked: number of masked timepoints
    """
    if signal.ndim == 1:
        n_samples = signal.shape[0]
    elif signal.ndim == 2:
        n_samples = signal.shape[1]
    else:
        raise ValueError(f"Expected 1-D or 2-D signal, got shape {signal.shape}")

    rng = np.random.default_rng(seed)
    mask_ratio = np.clip(mask_ratio, 0.0, 0.5)

    n_masked = max(1, int(round(n_samples * mask_ratio)))
    mask_indices = rng.choice(n_samples, size=n_masked, replace=False)
    mask = np.zeros(n_samples, dtype=bool)
    mask[mask_indices] = True

    masked_signal = signal.copy()
    if signal.ndim == 1:
        masked_signal[mask] = 0.0
    else:
        masked_signal[:, mask] = 0.0

    return {
        "masked_signal": masked_signal,
        "mask": mask,
        "target": signal.copy(),
        "mask_ratio": float(n_masked / n_samples),
        "n_masked": int(n_masked),
    }


def generate_contrastive_pair(
    signal: np.ndarray,
    negative_signal: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a contrastive learning pair.

    Positive pair: original signal and an augmented version of it.
    Negative: a different trial's signal (or generated random if not provided).

    Args:
        signal: EEG signal, shape (n_channels, n_samples) or (n_samples,).
        negative_signal: A different trial's signal for the negative example.
            If None, generates a random signal of the same shape.
        seed: Random seed.

    Returns:
        Dict with keys:
            anchor: original signal
            positive: augmented version of anchor
            negative: different trial signal
            label: 1 for positive pair, 0 for negative pair
            augmentations_applied: list of augmentation names
    """
    rng = np.random.default_rng(seed)

    anchor = signal.copy()

    # Create positive by augmenting anchor
    positive = _augment_eeg(signal, rng)

    # Create negative from different trial or random
    if negative_signal is not None:
        negative = negative_signal.copy()
    else:
        negative = rng.standard_normal(signal.shape) * np.std(signal)

    augmentations = ["time_shift", "amplitude_scale", "channel_dropout", "additive_noise"]

    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "label": 1,
        "augmentations_applied": augmentations,
    }


def generate_temporal_order_task(
    signal: np.ndarray,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a temporal order prediction task.

    Splits the signal into two halves, then either presents them in order
    (label=1) or reversed (label=0). The model must predict whether
    segment A comes before segment B.

    Args:
        signal: EEG signal, shape (n_channels, n_samples) or (n_samples,).
        seed: Random seed.

    Returns:
        Dict with keys:
            segment_a: first presented segment
            segment_b: second presented segment
            label: 1 if A is chronologically before B, 0 if reversed
            split_point: index where the signal was split
    """
    rng = np.random.default_rng(seed)

    if signal.ndim == 1:
        n_samples = signal.shape[0]
    else:
        n_samples = signal.shape[1]

    # Split roughly at the middle, with some jitter
    mid = n_samples // 2
    jitter = rng.integers(-max(1, n_samples // 8), max(1, n_samples // 8) + 1)
    split_point = max(1, min(n_samples - 1, mid + jitter))

    if signal.ndim == 1:
        first_half = signal[:split_point].copy()
        second_half = signal[split_point:].copy()
    else:
        first_half = signal[:, :split_point].copy()
        second_half = signal[:, split_point:].copy()

    # Randomly present in correct or reversed order
    in_order = bool(rng.random() > 0.5)

    if in_order:
        segment_a = first_half
        segment_b = second_half
        label = 1
    else:
        segment_a = second_half
        segment_b = first_half
        label = 0

    return {
        "segment_a": segment_a,
        "segment_b": segment_b,
        "label": label,
        "split_point": int(split_point),
    }


def create_pretext_task(
    signal: np.ndarray,
    task_type: str = "masked_prediction",
    negative_signal: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create a pretext task sample for self-supervised pretraining.

    Dispatches to the appropriate task generator based on task_type.

    Args:
        signal: EEG signal, shape (n_channels, n_samples) or (n_samples,).
        task_type: One of 'masked_prediction', 'contrastive', 'temporal_order'.
        negative_signal: Required for 'contrastive' task, optional otherwise.
        seed: Random seed.
        **kwargs: Extra arguments forwarded to the specific task generator.

    Returns:
        Dict with task-specific keys plus 'task_type'.
    """
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unknown task_type: {task_type!r}. Supported: {SUPPORTED_TASKS}"
        )

    if task_type == "masked_prediction":
        mask_ratio = kwargs.get("mask_ratio", DEFAULT_MASK_RATIO)
        result = generate_masked_sample(signal, mask_ratio=mask_ratio, seed=seed)
    elif task_type == "contrastive":
        result = generate_contrastive_pair(signal, negative_signal=negative_signal, seed=seed)
    elif task_type == "temporal_order":
        result = generate_temporal_order_task(signal, seed=seed)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    result["task_type"] = task_type
    return result


# ---- Few-shot calibration config ---------------------------------------------

@dataclass
class FewShotConfig:
    """Configuration for few-shot fine-tuning after pretraining."""
    n_samples_per_class: int = 5
    n_classes: int = 6
    learning_rate: float = 1e-3
    n_epochs: int = 50
    batch_size: int = 4
    freeze_encoder: bool = True
    augment_few_shot: bool = True
    augmentation_factor: int = 10
    class_names: List[str] = field(default_factory=lambda: list(EMOTION_CLASSES))
    total_training_samples: int = 0
    estimated_accuracy_range: Tuple[float, float] = (0.0, 0.0)


def compute_few_shot_config(
    n_samples_per_class: int = 5,
    n_classes: int = 6,
    n_channels: int = DEFAULT_N_CHANNELS,
    fs: int = DEFAULT_FS,
    epoch_duration: float = 4.0,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute optimal few-shot calibration configuration.

    Given the number of labeled samples available per class, computes
    recommended hyperparameters for fine-tuning a pretrained encoder.

    Args:
        n_samples_per_class: Labeled samples available per class (5-10).
        n_classes: Number of emotion classes.
        n_channels: Number of EEG channels.
        fs: Sampling rate in Hz.
        epoch_duration: Duration of each EEG epoch in seconds.
        class_names: Names of emotion classes.

    Returns:
        Dict with recommended config parameters and estimated accuracy.
    """
    n_samples_per_class = max(1, min(n_samples_per_class, 100))
    n_classes = max(2, min(n_classes, 20))

    if class_names is None:
        class_names = list(EMOTION_CLASSES[:n_classes])

    total_labeled = n_samples_per_class * n_classes

    # Heuristics for few-shot fine-tuning
    # With very few samples, freeze encoder and only train classifier head
    freeze_encoder = n_samples_per_class <= 10
    # More augmentation when fewer samples
    augmentation_factor = max(5, min(20, 100 // n_samples_per_class))
    total_with_augmentation = total_labeled * augmentation_factor if n_samples_per_class <= 10 else total_labeled

    # Learning rate: smaller when fewer samples to avoid overfitting
    if n_samples_per_class <= 5:
        lr = 5e-4
    elif n_samples_per_class <= 10:
        lr = 1e-3
    else:
        lr = 2e-3

    # Epochs: more epochs when fewer total samples
    n_epochs = max(20, min(100, 1000 // max(1, total_labeled)))

    # Batch size: don't exceed total samples
    batch_size = min(8, total_labeled)

    # Estimated accuracy range (rough heuristic)
    # Based on literature: 5 samples -> ~60-70%, 10 samples -> ~70-80%
    base_acc = 0.45 + 0.03 * min(n_samples_per_class, 20)
    acc_low = max(1.0 / n_classes, base_acc - 0.05)
    acc_high = min(0.95, base_acc + 0.10)

    config = FewShotConfig(
        n_samples_per_class=n_samples_per_class,
        n_classes=n_classes,
        learning_rate=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        freeze_encoder=freeze_encoder,
        augment_few_shot=n_samples_per_class <= 10,
        augmentation_factor=augmentation_factor,
        class_names=class_names[:n_classes],
        total_training_samples=total_with_augmentation,
        estimated_accuracy_range=(acc_low, acc_high),
    )

    return {
        "config": config,
        "n_samples_per_class": config.n_samples_per_class,
        "n_classes": config.n_classes,
        "learning_rate": config.learning_rate,
        "n_epochs": config.n_epochs,
        "batch_size": config.batch_size,
        "freeze_encoder": config.freeze_encoder,
        "augment_few_shot": config.augment_few_shot,
        "augmentation_factor": config.augmentation_factor,
        "class_names": config.class_names,
        "total_training_samples": config.total_training_samples,
        "estimated_accuracy_range": list(config.estimated_accuracy_range),
        "n_channels": n_channels,
        "fs": fs,
        "epoch_duration_s": epoch_duration,
        "samples_per_epoch": int(fs * epoch_duration),
    }


# ---- Pretraining config serialization ----------------------------------------

@dataclass
class PretrainingConfig:
    """Full pretraining configuration."""
    task_types: List[str] = field(default_factory=lambda: list(SUPPORTED_TASKS))
    mask_ratio: float = DEFAULT_MASK_RATIO
    n_channels: int = DEFAULT_N_CHANNELS
    fs: int = DEFAULT_FS
    epoch_duration: float = 4.0
    augmentation_params: Dict[str, Any] = field(default_factory=lambda: {
        "time_shift_samples": 25,
        "amplitude_range": [0.8, 1.2],
        "channel_drop_prob": 0.1,
        "noise_snr_db": 25.0,
    })
    n_pretraining_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    temperature: float = 0.1


def pretraining_config_to_dict(
    config: Optional[PretrainingConfig] = None,
) -> Dict[str, Any]:
    """Convert pretraining configuration to a JSON-serializable dictionary.

    Args:
        config: A PretrainingConfig instance. Uses defaults if None.

    Returns:
        Dict with all config fields as native Python types.
    """
    if config is None:
        config = PretrainingConfig()

    return {
        "task_types": list(config.task_types),
        "mask_ratio": float(config.mask_ratio),
        "n_channels": int(config.n_channels),
        "fs": int(config.fs),
        "epoch_duration_s": float(config.epoch_duration),
        "samples_per_epoch": int(config.fs * config.epoch_duration),
        "augmentation_params": dict(config.augmentation_params),
        "n_pretraining_epochs": int(config.n_pretraining_epochs),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "temperature": float(config.temperature),
        "supported_tasks": list(SUPPORTED_TASKS),
        "emotion_classes": list(EMOTION_CLASSES),
    }
