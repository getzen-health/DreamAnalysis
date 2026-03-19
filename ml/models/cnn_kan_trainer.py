"""CNN-KAN training pipeline configuration for 4-channel DEAP data.

Provides training pipeline config, data preprocessing, cross-validation
setup, and metric tracking for retraining the CNN-KAN model on 4-channel
DEAP EEG data (AF7, AF8, TP9, TP10 equivalent channels).

This module is config/pipeline-focused, not a training loop itself.
It generates configs that a separate training script consumes.

Data pipeline:
    Raw DEAP (32-ch, 8064 samples @ 128 Hz per trial)
    -> Extract 4 channels closest to Muse 2 positions
    -> Resample to 256 Hz
    -> Bandpass 1-45 Hz
    -> 4-sec windows (1024 samples @ 256 Hz)
    -> DE + PSD per band per channel -> pseudo-RGB feature image
    -> CNN-KAN model input

Cross-validation:
    - Leave-one-subject-out (LOSO): most rigorous, 32 folds
    - 5-fold stratified: faster iteration
    - Within-subject: for per-user fine-tuning baseline

DEAP 4-channel mapping:
    DEAP ch7  (AF3) -> approximate AF7 (left frontal)
    DEAP ch12 (AF4) -> approximate AF8 (right frontal)
    DEAP ch0  (Fp1) -> approximate TP9 (left temporal, rough)
    DEAP ch3  (Fp2) -> approximate TP10 (right temporal, rough)

    Note: DEAP doesn't have exact Muse 2 positions. These are the
    closest available channels. Accuracy will be lower than reported
    DEAP benchmarks which use all 32 channels.

Reference:
    Koelstra et al., "DEAP: A Database for Emotion Analysis using
    Physiological Signals" (2012). IEEE Trans. Affective Computing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# DEAP dataset parameters
DEAP_ORIGINAL_FS: float = 128.0
DEAP_N_CHANNELS: int = 32
DEAP_N_SUBJECTS: int = 32
DEAP_N_TRIALS: int = 40
DEAP_TRIAL_SAMPLES: int = 8064  # 63 sec at 128 Hz

# Target parameters (Muse 2 equivalent)
TARGET_FS: float = 256.0
TARGET_N_CHANNELS: int = 4
TARGET_WINDOW_SECONDS: float = 4.0
TARGET_WINDOW_SAMPLES: int = int(TARGET_WINDOW_SECONDS * TARGET_FS)  # 1024

# DEAP channel indices closest to Muse 2 positions
# AF3 (ch7) -> AF7, AF4 (ch12) -> AF8, Fp1 (ch0) -> TP9, Fp2 (ch3) -> TP10
DEAP_CHANNEL_MAP: Dict[str, int] = {
    "TP9": 0,   # Fp1 (closest to left temporal)
    "AF7": 7,   # AF3 (closest to left frontal)
    "AF8": 12,  # AF4 (closest to right frontal)
    "TP10": 3,  # Fp2 (closest to right temporal)
}
DEAP_CHANNEL_INDICES: List[int] = [
    DEAP_CHANNEL_MAP["TP9"],
    DEAP_CHANNEL_MAP["AF7"],
    DEAP_CHANNEL_MAP["AF8"],
    DEAP_CHANNEL_MAP["TP10"],
]
CHANNEL_NAMES: List[str] = ["TP9", "AF7", "AF8", "TP10"]

# Frequency bands for feature extraction
FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# DEAP valence labels: rating 1-9, split at 5.0
VALENCE_THRESHOLD: float = 5.0

# Emotion classes (3-class valence, matching CNN-KAN output)
EMOTION_CLASSES: List[str] = ["positive", "neutral", "negative"]


# ── Training config ──────────────────────────────────────────────────────────


def create_training_config(
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 100,
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
    cv_strategy: str = "5-fold",
    optimizer: str = "adam",
    weight_decay: float = 1e-4,
    scheduler: str = "cosine",
    label_smoothing: float = 0.1,
    n_classes: int = 3,
    window_seconds: float = TARGET_WINDOW_SECONDS,
    overlap: float = 0.5,
    bandpass_low: float = 1.0,
    bandpass_high: float = 45.0,
) -> Dict[str, Any]:
    """Create a complete training pipeline configuration.

    Args:
        learning_rate:           Initial learning rate.
        batch_size:              Training batch size.
        epochs:                  Maximum training epochs.
        validation_split:        Fraction for validation (within-subject).
        early_stopping_patience: Epochs without improvement before stopping.
        cv_strategy:             "5-fold", "loso" (leave-one-subject-out),
                                 or "within-subject".
        optimizer:               Optimizer name: "adam", "adamw", "sgd".
        weight_decay:            L2 regularization strength.
        scheduler:               LR scheduler: "cosine", "step", "plateau".
        label_smoothing:         Label smoothing epsilon for cross-entropy.
        n_classes:               Number of output classes (2 or 3).
        window_seconds:          Epoch window length in seconds.
        overlap:                 Window overlap fraction.
        bandpass_low:            High-pass cutoff frequency.
        bandpass_high:           Low-pass cutoff frequency.

    Returns:
        dict with full training pipeline specification.
    """
    window_samples = int(window_seconds * TARGET_FS)
    hop_samples = int(window_samples * (1.0 - overlap))

    # Estimate samples per subject
    usable_seconds_per_trial = DEAP_TRIAL_SAMPLES / DEAP_ORIGINAL_FS
    windows_per_trial = max(
        1, int((usable_seconds_per_trial - window_seconds) / (window_seconds * (1.0 - overlap))) + 1
    )
    windows_per_subject = windows_per_trial * DEAP_N_TRIALS
    total_windows = windows_per_subject * DEAP_N_SUBJECTS

    config = {
        "model": {
            "architecture": "CNN-KAN",
            "n_channels": TARGET_N_CHANNELS,
            "n_classes": n_classes,
            "emotion_classes": EMOTION_CLASSES[:n_classes],
            "input_shape": (TARGET_N_CHANNELS, window_samples),
        },
        "data": {
            "dataset": "DEAP",
            "n_subjects": DEAP_N_SUBJECTS,
            "n_trials_per_subject": DEAP_N_TRIALS,
            "original_fs": DEAP_ORIGINAL_FS,
            "target_fs": TARGET_FS,
            "channel_map": DEAP_CHANNEL_MAP,
            "channel_indices": DEAP_CHANNEL_INDICES,
            "channel_names": CHANNEL_NAMES,
            "bandpass": {"low": bandpass_low, "high": bandpass_high},
            "window_seconds": window_seconds,
            "window_samples": window_samples,
            "overlap": overlap,
            "hop_samples": hop_samples,
            "estimated_windows_per_trial": windows_per_trial,
            "estimated_windows_per_subject": windows_per_subject,
            "estimated_total_windows": total_windows,
            "valence_threshold": VALENCE_THRESHOLD,
        },
        "training": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": optimizer,
            "weight_decay": weight_decay,
            "scheduler": scheduler,
            "label_smoothing": label_smoothing,
            "early_stopping": {
                "enabled": True,
                "patience": early_stopping_patience,
                "monitor": "val_loss",
                "min_delta": 1e-4,
            },
        },
        "validation": {
            "strategy": cv_strategy,
            "validation_split": validation_split,
        },
        "features": {
            "type": "pseudo-rgb",
            "components": ["DE", "PSD", "EVI"],
            "frequency_bands": FREQUENCY_BANDS,
            "n_bands": len(FREQUENCY_BANDS),
        },
    }

    # Add CV-specific parameters
    if cv_strategy == "loso":
        config["validation"]["n_folds"] = DEAP_N_SUBJECTS
        config["validation"]["description"] = (
            f"Leave-one-subject-out: {DEAP_N_SUBJECTS} folds, "
            "each fold holds out 1 subject for testing."
        )
    elif cv_strategy == "5-fold":
        config["validation"]["n_folds"] = 5
        config["validation"]["description"] = (
            "5-fold stratified cross-validation across all subjects."
        )
    elif cv_strategy == "within-subject":
        config["validation"]["n_folds"] = DEAP_N_SUBJECTS
        config["validation"]["description"] = (
            "Within-subject train/test split per subject."
        )

    return config


# ── Data preprocessing ───────────────────────────────────────────────────────


def preprocess_deap_epoch(
    epoch: np.ndarray,
    original_fs: float = DEAP_ORIGINAL_FS,
    target_fs: float = TARGET_FS,
    bandpass_low: float = 1.0,
    bandpass_high: float = 45.0,
) -> np.ndarray:
    """Preprocess a single DEAP epoch for CNN-KAN training.

    Steps:
        1. Extract 4 channels closest to Muse 2 positions.
        2. Resample from 128 Hz to 256 Hz.
        3. Bandpass filter 1-45 Hz.

    Args:
        epoch:        (32, n_samples) raw DEAP EEG epoch.
        original_fs:  Original sampling rate (128 Hz for DEAP).
        target_fs:    Target sampling rate (256 Hz for Muse 2 compatibility).
        bandpass_low:  High-pass cutoff in Hz.
        bandpass_high: Low-pass cutoff in Hz.

    Returns:
        preprocessed: (4, n_resampled_samples) preprocessed EEG.
    """
    epoch = np.asarray(epoch, dtype=np.float64)

    if epoch.ndim == 1:
        raise ValueError("Epoch must be 2-D (n_channels, n_samples)")

    n_ch, n_samples = epoch.shape

    # Step 1: Extract 4 channels
    if n_ch >= max(DEAP_CHANNEL_INDICES) + 1:
        selected = epoch[DEAP_CHANNEL_INDICES, :]
    elif n_ch == TARGET_N_CHANNELS:
        selected = epoch  # Already 4 channels
    else:
        # Pad or truncate to 4 channels
        if n_ch < TARGET_N_CHANNELS:
            pad = np.zeros(
                (TARGET_N_CHANNELS - n_ch, n_samples), dtype=np.float64
            )
            selected = np.concatenate([epoch, pad], axis=0)
        else:
            selected = epoch[:TARGET_N_CHANNELS, :]

    # Step 2: Resample to target_fs
    if abs(original_fs - target_fs) > 0.1:
        resample_ratio = target_fs / original_fs
        new_n_samples = int(n_samples * resample_ratio)
        resampled = np.zeros(
            (TARGET_N_CHANNELS, new_n_samples), dtype=np.float64
        )
        for ch_idx in range(TARGET_N_CHANNELS):
            resampled[ch_idx] = _resample_signal(
                selected[ch_idx], new_n_samples
            )
        selected = resampled

    # Step 3: Bandpass filter
    filtered = _bandpass_filter(selected, target_fs, bandpass_low, bandpass_high)

    return filtered


# ── Cross-validation setup ───────────────────────────────────────────────────


def setup_cross_validation(
    n_subjects: int = DEAP_N_SUBJECTS,
    strategy: str = "5-fold",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Generate cross-validation fold assignments.

    Args:
        n_subjects: Number of subjects in the dataset.
        strategy:   "5-fold", "loso", or "within-subject".
        random_seed: Random seed for reproducibility.

    Returns:
        dict with:
            folds:    List of dicts, each with "train" and "test" subject indices.
            strategy: Strategy name.
            n_folds:  Number of folds.
    """
    rng = np.random.RandomState(random_seed)
    subjects = list(range(n_subjects))

    if strategy == "loso":
        # Leave-one-subject-out
        folds = []
        for held_out in subjects:
            train = [s for s in subjects if s != held_out]
            folds.append({
                "fold": held_out,
                "train_subjects": train,
                "test_subjects": [held_out],
                "n_train_subjects": len(train),
                "n_test_subjects": 1,
            })

    elif strategy == "5-fold":
        n_folds = 5
        shuffled = list(subjects)
        rng.shuffle(shuffled)
        fold_size = n_subjects // n_folds
        remainder = n_subjects % n_folds

        folds = []
        start = 0
        for fold_idx in range(n_folds):
            # Distribute remainder subjects across first folds
            end = start + fold_size + (1 if fold_idx < remainder else 0)
            test = shuffled[start:end]
            train = [s for s in shuffled if s not in test]
            folds.append({
                "fold": fold_idx,
                "train_subjects": train,
                "test_subjects": test,
                "n_train_subjects": len(train),
                "n_test_subjects": len(test),
            })
            start = end

    elif strategy == "within-subject":
        # Each subject gets its own train/test split
        folds = []
        for subj in subjects:
            folds.append({
                "fold": subj,
                "train_subjects": [subj],
                "test_subjects": [subj],
                "n_train_subjects": 1,
                "n_test_subjects": 1,
                "note": "Within-subject: train/test split within this subject's data",
            })

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use '5-fold', 'loso', or 'within-subject'.")

    return {
        "strategy": strategy,
        "n_folds": len(folds),
        "n_subjects": n_subjects,
        "random_seed": random_seed,
        "folds": folds,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_training_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 3,
) -> Dict[str, Any]:
    """Compute training metrics: accuracy, F1, confusion matrix.

    Args:
        y_true: (n_samples,) ground truth labels (0-indexed).
        y_pred: (n_samples,) predicted labels (0-indexed).
        n_classes: Number of classes.

    Returns:
        dict with accuracy, per-class F1, macro F1, and confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_per_class": {cls: 0.0 for cls in EMOTION_CLASSES[:n_classes]},
            "confusion_matrix": np.zeros((n_classes, n_classes), dtype=int).tolist(),
            "n_samples": 0,
        }

    # Accuracy
    accuracy = float(np.mean(y_true == y_pred))

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    # Per-class precision, recall, F1
    f1_per_class = {}
    for cls_idx in range(n_classes):
        tp = cm[cls_idx, cls_idx]
        fp = int(np.sum(cm[:, cls_idx]) - tp)
        fn = int(np.sum(cm[cls_idx, :]) - tp)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            2.0 * precision * recall / max(precision + recall, 1e-12)
        )

        cls_name = EMOTION_CLASSES[cls_idx] if cls_idx < len(EMOTION_CLASSES) else f"class_{cls_idx}"
        f1_per_class[cls_name] = round(float(f1), 4)

    f1_macro = float(np.mean(list(f1_per_class.values())))

    return {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
        "n_correct": int(np.sum(y_true == y_pred)),
        "class_distribution": {
            EMOTION_CLASSES[i] if i < len(EMOTION_CLASSES) else f"class_{i}": int(np.sum(y_true == i))
            for i in range(n_classes)
        },
    }


def training_report_to_dict(
    config: Dict[str, Any],
    fold_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a full training report from config and per-fold metrics.

    Args:
        config:       Training config from create_training_config().
        fold_metrics: List of per-fold metric dicts from compute_training_metrics().

    Returns:
        dict with overall summary, per-fold breakdown, and config reference.
    """
    if not fold_metrics:
        return {
            "status": "no_results",
            "config": config,
            "summary": {"n_folds": 0},
        }

    accuracies = [m["accuracy"] for m in fold_metrics]
    f1_macros = [m["f1_macro"] for m in fold_metrics]

    summary = {
        "n_folds": len(fold_metrics),
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "std_accuracy": round(float(np.std(accuracies)), 4),
        "min_accuracy": round(float(np.min(accuracies)), 4),
        "max_accuracy": round(float(np.max(accuracies)), 4),
        "mean_f1_macro": round(float(np.mean(f1_macros)), 4),
        "std_f1_macro": round(float(np.std(f1_macros)), 4),
        "total_samples": sum(m.get("n_samples", 0) for m in fold_metrics),
        "total_correct": sum(m.get("n_correct", 0) for m in fold_metrics),
    }

    # Per-class average F1 across folds
    class_names = list(fold_metrics[0].get("f1_per_class", {}).keys())
    per_class_avg = {}
    for cls_name in class_names:
        cls_f1s = [
            m["f1_per_class"].get(cls_name, 0.0) for m in fold_metrics
        ]
        per_class_avg[cls_name] = {
            "mean_f1": round(float(np.mean(cls_f1s)), 4),
            "std_f1": round(float(np.std(cls_f1s)), 4),
        }

    return {
        "status": "complete",
        "summary": summary,
        "per_class": per_class_avg,
        "per_fold": [
            {
                "fold": i,
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "n_samples": m.get("n_samples", 0),
            }
            for i, m in enumerate(fold_metrics)
        ],
        "config": {
            "model": config.get("model", {}),
            "training": config.get("training", {}),
            "validation": config.get("validation", {}),
        },
    }


# ── Private helpers ──────────────────────────────────────────────────────────


def _resample_signal(
    signal: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """Resample a 1-D signal to target_length using linear interpolation.

    For production: use scipy.signal.resample for proper anti-aliasing.
    This is a lightweight fallback that avoids scipy dependency.
    """
    n = len(signal)
    if n == target_length:
        return signal.copy()

    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, signal)


def _bandpass_filter(
    signals: np.ndarray,
    fs: float,
    low: float,
    high: float,
) -> np.ndarray:
    """Apply bandpass filter to multichannel signals.

    Uses scipy Butterworth if available, otherwise returns signals unfiltered
    (with a log warning).
    """
    try:
        from scipy.signal import butter, filtfilt

        order = 4
        nyq = fs / 2.0
        low_norm = max(low / nyq, 0.001)
        high_norm = min(high / nyq, 0.999)
        b, a = butter(order, [low_norm, high_norm], btype="band")

        filtered = np.zeros_like(signals)
        for ch_idx in range(signals.shape[0]):
            # Ensure signal is long enough for filtfilt
            if len(signals[ch_idx]) > 3 * max(len(b), len(a)):
                filtered[ch_idx] = filtfilt(b, a, signals[ch_idx])
            else:
                filtered[ch_idx] = signals[ch_idx]
        return filtered

    except ImportError:
        log.warning(
            "scipy not available for bandpass filter; returning unfiltered signal"
        )
        return signals.copy()
