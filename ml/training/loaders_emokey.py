"""EmoKey Moments EEG Dataset (EKM-ED) loader for MLX training pipeline.

Dataset: 45 subjects, real Muse S 4-channel EEG (TP9, AF7, AF8, TP10)
Sampling rate: 128 Hz (0.0078125s interval)
Labels: ANGER, FEAR, HAPPINESS, SADNESS + NEUTRAL variants

Returns (X, y) where:
  X: (n_epochs, 4, 512) float32  — 4 channels × 4-sec epochs at 128 Hz
  y: (n_epochs,) int64            — 0=happy 1=sad 2=angry 3=fear 4=neutral

Usage:
    from training.loaders_emokey import load_emokey
    X, y = load_emokey()  # loads all 45 subjects
    X, y = load_emokey(max_subjects=10)  # subset for quick tests
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

FS = 128.0          # sampling rate (Hz)
EPOCH_SEC = 4.0     # epoch length
OVERLAP = 0.5       # 50% overlap
EPOCH_SAMPLES = int(FS * EPOCH_SEC)   # 512 samples per epoch
HOP = int(EPOCH_SAMPLES * (1 - OVERLAP))   # 256 samples hop

RAW_COLS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
HSI_COLS = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]

LABEL_MAP = {
    "HAPPINESS":        0,  # happy
    "SADNESS":          1,  # sad
    "ANGER":            2,  # angry
    "FEAR":             3,  # fear
    "NEUTRAL_HAPPINESS": 4, # neutral
    "NEUTRAL_SADNESS":   4,
    "NEUTRAL_ANGER":     4,
    "NEUTRAL_FEAR":      4,
}

LABEL_NAMES = ["happy", "sad", "angry", "fear", "neutral"]
N_CLASSES = len(LABEL_NAMES)

EMOKEY_ROOT = (
    pathlib.Path(__file__).parent.parent
    / "data" / "emokey"
    / "EmoKey Moments EEG Dataset (EKM-ED)"
    / "muse_wearable_data"
    / "preprocessed"
    / "clean-signals"
    / "0.0078125S"
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _zscore_epoch(epoch: np.ndarray) -> np.ndarray:
    """Z-score each channel independently. epoch shape: (4, n_samples)."""
    mean = epoch.mean(axis=1, keepdims=True)
    std  = epoch.std(axis=1, keepdims=True) + 1e-7
    return (epoch - mean) / std


def _load_csv(csv_path: pathlib.Path, label: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load one EmoKey CSV → (X_epochs, y_epochs).

    Applies:
      1. HSI quality filter — drops samples where any channel HSI > 2
      2. Amplitude filter   — drops epochs where any channel |z| > 5 (artifact)
      3. 4-second windowing with 50% overlap
      4. Per-epoch z-score normalisation
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    df = pd.read_csv(csv_path)

    # Drop rows with bad electrode contact (HSI > 2)
    hsi_mask = (df[HSI_COLS] <= 2).all(axis=1)
    df = df[hsi_mask].reset_index(drop=True)

    if len(df) < EPOCH_SAMPLES:
        log.debug("Skipping %s — too few good samples (%d)", csv_path.name, len(df))
        return np.empty((0, 4, EPOCH_SAMPLES), dtype=np.float32), np.empty(0, dtype=np.int64)

    raw = df[RAW_COLS].values.T.astype(np.float32)   # (4, n_samples)
    n_samples = raw.shape[1]

    X_list = []
    start = 0
    while start + EPOCH_SAMPLES <= n_samples:
        epoch = raw[:, start:start + EPOCH_SAMPLES]
        epoch = _zscore_epoch(epoch)

        # Artifact rejection: drop epoch if any value > 5σ (already z-scored)
        if np.abs(epoch).max() < 5.0:
            X_list.append(epoch)

        start += HOP

    if not X_list:
        return np.empty((0, 4, EPOCH_SAMPLES), dtype=np.float32), np.empty(0, dtype=np.int64)

    X = np.stack(X_list)                                  # (n_epochs, 4, 512)
    y = np.full(len(X_list), label, dtype=np.int64)
    return X, y


# ── Public API ─────────────────────────────────────────────────────────────

def load_emokey(
    data_dir: Optional[pathlib.Path] = None,
    max_subjects: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load EmoKey dataset → (X, y).

    Args:
        data_dir:     Override default EMOKEY_ROOT path.
        max_subjects: Cap number of subjects (for quick dev runs).
        seed:         RNG seed for shuffling.

    Returns:
        X: (n_epochs, 4, 512) float32
        y: (n_epochs,)        int64   — indices into LABEL_NAMES
    """
    root = pathlib.Path(data_dir) if data_dir else EMOKEY_ROOT

    if not root.exists():
        raise FileNotFoundError(f"EmoKey not found at {root}")

    subject_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]

    log.info("Loading EmoKey — %d subjects from %s", len(subject_dirs), root)

    X_all, y_all = [], []
    skipped_files = 0

    for subj_dir in subject_dirs:
        for csv_path in sorted(subj_dir.glob("*.csv")):
            stem = csv_path.stem  # e.g. "ANGER", "NEUTRAL_FEAR"
            label = LABEL_MAP.get(stem)
            if label is None:
                log.debug("Unknown label file: %s — skipping", csv_path.name)
                skipped_files += 1
                continue

            X, y = _load_csv(csv_path, label)
            if len(X):
                X_all.append(X)
                y_all.append(y)

    if not X_all:
        raise ValueError("No usable epochs found in EmoKey dataset.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    # Shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    # Report
    counts = {LABEL_NAMES[i]: int((y == i).sum()) for i in range(N_CLASSES)}
    log.info(
        "EmoKey loaded: %d epochs, 4 ch, %d samples/epoch (%.1f sec) | %s",
        len(y), EPOCH_SAMPLES, EPOCH_SEC, counts,
    )

    return X, y


def load_emokey_splits(
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    **kwargs,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Convenience wrapper: returns (train, val, test) tuples."""
    X, y = load_emokey(**kwargs)
    n = len(y)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)

    X_test,  y_test  = X[:n_test],       y[:n_test]
    X_val,   y_val   = X[n_test:n_test+n_val], y[n_test:n_test+n_val]
    X_train, y_train = X[n_test+n_val:], y[n_test+n_val:]

    log.info("Train: %d  Val: %d  Test: %d", len(y_train), len(y_val), len(y_test))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    X, y = load_emokey()
    print(f"X shape: {X.shape}  y shape: {y.shape}")
    print(f"Label distribution: { {LABEL_NAMES[i]: int((y==i).sum()) for i in range(N_CLASSES)} }")
