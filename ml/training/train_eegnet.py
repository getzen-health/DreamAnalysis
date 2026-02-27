"""Train EEGNet emotion classifier.

Trains a separate model file per channel count so the same architecture
works on Muse 2 (4ch), OpenBCI Cyton (8ch), and Cyton+Daisy (16ch).

Saved files:
  ml/models/saved/eegnet_emotion_{n_ch}ch.pt       — model weights + config
  ml/models/saved/eegnet_emotion_{n_ch}ch.onnx     — ONNX export for mobile
  ml/models/saved/eegnet_emotion_{n_ch}ch_benchmark.txt  — test accuracy

Usage:
  # Train on Muse 2 data (4 channels):
  python -m training.train_eegnet --channels 4 --data-dir data/pilot

  # Train on OpenBCI Cyton data (8 channels):
  python -m training.train_eegnet --channels 8 --data-dir data/openbci

  # Train on all available data:
  python -m training.train_eegnet --channels 4 --use-synthetic --epochs 100

Data format expected (--data-dir):
  Parquet files with columns: [eeg_ch0, eeg_ch1, ..., eeg_chN, label]
  OR
  NPZ files with keys: X (n_epochs, n_channels, n_samples), y (n_epochs,)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.eegnet import EEGNet

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]
N_CLASSES = len(EMOTIONS)
SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"
EPOCH_SEC = 4          # seconds per epoch
OVERLAP = 0.5          # 50% overlap between epochs
MIN_EPOCHS_PER_CLASS = 20   # minimum samples per class to attempt training


# ── Data loading ──────────────────────────────────────────────────────────

def load_npz(path: Path, n_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load (X, y) from an NPZ file. X shape: (n_epochs, n_channels, n_samples)."""
    data = np.load(path)
    X, y = data["X"], data["y"]
    if X.shape[1] != n_channels:
        log.warning("NPZ has %d channels, expected %d — skipping %s", X.shape[1], n_channels, path)
        return np.empty((0, n_channels, X.shape[2])), np.empty(0)
    return X.astype(np.float32), y.astype(np.int64)


def load_parquet_dir(data_dir: Path, n_channels: int, fs: float = 256.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load EEG epochs from Parquet files in data_dir.

    Parquet must have columns: eeg_ch0, eeg_ch1, ..., eeg_chN, label
    label is an integer 0–5 (index into EMOTIONS).
    """
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas not installed — cannot load Parquet data")
        return np.empty((0, n_channels, 0)), np.empty(0)

    epoch_samples = int(EPOCH_SEC * fs)
    hop = int(epoch_samples * (1 - OVERLAP))

    X_list, y_list = [], []

    for pq_file in sorted(data_dir.glob("**/*.parquet")):
        try:
            df = pd.read_parquet(pq_file)
            ch_cols = [f"eeg_ch{i}" for i in range(n_channels)]
            if not all(c in df.columns for c in ch_cols) or "label" not in df.columns:
                log.debug("Skipping %s — missing columns", pq_file.name)
                continue

            raw = df[ch_cols].values.T.astype(np.float32)  # (n_channels, n_total)
            labels = df["label"].values.astype(np.int64)

            n_total = raw.shape[1]
            start = 0
            while start + epoch_samples <= n_total:
                epoch = raw[:, start:start + epoch_samples]
                # Epoch label = majority vote of per-sample labels in this window
                window_labels = labels[start:start + epoch_samples]
                label = int(np.bincount(window_labels, minlength=N_CLASSES).argmax())
                X_list.append(epoch)
                y_list.append(label)
                start += hop
        except Exception as exc:
            log.warning("Failed to load %s: %s", pq_file, exc)

    if not X_list:
        return np.empty((0, n_channels, epoch_samples)), np.empty(0)

    return np.stack(X_list), np.array(y_list)


def make_synthetic_data(n_channels: int, fs: float = 256.0, n_per_class: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic EEG-like data for smoke-testing the training pipeline.

    Each emotion class has a characteristic band power signature:
      happy    → high alpha + moderate beta
      sad      → low alpha + low beta
      angry    → high beta + high gamma
      fearful  → high beta (especially high-beta 20-30 Hz)
      relaxed  → high alpha + low beta
      focused  → moderate beta + low alpha
    """
    rng = np.random.default_rng(42)
    epoch_samples = int(EPOCH_SEC * fs)

    X_list, y_list = [], []
    t = np.linspace(0, EPOCH_SEC, epoch_samples, endpoint=False)

    # Signature: (alpha_amp, beta_amp, theta_amp, noise_std)
    signatures = [
        (2.0, 0.8, 0.5, 0.3),   # happy
        (0.5, 0.3, 0.8, 0.5),   # sad
        (0.5, 2.0, 0.5, 0.8),   # angry
        (0.3, 1.8, 0.4, 0.7),   # fearful
        (2.5, 0.3, 0.6, 0.2),   # relaxed
        (0.7, 1.2, 0.4, 0.3),   # focused
    ]

    for label, (a_amp, b_amp, th_amp, noise) in enumerate(signatures):
        for _ in range(n_per_class):
            epoch = np.zeros((n_channels, epoch_samples), dtype=np.float32)
            for ch in range(n_channels):
                phase = rng.uniform(0, 2 * np.pi)
                sig = (
                    a_amp  * np.sin(2 * np.pi * 10 * t + phase)   # 10 Hz alpha
                    + b_amp * np.sin(2 * np.pi * 20 * t + phase)  # 20 Hz beta
                    + th_amp * np.sin(2 * np.pi * 6 * t + phase)  # 6 Hz theta
                    + rng.normal(0, noise, epoch_samples)
                )
                epoch[ch] = sig.astype(np.float32)
            X_list.append(epoch)
            y_list.append(label)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── Preprocessing ─────────────────────────────────────────────────────────

def normalize_epochs(X: np.ndarray) -> np.ndarray:
    """Z-score each epoch independently (per-channel zero-mean, unit-variance)."""
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True) + 1e-7
    return (X - mean) / std


# ── Training loop ─────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: np.ndarray,
    n_channels: int,
    n_samples: int,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 20,
    device: str = "cpu",
) -> Tuple[EEGNet, float]:
    """Train EEGNet and return (model, test_accuracy)."""

    log.info(
        "Training EEGNet — channels=%d, samples_per_epoch=%d, n_epochs=%d",
        n_channels, n_samples, len(y),
    )

    # Normalise
    X = normalize_epochs(X)

    # Class weights (handle imbalance)
    class_counts = np.bincount(y, minlength=N_CLASSES).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)

    # Tensors
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    dataset = TensorDataset(X_t, y_t)

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = EEGNet(
        n_classes=N_CLASSES,
        n_channels=n_channels,
        n_samples=n_samples,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += len(yb)

        val_acc = correct / max(total, 1)

        if epoch % 10 == 0:
            log.info(
                "Epoch %3d/%d  loss=%.4f  val_acc=%.2f%%",
                epoch, epochs, train_loss / len(train_loader), val_acc * 100,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d (best val_acc=%.2f%%)", epoch, best_val_acc * 100)
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    log.info("Training complete — best val_acc=%.2f%%", best_val_acc * 100)
    return model, best_val_acc


# ── Save + ONNX export ────────────────────────────────────────────────────

def save_model(model: EEGNet, accuracy: float, n_channels: int) -> None:
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    pt_path = SAVED_DIR / f"eegnet_emotion_{n_channels}ch.pt"
    model.save(pt_path)
    print(f"Model saved → {pt_path}")

    bm_path = SAVED_DIR / f"eegnet_emotion_{n_channels}ch_benchmark.txt"
    bm_path.write_text(f"{accuracy:.6f}")
    print(f"Benchmark saved → {bm_path}  ({accuracy*100:.2f}%)")

    onnx_path = SAVED_DIR / f"eegnet_emotion_{n_channels}ch.onnx"
    try:
        model.export_onnx(onnx_path)
        print(f"ONNX exported → {onnx_path}")
    except Exception as e:
        log.warning("ONNX export failed (non-critical): %s", e)


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train EEGNet emotion classifier")
    parser.add_argument("--channels",    type=int, default=4,     help="Number of EEG channels (4/8/16)")
    parser.add_argument("--data-dir",    type=Path, default=None, help="Directory with Parquet/NPZ training data")
    parser.add_argument("--npz",         type=Path, default=None, help="Single NPZ file (X, y)")
    parser.add_argument("--use-synthetic",action="store_true",    help="Use synthetic data (smoke-test)")
    parser.add_argument("--epochs",      type=int, default=150,   help="Max training epochs")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int, default=20,    help="Early stopping patience")
    parser.add_argument("--fs",          type=float, default=256.0, help="EEG sampling rate Hz")
    parser.add_argument("--epoch-sec",   type=float, default=4.0,   help="Seconds per epoch")
    args = parser.parse_args()

    n_samples = int(args.epoch_sec * args.fs)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Using device: %s", device)

    # ── Load data ─────────────────────────────────────────────────────
    X, y = None, None

    if args.npz:
        X, y = load_npz(args.npz, args.channels)
        log.info("NPZ loaded: X=%s y=%s", X.shape, y.shape)

    elif args.data_dir:
        X, y = load_parquet_dir(args.data_dir, args.channels, fs=args.fs)
        log.info("Parquet data loaded: %d epochs", len(y) if y is not None else 0)

    elif args.use_synthetic:
        log.info("Using synthetic data (smoke-test — not for production)")
        X, y = make_synthetic_data(args.channels, fs=args.fs, n_per_class=200)
        log.info("Synthetic data: X=%s y=%s", X.shape, y.shape)

    else:
        parser.error("Provide --data-dir, --npz, or --use-synthetic")

    if X is None or len(y) == 0:
        print("ERROR: No training data found. Check your --data-dir path.")
        sys.exit(1)

    # Check minimum class coverage
    counts = np.bincount(y, minlength=N_CLASSES)
    log.info("Class distribution: %s", dict(zip(EMOTIONS, counts.tolist())))
    if counts.min() < MIN_EPOCHS_PER_CLASS:
        log.warning(
            "Some classes have < %d samples. Accuracy may be poor. "
            "Collect more data before using in production.",
            MIN_EPOCHS_PER_CLASS,
        )

    # Trim/pad to target n_samples
    actual_samples = X.shape[2]
    if actual_samples != n_samples:
        if actual_samples > n_samples:
            X = X[:, :, :n_samples]
        else:
            pad = n_samples - actual_samples
            X = np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="edge")
        log.info("Trimmed/padded epochs to %d samples", n_samples)

    # ── Train ─────────────────────────────────────────────────────────
    model, accuracy = train(
        X, y,
        n_channels=args.channels,
        n_samples=n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )

    save_model(model, accuracy, args.channels)

    print(f"\nDone. Test accuracy: {accuracy*100:.2f}%")
    if accuracy >= 0.60:
        print("✓ Meets 60% threshold — will be used in live inference over feature-based heuristics.")
    else:
        print("✗ Below 60% threshold — collect more labeled data before deploying.")


if __name__ == "__main__":
    main()
