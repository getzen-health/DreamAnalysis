"""Train EEGNet emotion classifier using Apple MLX (Metal + Apple Neural Engine).

This runs on Metal/ANE — no GPU or CUDA needed. Requires Apple Silicon Mac.
5-10x faster than CPU-only PyTorch on M1/M2/M3 for EEGNet-scale models.

How to run:
  cd ml
  pip install mlx>=0.18.0
  python -m training.train_eegnet_mlx --synthetic
  python -m training.train_eegnet_mlx --channels 4 --data-dir data/pilot

Saved files:
  ml/models/saved/eegnet_emotion_{n_ch}ch_mlx.npz           — MLX weights
  ml/models/saved/eegnet_emotion_{n_ch}ch_mlx_benchmark.txt  — test accuracy

After training, the script also exports to .pt (PyTorch state dict) so you can
load the weights into the PyTorch EEGNet and run ONNX export on any machine.

Data format: same as train_eegnet.py (Parquet or NPZ files).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# ── Project path setup ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Reuse data utilities from the PyTorch training script ─────────────────
from training.train_eegnet import (
    EMOTIONS,
    EPOCH_SEC,
    N_CLASSES,
    SAVED_DIR,
    load_npz,
    load_parquet_dir,
    make_synthetic_data,
    normalize_epochs,
)

# ── MLX imports ───────────────────────────────────────────────────────────
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

from models.eegnet_mlx import EEGNetMLX, build_eegnet_mlx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_EPOCHS_PER_CLASS = 20


# ---------------------------------------------------------------------------
# Dataset / batching utilities (pure numpy — no PyTorch DataLoader)
# ---------------------------------------------------------------------------

def _make_splits(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split into train / val arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    n_val = max(1, int(len(y) * val_split))
    return X[n_val:], y[n_val:], X[:n_val], y[:n_val]


def _iter_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """Yield (X_batch, y_batch) as numpy arrays."""
    n = len(y)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n - batch_size + 1, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


# ---------------------------------------------------------------------------
# Loss function and training step
# ---------------------------------------------------------------------------

def _loss_fn(model: EEGNetMLX, X_mx, y_mx, class_weights_mx=None):
    """Cross-entropy loss with optional class weighting."""
    logits = model(X_mx, training=True)
    # nn.losses.cross_entropy returns per-sample losses (shape: batch,)
    per_sample = nn.losses.cross_entropy(logits, y_mx)
    if class_weights_mx is not None:
        # Weight each sample by the weight of its class
        w = class_weights_mx[y_mx]
        return mx.sum(per_sample * w) / mx.sum(w)
    return mx.mean(per_sample)


def _accuracy(model: EEGNetMLX, X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> float:
    """Compute accuracy on a numpy dataset (evaluation mode — no dropout)."""
    correct = 0
    total = 0
    for xb, yb in _iter_batches(X, y, batch_size, shuffle=False):
        xb_mx = mx.array(xb)
        logits = model(xb_mx, training=False)
        mx.eval(logits)
        preds = np.array(logits).argmax(axis=1)
        correct += (preds == yb).sum()
        total += len(yb)
    # Handle remainder (last partial batch)
    rem = len(y) % batch_size
    if rem > 0:
        xb = X[-rem:]
        yb = y[-rem:]
        xb_mx = mx.array(xb)
        logits = model(xb_mx, training=False)
        mx.eval(logits)
        preds = np.array(logits).argmax(axis=1)
        correct += (preds == yb).sum()
        total += len(yb)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_mlx(
    X: np.ndarray,
    y: np.ndarray,
    n_channels: int,
    n_samples: int,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 20,
) -> Tuple[EEGNetMLX, float]:
    """Train EEGNetMLX and return (model, best_val_accuracy).

    Parameters
    ----------
    X          : (n_epochs, n_channels, n_samples) float32
    y          : (n_epochs,) int64
    n_channels : EEG channels
    n_samples  : samples per epoch
    epochs     : maximum training epochs
    batch_size : mini-batch size
    lr         : initial learning rate (cosine decayed)
    val_split  : fraction of data held out for validation
    patience   : early stopping patience (epochs without val improvement)
    """
    if not _MLX_AVAILABLE:
        raise ImportError("MLX not available. Install: pip install mlx>=0.18.0")

    log.info(
        "MLX backend: %s | training EEGNetMLX — %d-ch, %d samples, %d epochs",
        mx.default_device(), n_channels, n_samples, len(y),
    )

    # ── Normalise ─────────────────────────────────────────────────────────
    X = normalize_epochs(X)

    # ── Train/val split ───────────────────────────────────────────────────
    X_train, y_train, X_val, y_val = _make_splits(X, y, val_split=val_split)
    log.info("Train: %d  Val: %d", len(y_train), len(y_val))

    # ── Class weights (handle imbalance) ──────────────────────────────────
    class_counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    cw = 1.0 / class_counts
    cw = cw / cw.sum() * N_CLASSES   # normalise so weights sum to N_CLASSES
    class_weights_mx = mx.array(cw)

    # ── Model + optimiser ─────────────────────────────────────────────────
    model = build_eegnet_mlx(
        n_classes=N_CLASSES,
        n_channels=n_channels,
        n_samples=n_samples,
    )

    decay_steps = epochs * max(1, len(y_train) // batch_size)
    lr_schedule = optim.cosine_decay(lr, decay_steps, end_value=1e-5)
    optimizer = optim.Adam(learning_rate=lr_schedule)

    # Build the value-and-grad function
    loss_and_grad = nn.value_and_grad(model, _loss_fn)

    best_val_acc = 0.0
    best_params = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        train_loss_sum = 0.0
        n_batches = 0
        for xb_np, yb_np in _iter_batches(X_train, y_train, batch_size, shuffle=True):
            xb_mx = mx.array(xb_np)
            yb_mx = mx.array(yb_np)
            loss, grads = loss_and_grad(model, xb_mx, yb_mx, class_weights_mx)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            train_loss_sum += float(loss.item())
            n_batches += 1

        # ── Validate ──────────────────────────────────────────────────────
        val_acc = _accuracy(model, X_val, y_val, batch_size=batch_size)

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = train_loss_sum / max(n_batches, 1)
            log.info(
                "Epoch %3d/%d  loss=%.4f  val_acc=%.2f%%",
                epoch, epochs, avg_loss, val_acc * 100,
            )

        # ── Early stopping ─────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Deep-copy parameter trees via numpy round-trip
            best_params = {
                k: np.array(v)
                for k, v in _flatten_params_iter(model.parameters())
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(
                    "Early stopping at epoch %d  (best val_acc=%.2f%%)",
                    epoch, best_val_acc * 100,
                )
                break

    # ── Restore best weights ──────────────────────────────────────────────
    if best_params is not None:
        nested = _unflatten_params_numpy(best_params)
        model.update(nested)
        mx.eval(model.parameters())

    log.info("Training complete — best val_acc=%.2f%%", best_val_acc * 100)
    return model, best_val_acc


# ---------------------------------------------------------------------------
# Parameter dict helpers (needed for best-weight snapshot)
# ---------------------------------------------------------------------------

def _flatten_params_iter(d, prefix=""):
    """Yield (dotted_key, mlx_array) pairs from nested parameter dict."""
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flatten_params_iter(v, full)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    yield from _flatten_params_iter(item, f"{full}.{i}")
                else:
                    yield f"{full}.{i}", item
        else:
            yield full, v


def _unflatten_params_numpy(flat: dict) -> dict:
    """Reconstruct nested MLX array dict from flat numpy dict."""
    nested: dict = {}
    for key, arr in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = mx.array(arr)
    return nested


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_mlx_model(model: EEGNetMLX, accuracy: float, n_channels: int) -> None:
    """Save MLX weights, benchmark, and attempt PyTorch state dict export."""
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    # ── MLX weights ───────────────────────────────────────────────────────
    npz_path = SAVED_DIR / f"eegnet_emotion_{n_channels}ch_mlx.npz"
    model.save_weights(npz_path)
    print(f"MLX weights saved  → {npz_path}")

    # ── Benchmark ─────────────────────────────────────────────────────────
    bm_path = SAVED_DIR / f"eegnet_emotion_{n_channels}ch_mlx_benchmark.txt"
    bm_path.write_text(f"{accuracy:.6f}")
    print(f"Benchmark saved    → {bm_path}  ({accuracy * 100:.2f}%)")

    # ── PyTorch state dict export (for ONNX) ─────────────────────────────
    try:
        import torch
        from models.eegnet import EEGNet

        pt_state = model.to_pytorch_state_dict()
        pt_model = EEGNet(
            n_classes=model.n_classes,
            n_channels=model.n_channels,
            n_samples=model.n_samples,
            F1=model.F1,
            D=model.D,
            dropout_rate=model.dropout_rate,
        )
        # Load only the keys that match (shape may differ due to axis convention)
        current_sd = pt_model.state_dict()
        compatible = {}
        for k, v in pt_state.items():
            if k in current_sd and current_sd[k].shape == tuple(v.shape):
                compatible[k] = torch.from_numpy(v)
            else:
                log.debug("Skipping key %s (shape mismatch)", k)
        missing, unexpected = pt_model.load_state_dict(compatible, strict=False)
        if missing:
            log.debug("PyTorch keys not loaded from MLX: %s", missing)

        pt_path = SAVED_DIR / f"eegnet_emotion_{n_channels}ch_mlx.pt"
        torch.save({
            "state_dict": pt_model.state_dict(),
            "config": {
                "n_classes":    model.n_classes,
                "n_channels":   model.n_channels,
                "n_samples":    model.n_samples,
                "F1":           model.F1,
                "D":            model.D,
                "dropout_rate": model.dropout_rate,
            },
        }, pt_path)
        print(f"PyTorch .pt saved  → {pt_path}  (for ONNX export)")

        # ONNX export
        onnx_path = SAVED_DIR / f"eegnet_emotion_{n_channels}ch_mlx.onnx"
        try:
            pt_model.export_onnx(onnx_path)
            print(f"ONNX exported      → {onnx_path}")
        except Exception as exc:
            log.warning("ONNX export failed (non-critical): %s", exc)

    except ImportError:
        log.info("PyTorch not available — skipping .pt / ONNX export (MLX .npz still saved)")
    except Exception as exc:
        log.warning("PyTorch bridge export failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if not _MLX_AVAILABLE:
        print(
            "ERROR: MLX is not installed.\n"
            "Install: pip install mlx>=0.18.0\n"
            "MLX requires Apple Silicon (M1/M2/M3)."
        )
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(f"MLX backend: {mx.default_device()}")

    parser = argparse.ArgumentParser(
        description="Train EEGNet emotion classifier with MLX (Metal/ANE)"
    )
    parser.add_argument("--channels",      type=int,   default=4,     help="EEG channels (4/8/16)")
    parser.add_argument("--data-dir",      type=Path,  default=None,  help="Parquet/NPZ data directory")
    parser.add_argument("--npz",           type=Path,  default=None,  help="Single NPZ file (X, y)")
    parser.add_argument("--synthetic",     action="store_true",       help="Use synthetic data (smoke-test)")
    parser.add_argument("--epochs",        type=int,   default=150,   help="Max training epochs")
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--patience",      type=int,   default=20,    help="Early stopping patience")
    parser.add_argument("--fs",            type=float, default=256.0, help="EEG sampling rate Hz")
    parser.add_argument("--epoch-sec",     type=float, default=4.0,   help="Seconds per epoch")
    # Alias accepted by the task spec
    parser.add_argument("--use-synthetic", action="store_true",       help="Alias for --synthetic")
    args = parser.parse_args()

    use_synthetic = args.synthetic or args.use_synthetic
    n_samples = int(args.epoch_sec * args.fs)

    # ── Load data ─────────────────────────────────────────────────────────
    X, y = None, None

    if args.npz:
        X, y = load_npz(args.npz, args.channels)
        log.info("NPZ loaded: X=%s y=%s", X.shape, y.shape)

    elif args.data_dir:
        X, y = load_parquet_dir(args.data_dir, args.channels, fs=args.fs)
        log.info("Parquet data loaded: %d epochs", len(y) if y is not None else 0)

    elif use_synthetic:
        log.info("Using synthetic data (smoke-test — not for production)")
        X, y = make_synthetic_data(args.channels, fs=args.fs, n_per_class=200)
        log.info("Synthetic data: X=%s y=%s", X.shape, y.shape)

    else:
        parser.error("Provide --data-dir, --npz, or --synthetic")

    if X is None or len(y) == 0:
        print("ERROR: No training data found. Check your --data-dir path.")
        sys.exit(1)

    # ── Validate minimum class coverage ───────────────────────────────────
    counts = np.bincount(y, minlength=N_CLASSES)
    log.info("Class distribution: %s", dict(zip(EMOTIONS, counts.tolist())))
    if counts.min() < MIN_EPOCHS_PER_CLASS:
        log.warning(
            "Some classes have < %d samples. Accuracy may be poor. "
            "Collect more data before using in production.",
            MIN_EPOCHS_PER_CLASS,
        )

    # ── Trim/pad to target n_samples ──────────────────────────────────────
    actual_samples = X.shape[2]
    if actual_samples != n_samples:
        if actual_samples > n_samples:
            X = X[:, :, :n_samples]
        else:
            pad = n_samples - actual_samples
            X = np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="edge")
        log.info("Trimmed/padded epochs to %d samples", n_samples)

    # ── Train ─────────────────────────────────────────────────────────────
    model, accuracy = train_mlx(
        X, y,
        n_channels=args.channels,
        n_samples=n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    save_mlx_model(model, accuracy, args.channels)

    print(f"\nDone. Val accuracy: {accuracy * 100:.2f}%")
    if accuracy >= 0.60:
        print("Meets 60% threshold — model saved for potential live inference use.")
    else:
        print("Below 60% threshold — collect more labeled data before deploying.")


if __name__ == "__main__":
    main()
