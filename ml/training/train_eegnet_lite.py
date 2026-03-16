"""Train EEGNet-Lite for 3-class EEG emotion classification and export to ONNX.

EEGNet-Lite uses depthwise-separable 2-D convolutions to stay under 2,000
trainable parameters — small enough to export as an ONNX file < 50 KB and
run in the browser via onnxruntime-web.

Data formats accepted:
    --npz PATH      NPZ file with X (n_epochs, 4, n_samples) and y (n_epochs,)
    --data-dir DIR  Directory containing .npz or .parquet files
    --synthetic     Generate synthetic EEG-like data (smoke-test / baseline)

Saved artefacts:
    ml/models/saved/eegnet_lite_emotion.pt      PyTorch weights + config
    ml/models/saved/eegnet_lite_emotion.onnx    ONNX model (< 50 KB)
    ml/models/saved/eegnet_lite_emotion_bench.txt  Validation accuracy (float)

Classes (3-class valence):
    0 — positive  (happy, excited, calm)
    1 — neutral   (alert, focused, baseline)
    2 — negative  (sad, fearful, stressed)

Usage examples:
    # Smoke-test with synthetic data:
    python -m training.train_eegnet_lite --synthetic --epochs 30

    # Train on an NPZ dataset:
    python -m training.train_eegnet_lite --npz data/my_dataset.npz --epochs 100

    # Train on a directory of NPZ/parquet files, then export ONNX:
    python -m training.train_eegnet_lite --data-dir data/pilot --epochs 150 --export-onnx

    # GPU training on Apple Silicon:
    python -m training.train_eegnet_lite --synthetic --device mps --export-onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_ML_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ML_ROOT))

# ── Project imports (after path setup) ───────────────────────────────────────
from models.eegnet_lite import _EEGNetLiteModule, export_onnx, EMOTION_CLASSES

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

N_CLASSES  = len(EMOTION_CLASSES)   # 3
N_CHANNELS = 4
FS         = 256.0
EPOCH_SEC  = 4.0
N_SAMPLES  = int(EPOCH_SEC * FS)    # 1024
SAVED_DIR  = _ML_ROOT / "models" / "saved"


# ── Synthetic data ─────────────────────────────────────────────────────────────

def make_synthetic_data(
    n_per_class: int = 200,
    n_channels: int = N_CHANNELS,
    fs: float = FS,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic EEG-like epochs with class-specific band-power signatures.

    Class spectral profiles (based on EEG emotion correlates):
        positive (0): dominant alpha (10 Hz), moderate beta, positive FAA
        neutral  (1): balanced alpha + beta, moderate theta
        negative (2): dominant beta (18/24 Hz), suppressed alpha, negative FAA

    Args:
        n_per_class: Epochs per class.
        n_channels:  EEG channels (default 4).
        fs:          Sampling rate Hz.
        seed:        RNG seed.

    Returns:
        X: float32 array (n_per_class * n_classes, n_channels, n_samples)
        y: int64 label array (n_per_class * n_classes,)
    """
    rng = np.random.default_rng(seed)
    n_samples = int(EPOCH_SEC * fs)
    t = np.linspace(0, EPOCH_SEC, n_samples, endpoint=False)

    # (alpha_amp, beta_amp, high_beta_amp, theta_amp, noise_std)
    signatures = [
        (2.5, 0.6, 0.2, 0.5, 0.4),   # positive: dominant alpha
        (1.0, 1.0, 0.5, 0.8, 0.5),   # neutral:  balanced
        (0.4, 2.0, 1.0, 0.4, 0.6),   # negative: dominant beta
    ]

    X_list, y_list = [], []
    for label, (a_amp, b_amp, hb_amp, th_amp, noise) in enumerate(signatures):
        for _ in range(n_per_class):
            epoch = np.zeros((n_channels, n_samples), dtype=np.float32)
            for ch in range(n_channels):
                phase_a  = rng.uniform(0, 2 * np.pi)
                phase_b  = rng.uniform(0, 2 * np.pi)
                phase_hb = rng.uniform(0, 2 * np.pi)
                phase_th = rng.uniform(0, 2 * np.pi)

                # Slight FAA: ch1=AF7, ch2=AF8 — positive → more right alpha
                faa_mod = 1.0
                if ch == 2 and label == 0:   # AF8, positive: right alpha up
                    faa_mod = 1.1
                elif ch == 2 and label == 2: # AF8, negative: right alpha down
                    faa_mod = 0.9

                sig = (
                    faa_mod * a_amp  * np.sin(2 * np.pi * 10.0 * t + phase_a)
                    + b_amp          * np.sin(2 * np.pi * 18.0 * t + phase_b)
                    + hb_amp         * np.sin(2 * np.pi * 24.0 * t + phase_hb)
                    + th_amp         * np.sin(2 * np.pi * 6.0  * t + phase_th)
                    + rng.normal(0, noise, n_samples)
                )
                epoch[ch] = sig.astype(np.float32)

            X_list.append(epoch)
            y_list.append(label)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_npz(path: Path, n_channels: int = N_CHANNELS) -> Tuple[np.ndarray, np.ndarray]:
    """Load (X, y) from an NPZ file.

    Expected keys:
        X: (n_epochs, n_channels, n_samples) float array
        y: (n_epochs,) int array — labels in 0..N_CLASSES-1
    """
    data = np.load(path)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    if X.shape[1] != n_channels:
        log.warning(
            "NPZ has %d channels, expected %d — skipping %s",
            X.shape[1], n_channels, path,
        )
        return np.empty((0, n_channels, X.shape[2]), dtype=np.float32), \
               np.empty(0, dtype=np.int64)
    return X, y


def load_data_dir(
    data_dir: Path,
    n_channels: int = N_CHANNELS,
    fs: float = FS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate all NPZ / parquet files in data_dir.

    Parquet files must have columns: eeg_ch0 ... eeg_chN, label (int 0-2).
    """
    X_parts, y_parts = [], []
    n_samples = int(EPOCH_SEC * fs)

    for npz_file in sorted(data_dir.glob("**/*.npz")):
        X_part, y_part = load_npz(npz_file, n_channels)
        if len(y_part) > 0:
            X_parts.append(X_part)
            y_parts.append(y_part)
            log.info("NPZ %s: %d epochs", npz_file.name, len(y_part))

    for pq_file in sorted(data_dir.glob("**/*.parquet")):
        try:
            import pandas as pd
            df = pd.read_parquet(pq_file)
            ch_cols = [f"eeg_ch{i}" for i in range(n_channels)]
            if not all(c in df.columns for c in ch_cols) or "label" not in df.columns:
                log.debug("Skipping %s — missing columns", pq_file.name)
                continue

            raw = df[ch_cols].values.T.astype(np.float32)   # (n_channels, n_total)
            labels = df["label"].values.astype(np.int64)
            hop = n_samples // 2                              # 50% overlap

            ep_X, ep_y = [], []
            start = 0
            while start + n_samples <= raw.shape[1]:
                epoch = raw[:, start:start + n_samples]
                win_labels = labels[start:start + n_samples]
                label = int(np.bincount(win_labels, minlength=N_CLASSES).argmax())
                ep_X.append(epoch)
                ep_y.append(label)
                start += hop

            if ep_X:
                X_parts.append(np.stack(ep_X))
                y_parts.append(np.array(ep_y, dtype=np.int64))
                log.info("Parquet %s: %d epochs", pq_file.name, len(ep_y))
        except Exception as exc:
            log.warning("Failed to load %s: %s", pq_file, exc)

    if not X_parts:
        return (
            np.empty((0, n_channels, n_samples), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def normalize_epochs(X: np.ndarray) -> np.ndarray:
    """Per-channel, per-epoch z-score normalisation."""
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True) + 1e-7
    return (X - mean) / std


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: np.ndarray,
    n_channels: int = N_CHANNELS,
    F1: int = 8,
    D: int = 2,
    F2: int = 16,
    dropout: float = 0.25,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 15,
    device: str = "cpu",
) -> Tuple["_EEGNetLiteModule", float]:
    """Train EEGNet-Lite and return (best_model, best_val_accuracy).

    Uses AdamW + cosine LR schedule + early stopping with class-balanced loss.
    Input X must have shape (n_epochs, n_channels, n_samples); labels are
    added an extra dimension to (n_epochs, 1, n_channels, n_samples) here.

    Args:
        X:          (n_epochs, n_channels, n_samples) float32.
        y:          (n_epochs,) int64 labels.
        n_channels: Number of EEG channels.
        F1:         Temporal filter count.
        D:          Depth multiplier.
        F2:         Separable filter count.
        dropout:    Dropout probability.
        epochs:     Maximum training epochs.
        batch_size: Minibatch size.
        lr:         Initial learning rate.
        val_split:  Fraction held out for validation.
        patience:   Early stopping patience.
        device:     PyTorch device string.

    Returns:
        model:    Best _EEGNetLiteModule (restored to best val checkpoint).
        best_acc: Best validation accuracy.
    """
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, TensorDataset, random_split

    n_samples = X.shape[2]
    log.info(
        "EEGNet-Lite training — epochs=%d, channels=%d, samples=%d, n=%d, device=%s",
        epochs, n_channels, n_samples, len(y), device,
    )

    X = normalize_epochs(X)

    # Class-balanced loss weights
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = torch.tensor(1.0 / counts, dtype=torch.float32, device=device)

    # EEGNet-Lite expects (batch, 1, n_channels, n_samples)
    X_t = torch.from_numpy(X).unsqueeze(1)   # add channel dim
    y_t = torch.from_numpy(y)
    dataset = TensorDataset(X_t, y_t)

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = _EEGNetLiteModule(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=N_CLASSES,
        F1=F1,
        D=D,
        F2=F2,
        dropout=dropout,
    ).to(device)

    log.info("EEGNet-Lite parameters: %d", model.count_parameters())

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        # Train
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

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += len(yb)

        val_acc = correct / max(total, 1)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d  loss=%.4f  val_acc=%.1f%%",
                epoch, epochs, train_loss / max(len(train_loader), 1), val_acc * 100,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(
                    "Early stopping at epoch %d — best val_acc=%.1f%%",
                    epoch, best_val_acc * 100,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    log.info("Training complete — best val_acc=%.1f%%", best_val_acc * 100)
    return model, best_val_acc


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_artefacts(
    model: "_EEGNetLiteModule",
    accuracy: float,
    do_onnx: bool = True,
) -> None:
    """Save .pt checkpoint, accuracy benchmark, and optionally ONNX model."""
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    pt_path = SAVED_DIR / "eegnet_lite_emotion.pt"
    model.save(pt_path)
    print(f"PyTorch model  -> {pt_path}")

    bench_path = SAVED_DIR / "eegnet_lite_emotion_bench.txt"
    bench_path.write_text(f"{accuracy:.6f}")
    print(f"Benchmark      -> {bench_path}  ({accuracy * 100:.2f}%)")

    if do_onnx:
        try:
            onnx_path = SAVED_DIR / "eegnet_lite_emotion.onnx"
            export_onnx(
                model,
                onnx_path,
                n_channels=model.n_channels,
                n_samples=model.n_samples,
            )
            size_kb = onnx_path.stat().st_size / 1024
            print(f"ONNX model     -> {onnx_path}  ({size_kb:.1f} KB)")
        except Exception as exc:
            log.warning("ONNX export failed: %s", exc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Train EEGNet-Lite EEG emotion classifier (3-class valence) "
            "and optionally export to ONNX for browser/mobile deployment."
        )
    )

    # Data sources (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic EEG data for smoke-testing (not for production)",
    )
    data_group.add_argument(
        "--npz", type=Path, default=None,
        help="Single NPZ file: keys X (epochs, channels, samples) and y (labels)",
    )
    data_group.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing .npz or .parquet files",
    )

    # Architecture hyper-parameters
    parser.add_argument("--F1",      type=int,   default=8,   help="Temporal filter count (default 8)")
    parser.add_argument("--D",       type=int,   default=2,   help="Depth multiplier (default 2)")
    parser.add_argument("--F2",      type=int,   default=16,  help="Separable filter count (default 16)")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout probability (default 0.25)")
    parser.add_argument("--channels", type=int,  default=N_CHANNELS, help="Number of EEG channels")

    # Training hyper-parameters
    parser.add_argument("--epochs",      type=int,   default=100,  help="Max training epochs (default 100)")
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience",    type=int,   default=15,   help="Early stopping patience")
    parser.add_argument("--n-per-class", type=int,   default=200,  help="Synthetic epochs per class")

    # Export
    parser.add_argument(
        "--export-onnx", action="store_true",
        help="Export trained model to ONNX after training (requires torch.onnx)",
    )

    # Hardware
    parser.add_argument(
        "--device", type=str, default=None,
        help="PyTorch device: cpu | cuda | mps. Auto-detected if not set.",
    )

    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required for training. Install with: pip install torch")
        sys.exit(1)

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Using device: %s", device)

    # Load data
    if args.synthetic:
        log.info("Generating synthetic data (%d per class)...", args.n_per_class)
        X, y = make_synthetic_data(n_per_class=args.n_per_class, n_channels=args.channels)
    elif args.npz:
        X, y = load_npz(args.npz, args.channels)
    else:
        X, y = load_data_dir(args.data_dir, args.channels)

    if X is None or len(y) == 0:
        print("ERROR: No training data. Check --npz / --data-dir path or use --synthetic.")
        sys.exit(1)

    counts = np.bincount(y, minlength=N_CLASSES)
    log.info(
        "Dataset: %d epochs — class counts: %s",
        len(y),
        {EMOTION_CLASSES[i]: int(c) for i, c in enumerate(counts)},
    )

    # Train
    model, best_acc = train(
        X, y,
        n_channels=args.channels,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )

    print(f"\nBest validation accuracy: {best_acc * 100:.2f}%")
    print(f"Parameters: {model.count_parameters()}")

    # Save
    save_artefacts(model, best_acc, do_onnx=args.export_onnx)

    print("\nDone.")
    if best_acc < 0.50:
        print(
            "NOTE: Accuracy below 50% — consider more data, longer training, "
            "or use --synthetic for a smoke-test baseline."
        )


if __name__ == "__main__":
    main()
