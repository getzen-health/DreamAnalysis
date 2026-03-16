"""Train Barlow Twins EEG self-supervised encoder.

Self-supervised pre-training on *unlabeled* EEG data.  The resulting encoder
can be fine-tuned for any downstream task (emotion, sleep staging, stress) with
far less labeled data than training from scratch.

Saved files (ml/models/saved/)
───────────────────────────────
  barlow_twins_eeg.pt         — full checkpoint (encoder + projector + config)
  barlow_twins_encoder.pt     — encoder-only weights for downstream transfer

Usage
─────
  # Train on NPZ data (X.shape = (N, 4, 1024) — no labels needed):
  python -m training.train_barlow_twins --npz data/unlabeled_eeg.npz

  # Train on a directory of Parquet files:
  python -m training.train_barlow_twins --data-dir data/recordings/

  # Quick smoke-test with synthetic data:
  python -m training.train_barlow_twins --use-synthetic --epochs 10

  # Fine-tune a pre-trained encoder for emotion classification:
  python -m training.train_barlow_twins \\
      --fine-tune --npz data/labeled.npz \\
      --encoder-ckpt models/saved/barlow_twins_encoder.pt

Data format
───────────
  NPZ file:
    X   — (N, 4, 1024)  float32  raw EEG epochs
    y   — (N,)          int64    optional labels (used only in --fine-tune mode)

  Parquet directory:
    Files with columns: [eeg_ch0, eeg_ch1, eeg_ch2, eeg_ch3]
    Label column optional — ignored during self-supervised training.
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
from torch.utils.data import DataLoader, TensorDataset

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.barlow_twins_eeg import BarlowTwinsEEG, EEGEncoder

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"
N_CHANNELS = 4
N_SAMPLES = 1024     # 4 sec × 256 Hz
FS = 256.0


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load EEG epochs from an NPZ file.

    Returns:
        (X, y) where X is (N, 4, 1024) float32 and y is (N,) int64 or None.
    """
    data = np.load(path)
    X = data["X"].astype(np.float32)

    # Accept (N, 4, T) or (N, T, 4) — normalise to (N, 4, T)
    if X.ndim == 3 and X.shape[-1] == N_CHANNELS and X.shape[1] != N_CHANNELS:
        X = X.transpose(0, 2, 1)

    if X.ndim != 3 or X.shape[1] != N_CHANNELS:
        raise ValueError(
            f"Expected X.shape (N, {N_CHANNELS}, T), got {X.shape}"
        )

    # Pad / trim time axis
    n_t = X.shape[2]
    if n_t < N_SAMPLES:
        X = np.pad(X, ((0, 0), (0, 0), (0, N_SAMPLES - n_t)), mode="edge")
    elif n_t > N_SAMPLES:
        X = X[:, :, :N_SAMPLES]

    y = data["y"].astype(np.int64) if "y" in data else None
    log.info("NPZ loaded: X=%s  labels=%s", X.shape, "yes" if y is not None else "no")
    return X, y


def load_parquet_dir(data_dir: Path) -> np.ndarray:
    """Load EEG epochs from a directory of Parquet files (no labels needed).

    Returns:
        X: (N, 4, N_SAMPLES) float32
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load Parquet files: pip install pandas")

    epoch_samples = N_SAMPLES
    hop = epoch_samples // 2        # 50% overlap

    ch_cols = [f"eeg_ch{i}" for i in range(N_CHANNELS)]
    X_list = []

    for pq_file in sorted(data_dir.glob("**/*.parquet")):
        try:
            df = pd.read_parquet(pq_file)
            if not all(c in df.columns for c in ch_cols):
                log.debug("Skipping %s — missing EEG columns", pq_file.name)
                continue
            raw = df[ch_cols].values.T.astype(np.float32)   # (4, n_total)
            n_total = raw.shape[1]
            start = 0
            while start + epoch_samples <= n_total:
                X_list.append(raw[:, start:start + epoch_samples])
                start += hop
        except Exception as exc:
            log.warning("Failed to load %s: %s", pq_file, exc)

    if not X_list:
        raise ValueError(f"No EEG epochs found in {data_dir}")

    X = np.stack(X_list, axis=0)
    log.info("Parquet dir loaded: %d epochs from %s", len(X), data_dir)
    return X


def make_synthetic_data(n_per_state: int = 200) -> np.ndarray:
    """Generate synthetic EEG-like data for smoke-testing the pipeline.

    Creates unlabeled epochs with diverse band-power signatures mimicking
    different mental states. No labels are required for self-supervised training.
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4.0, N_SAMPLES, endpoint=False)
    X_list = []

    # (alpha_amp, beta_amp, theta_amp, noise_std)
    state_sigs = [
        (2.0, 0.8, 0.5, 0.3),   # relaxed (high alpha)
        (0.5, 2.0, 0.5, 0.8),   # focused (high beta)
        (0.5, 0.3, 2.0, 0.6),   # drowsy  (high theta)
        (0.3, 1.5, 0.4, 0.7),   # stressed (high beta, low alpha)
        (2.5, 0.3, 1.0, 0.2),   # meditative
    ]

    for a_amp, b_amp, th_amp, noise in state_sigs:
        for _ in range(n_per_state):
            epoch = np.zeros((N_CHANNELS, N_SAMPLES), dtype=np.float32)
            for ch in range(N_CHANNELS):
                phase = rng.uniform(0, 2 * np.pi)
                sig = (
                    a_amp  * np.sin(2 * np.pi * 10 * t + phase)
                    + b_amp  * np.sin(2 * np.pi * 20 * t + phase)
                    + th_amp * np.sin(2 * np.pi * 6  * t + phase)
                    + rng.normal(0, noise, N_SAMPLES)
                )
                epoch[ch] = sig.astype(np.float32)
            X_list.append(epoch)

    X = np.stack(X_list)
    idx = rng.permutation(len(X))
    log.info("Synthetic data generated: %d epochs", len(X))
    return X[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def normalize_epochs(X: np.ndarray) -> np.ndarray:
    """Z-score each epoch independently (per-channel zero-mean, unit-variance)."""
    mu  = X.mean(axis=-1, keepdims=True)
    sd  = X.std(axis=-1, keepdims=True) + 1e-7
    return (X - mu) / sd


# ══════════════════════════════════════════════════════════════════════════════
# Self-supervised training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_self_supervised(
    X: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    lambda_: float = 0.005,
    device: str = "cpu",
    log_every: int = 10,
) -> Tuple[BarlowTwinsEEG, list]:
    """Train a BarlowTwinsEEG model on unlabeled EEG epochs.

    Args:
        X:            (N, 4, 1024) float32 normalised EEG epochs
        epochs:       number of training epochs
        batch_size:   mini-batch size
        lr:           initial learning rate (AdamW)
        weight_decay: L2 regularisation weight
        lambda_:      Barlow Twins off-diagonal loss weight
        device:       torch device string
        log_every:    log loss every N epochs

    Returns:
        (model, loss_history) — BarlowTwinsEEG in eval mode, list of epoch losses
    """
    log.info(
        "Barlow Twins training — n_epochs=%d  batch=%d  lr=%.0e  lambda=%.4f  device=%s",
        len(X), batch_size, lr, lambda_, device,
    )

    X_t = torch.from_numpy(X.astype(np.float32))
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = BarlowTwinsEEG(
        n_channels=N_CHANNELS,
        n_samples=N_SAMPLES,
        lambda_=lambda_,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            loss, _ = model.training_step(xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if epoch % log_every == 0 or epoch == 1:
            log.info("Epoch %3d/%d  loss=%.6f", epoch, epochs, avg_loss)

    model.eval()
    log.info("Barlow Twins pre-training complete. Final loss=%.6f", loss_history[-1])
    return model, loss_history


# ══════════════════════════════════════════════════════════════════════════════
# Linear probe fine-tuning (optional downstream evaluation)
# ══════════════════════════════════════════════════════════════════════════════

def fine_tune_linear_probe(
    encoder: EEGEncoder,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = "cpu",
) -> Tuple[nn.Module, float]:
    """Train a linear classification head on top of a frozen encoder.

    This is the standard Barlow Twins downstream evaluation protocol: freeze
    the encoder entirely and train only a single linear layer.

    Args:
        encoder:   pre-trained EEGEncoder (weights frozen)
        X:         (N, 4, 1024) float32 EEG epochs
        y:         (N,) int64 class labels
        n_classes: number of output classes
        epochs:    training epochs for the linear head
        batch_size:mini-batch size
        lr:        linear head learning rate
        val_split: fraction of data held out for validation
        device:    torch device string

    Returns:
        (classifier, val_accuracy) where classifier is a simple nn.Linear head.
    """
    log.info("Linear probe fine-tuning — n_samples=%d  n_classes=%d", len(X), n_classes)

    encoder.eval()
    encoder.to(device)

    # Pre-compute embeddings (frozen encoder — no backprop needed)
    X_norm = normalize_epochs(X)
    X_t = torch.from_numpy(X_norm.astype(np.float32)).to(device)
    with torch.no_grad():
        embeddings = encoder(X_t).cpu()         # (N, enc_dim)

    y_t = torch.from_numpy(y.astype(np.int64))
    dataset = TensorDataset(embeddings, y_t)

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    enc_dim = embeddings.shape[1]
    head = nn.Linear(enc_dim, n_classes).to(device)
    optimizer = AdamW(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        head.train()
        for emb_b, yb in train_loader:
            emb_b, yb = emb_b.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(head(emb_b), yb)
            loss.backward()
            optimizer.step()

        head.eval()
        correct = total = 0
        with torch.no_grad():
            for emb_b, yb in val_loader:
                emb_b, yb = emb_b.to(device), yb.to(device)
                preds = head(emb_b).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)

        val_acc = correct / max(total, 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if epoch % 10 == 0:
            log.info("Linear probe epoch %3d/%d  val_acc=%.2f%%", epoch, epochs, val_acc * 100)

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    log.info("Linear probe done — best val_acc=%.2f%%", best_val_acc * 100)
    return head, best_val_acc


# ══════════════════════════════════════════════════════════════════════════════
# Save helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    model: BarlowTwinsEEG,
    loss_history: list,
    val_acc: Optional[float] = None,
) -> None:
    """Save model checkpoints and optional benchmark file."""
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    full_path = SAVED_DIR / "barlow_twins_eeg.pt"
    model.save(full_path)
    print(f"Full checkpoint saved   → {full_path}")

    enc_path = SAVED_DIR / "barlow_twins_encoder.pt"
    model.save_encoder(enc_path)
    print(f"Encoder-only saved      → {enc_path}")

    loss_path = SAVED_DIR / "barlow_twins_loss_history.npy"
    np.save(loss_path, np.array(loss_history))
    print(f"Loss history saved      → {loss_path}")

    if val_acc is not None:
        bm_path = SAVED_DIR / "barlow_twins_linear_probe.txt"
        bm_path.write_text(f"{val_acc:.6f}")
        print(f"Linear probe accuracy   → {bm_path}  ({val_acc*100:.2f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Barlow Twins self-supervised EEG pre-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--npz", type=Path, metavar="FILE",
        help="NPZ file with X (N, 4, 1024) array",
    )
    data_group.add_argument(
        "--data-dir", type=Path, metavar="DIR",
        help="Directory of Parquet files",
    )
    data_group.add_argument(
        "--use-synthetic", action="store_true",
        help="Generate synthetic training data (smoke-test)",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=200,   help="Number of pre-training epochs (default 200)")
    parser.add_argument("--batch-size",   type=int,   default=64,    help="Batch size (default 64)")
    parser.add_argument("--lr",           type=float, default=3e-4,  help="AdamW learning rate (default 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,  help="AdamW weight decay (default 1e-4)")
    parser.add_argument("--lambda",       type=float, default=0.005, dest="lambda_",
                        help="Off-diagonal loss weight (default 0.005)")
    parser.add_argument("--log-every",    type=int,   default=10,    help="Log every N epochs (default 10)")

    # ── Downstream evaluation ─────────────────────────────────────────────────
    parser.add_argument(
        "--fine-tune", action="store_true",
        help="After pre-training, run linear probe evaluation (requires labels in NPZ)",
    )
    parser.add_argument(
        "--n-classes", type=int, default=6,
        help="Number of downstream classes for linear probe (default 6)",
    )
    parser.add_argument(
        "--probe-epochs", type=int, default=50,
        help="Linear probe training epochs (default 50)",
    )

    # ── Hardware ──────────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default=None,
                        help="torch device (default: auto-detect cuda/mps/cpu)")

    args = parser.parse_args()

    # ── Device selection ──────────────────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Using device: %s", device)

    # ── Load data ─────────────────────────────────────────────────────────────
    y = None
    if args.npz:
        X, y = load_npz(args.npz)
    elif args.data_dir:
        X = load_parquet_dir(args.data_dir)
    elif args.use_synthetic:
        log.info("Using synthetic data (smoke-test — not for production)")
        X = make_synthetic_data(n_per_state=200)
    else:
        parser.error("Provide --npz, --data-dir, or --use-synthetic")

    X = normalize_epochs(X)
    log.info("Dataset ready: %d epochs  (%d channels × %d samples)", len(X), X.shape[1], X.shape[2])

    # ── Self-supervised pre-training ──────────────────────────────────────────
    model, loss_history = train_self_supervised(
        X,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_=args.lambda_,
        device=device,
        log_every=args.log_every,
    )

    # ── Optional linear probe evaluation ─────────────────────────────────────
    val_acc = None
    if args.fine_tune:
        if y is None:
            log.warning("--fine-tune requested but no labels found in data. Skipping linear probe.")
        else:
            _, val_acc = fine_tune_linear_probe(
                model.encoder, X, y,
                n_classes=args.n_classes,
                epochs=args.probe_epochs,
                device=device,
            )
            print(f"\nLinear probe val accuracy: {val_acc * 100:.2f}%")
            if val_acc >= 0.60:
                print("  Meets 60% threshold — encoder is transferable.")
            else:
                print("  Below 60% — more unlabeled data or longer pre-training may help.")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(model, loss_history, val_acc)

    # ── Summary ───────────────────────────────────────────────────────────────
    params = model.param_count()
    print(f"\nModel summary:")
    print(f"  Encoder params   : {params['encoder']:,}")
    print(f"  Projector params : {params['projector']:,}")
    print(f"  Total params     : {params['total']:,}")
    print(f"  Final loss       : {loss_history[-1]:.6f}")
    print(f"\nTransfer learning:")
    print(f"  Load encoder with: BarlowTwinsEEG.load_encoder_only('{SAVED_DIR}/barlow_twins_encoder.pt')")
    print(f"  Embedding shape  : (batch, {model.enc_dim})")


if __name__ == "__main__":
    main()
