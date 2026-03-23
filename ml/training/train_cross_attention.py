"""Train cross-attention fusion model for EEG + Voice emotion.

Uses knowledge distillation with synthetic paired data:
- Sample EEG probs from Dirichlet(6) and voice probs from Dirichlet(5)
- Teacher label: argmax of the more confident prediction
- When both agree, strengthen; when they disagree, attend to the confident one

After training, exports to both .pt (PyTorch) and .onnx formats.

Usage:
    cd ml
    python3 -m training.train_cross_attention
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure ml/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cross_attention_fusion import CrossModalFusion, export_to_onnx

log = logging.getLogger(__name__)

_ML_ROOT = Path(__file__).resolve().parent.parent
_SAVE_DIR = _ML_ROOT / "models" / "saved"


def generate_synthetic_data(
    n_samples: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic paired EEG+Voice probability distributions.

    Returns:
        eeg_probs: (N, 6) -- Dirichlet-sampled 6-class EEG distributions
        voice_probs: (N, 5) -- Dirichlet-sampled 5-class voice distributions
        labels: (N,) -- teacher labels (argmax of more confident modality)
    """
    rng = np.random.default_rng(seed)

    # Sample probability distributions
    eeg_probs = rng.dirichlet(np.ones(6) * 2.0, n_samples).astype(np.float32)
    voice_probs = rng.dirichlet(np.ones(5) * 2.0, n_samples).astype(np.float32)

    # Teacher labels: argmax of the more confident prediction
    labels = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        eeg_conf = eeg_probs[i].max()
        voice_conf = voice_probs[i].max()
        if eeg_conf > voice_conf:
            labels[i] = int(eeg_probs[i].argmax())
        else:
            # Voice has 5 classes -- map to 6-class (cap at 5)
            labels[i] = min(int(voice_probs[i].argmax()), 5)

    return eeg_probs, voice_probs, labels


def train(
    n_samples: int = 5000,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    n_heads: int = 2,
    seed: int = 42,
) -> CrossModalFusion:
    """Train the CrossModalFusion model.

    Args:
        n_samples: Number of synthetic training samples.
        n_epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        hidden_dim: Attention hidden dimension.
        n_heads: Number of attention heads.
        seed: Random seed.

    Returns:
        Trained CrossModalFusion model.
    """
    torch.manual_seed(seed)

    log.info("Generating %d synthetic training samples...", n_samples)
    eeg_probs, voice_probs, labels = generate_synthetic_data(n_samples, seed)

    # Split: 80% train, 20% val
    split = int(0.8 * n_samples)
    train_ds = TensorDataset(
        torch.from_numpy(eeg_probs[:split]),
        torch.from_numpy(voice_probs[:split]),
        torch.from_numpy(labels[:split]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(eeg_probs[split:]),
        torch.from_numpy(voice_probs[split:]),
        torch.from_numpy(labels[split:]),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CrossModalFusion(
        eeg_dim=6,
        voice_dim=5,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_classes=6,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for eeg, voice, target in train_loader:
            optimizer.zero_grad()
            logits = model(eeg, voice)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * eeg.size(0)
            train_correct += (logits.argmax(dim=-1) == target).sum().item()
            train_total += eeg.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for eeg, voice, target in val_loader:
                logits = model(eeg, voice)
                val_correct += (logits.argmax(dim=-1) == target).sum().item()
                val_total += eeg.size(0)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d  train_loss=%.4f  train_acc=%.1f%%  val_acc=%.1f%%",
                epoch, n_epochs,
                train_loss / train_total,
                train_acc, val_acc,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Loaded best model state with val_acc=%.1f%%", best_val_acc)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train cross-attention EEG+Voice fusion model"
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    model = train(
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        seed=args.seed,
    )

    # Save PyTorch model
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = _SAVE_DIR / "cross_attention_fusion.pt"
    torch.save(model.state_dict(), str(pt_path))
    log.info("Saved PyTorch model to %s", pt_path)

    # Export to ONNX
    onnx_path = _SAVE_DIR / "cross_attention_fusion.onnx"
    export_to_onnx(model, str(onnx_path))

    # Also copy ONNX to public/models for client-side inference
    public_models = _ML_ROOT.parent / "public" / "models"
    if public_models.exists():
        public_onnx = public_models / "cross_attention_fusion.onnx"
        import shutil
        shutil.copy2(str(onnx_path), str(public_onnx))
        log.info("Copied ONNX model to %s", public_onnx)
    else:
        log.info("public/models/ not found, skipping client-side copy")

    log.info("Training complete.")


if __name__ == "__main__":
    main()
