"""Self-supervised pretraining on unlabeled EEG data.

Uses all available EEG sessions (no labels needed) or synthetic EEG to pretrain
the EEGContrastiveEncoder. Produces a pretrained encoder that can be fine-tuned
for downstream emotion classification.

Usage:
    python -m training.pretrain_eeg_ssl [--n_epochs 100] [--batch_size 64] [--lr 3e-4]

Output:
    models/saved/eeg_ssl_encoder.pt — pretrained encoder checkpoint
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add ml/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.eeg_ssl import EEGContrastiveEncoder, augment_eeg, nt_xent_loss

log = logging.getLogger(__name__)

# ---------- Synthetic EEG Generation for Pretraining ----------

# Frequency profiles for diverse synthetic EEG patterns
_PRETRAINING_STATES = {
    "rest_alpha_dominant": {"delta": 0.10, "theta": 0.15, "alpha": 0.50, "beta": 0.15, "gamma": 0.10},
    "focus_beta_dominant": {"delta": 0.05, "theta": 0.10, "alpha": 0.15, "beta": 0.50, "gamma": 0.20},
    "drowsy_theta_high": {"delta": 0.20, "theta": 0.40, "alpha": 0.20, "beta": 0.10, "gamma": 0.10},
    "deep_sleep_delta": {"delta": 0.65, "theta": 0.20, "alpha": 0.05, "beta": 0.05, "gamma": 0.05},
    "stress_high_beta": {"delta": 0.05, "theta": 0.05, "alpha": 0.10, "beta": 0.55, "gamma": 0.25},
    "meditation_theta": {"delta": 0.10, "theta": 0.45, "alpha": 0.30, "beta": 0.10, "gamma": 0.05},
    "relaxed_alpha": {"delta": 0.10, "theta": 0.10, "alpha": 0.55, "beta": 0.15, "gamma": 0.10},
    "anxious": {"delta": 0.05, "theta": 0.10, "alpha": 0.05, "beta": 0.50, "gamma": 0.30},
    "creative": {"delta": 0.05, "theta": 0.35, "alpha": 0.35, "beta": 0.15, "gamma": 0.10},
    "alert": {"delta": 0.05, "theta": 0.10, "alpha": 0.20, "beta": 0.40, "gamma": 0.25},
}

_BAND_CENTERS = {
    "delta": 2.0,
    "theta": 6.0,
    "alpha": 10.0,
    "beta": 22.0,
    "gamma": 40.0,
}


def _generate_epoch(
    profile: dict,
    n_channels: int = 4,
    n_samples: int = 1024,
    fs: float = 256.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single synthetic EEG epoch from a frequency profile.

    Args:
        profile: Dict mapping band names to relative power fractions.
        n_channels: Number of channels.
        n_samples: Number of time samples.
        fs: Sampling rate.
        rng: NumPy random generator.

    Returns:
        (n_channels, n_samples) array.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(n_samples) / fs
    epoch = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        signal = np.zeros(n_samples)
        for band_name, rel_power in profile.items():
            center = _BAND_CENTERS.get(band_name, 10.0)
            amplitude = np.sqrt(rel_power) * 20.0  # scale to ~20 uV RMS

            # Add 3 oscillators per band for realism
            for _ in range(3):
                freq = center + rng.uniform(-2.0, 2.0)
                phase = rng.uniform(0, 2 * np.pi)
                amp_var = amplitude * rng.uniform(0.5, 1.5) / np.sqrt(3)
                signal += amp_var * np.sin(2 * np.pi * freq * t + phase)

        # Add pink noise background
        noise = rng.standard_normal(n_samples) * 3.0
        signal += noise

        # Per-channel amplitude variation
        signal *= rng.uniform(0.7, 1.3)

        epoch[ch] = signal

    return epoch


def generate_pretraining_data(
    n_epochs: int = 10000,
    n_channels: int = 4,
    n_samples: int = 1024,
    fs: float = 256.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate diverse synthetic EEG epochs for SSL pretraining.

    Creates epochs from multiple brain state profiles with random
    variations, producing a dataset that covers the expected feature
    space of real consumer EEG.

    Args:
        n_epochs: Total number of epochs to generate.
        n_channels: Number of channels per epoch.
        n_samples: Number of time samples per epoch.
        fs: Sampling rate.
        seed: Random seed for reproducibility.

    Returns:
        (n_epochs, n_channels, n_samples) array.
    """
    rng = np.random.default_rng(seed)
    states = list(_PRETRAINING_STATES.values())
    data = np.zeros((n_epochs, n_channels, n_samples))

    for i in range(n_epochs):
        # Pick a random state profile
        profile = states[rng.integers(len(states))]

        # Add random variation to the profile
        varied = {}
        for band, power in profile.items():
            varied[band] = max(0.01, power + rng.uniform(-0.1, 0.1))

        # Normalize to sum to 1
        total = sum(varied.values())
        varied = {k: v / total for k, v in varied.items()}

        data[i] = _generate_epoch(varied, n_channels, n_samples, fs, rng)

    return data


def pretrain(
    n_training_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    n_data_epochs: int = 10000,
    n_channels: int = 4,
    n_samples: int = 1024,
    temperature: float = 0.5,
    save_path: str | None = None,
    seed: int = 42,
) -> dict:
    """Run self-supervised contrastive pretraining.

    Args:
        n_training_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        n_data_epochs: Number of synthetic data epochs to generate.
        n_channels: EEG channels.
        n_samples: Samples per epoch.
        temperature: NT-Xent temperature.
        save_path: Where to save the pretrained encoder. Defaults to models/saved/eeg_ssl_encoder.pt.
        seed: Random seed.

    Returns:
        Dict with training stats: final_loss, loss_history, n_epochs_trained, save_path.
    """
    if not TORCH_AVAILABLE:
        return {
            "error": "PyTorch required for pretraining",
            "torch_available": False,
        }

    import random as py_random

    if save_path is None:
        save_path = str(Path(__file__).parent.parent / "models" / "saved" / "eeg_ssl_encoder.pt")

    log.info("Generating %d synthetic EEG epochs for pretraining...", n_data_epochs)
    data = generate_pretraining_data(
        n_epochs=n_data_epochs,
        n_channels=n_channels,
        n_samples=n_samples,
        seed=seed,
    )

    encoder = EEGContrastiveEncoder(
        n_channels=n_channels,
        n_samples=n_samples,
        embed_dim=64,
        proj_dim=32,
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_training_epochs)

    rng = py_random.Random(seed)
    loss_history = []
    N = len(data)

    log.info("Starting pretraining: %d epochs, batch_size=%d, lr=%g", n_training_epochs, batch_size, lr)

    for epoch in range(n_training_epochs):
        encoder.train()
        indices = np.random.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            batch_idx = indices[i : i + batch_size]
            if len(batch_idx) < 2:
                continue  # NT-Xent needs at least 2 samples

            batch = torch.FloatTensor(data[batch_idx])

            # Create two augmented views
            aug1 = torch.stack([augment_eeg(b, rng) for b in batch])
            aug2 = torch.stack([augment_eeg(b, rng) for b in batch])

            # Forward pass
            _, z1 = encoder(aug1)
            _, z2 = encoder(aug2)

            # Contrastive loss
            loss = nt_xent_loss(z1, z2, temperature=temperature)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(
                "Epoch %d/%d — loss: %.4f — lr: %.2e",
                epoch + 1,
                n_training_epochs,
                avg_loss,
                scheduler.get_last_lr()[0],
            )

    # Save pretrained encoder
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    encoder.save(save_path)
    log.info("Pretrained encoder saved -> %s", save_path)

    return {
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "n_epochs_trained": n_training_epochs,
        "n_data_epochs": n_data_epochs,
        "save_path": save_path,
        "embed_dim": 64,
        "proj_dim": 32,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Pretrain EEG SSL encoder")
    parser.add_argument("--n_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n_data", type=int, default=10000, help="Number of synthetic data epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    result = pretrain(
        n_training_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_data_epochs=args.n_data,
        seed=args.seed,
    )

    print(f"\nPretraining complete:")
    print(f"  Final loss: {result.get('final_loss', 'N/A'):.4f}")
    print(f"  Epochs: {result.get('n_epochs_trained', 0)}")
    print(f"  Saved to: {result.get('save_path', 'N/A')}")
