"""Self-supervised contrastive EEG encoder for 4-channel consumer EEG.

SimCLR-style encoder that learns EEG representations without emotion labels.
Architecture: 3-layer 1D-CNN -> global average pooling -> projection head.

Training uses NT-Xent contrastive loss:
    - Positive pairs: two augmented views of the same EEG epoch
    - Negative pairs: epochs from different time windows
    - Augmentations: Gaussian noise, temporal shift, channel dropout, amplitude scaling

References:
    - MDPI Mathematics (2025): Self-supervised framework on 4-channel EEG
      achieves +20.4% accuracy over conventional features
    - Chen et al. (2020): SimCLR — A Simple Framework for Contrastive Learning
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

log = logging.getLogger(__name__)


# ---------- EEG Augmentations for Contrastive Learning ----------


def augment_eeg(x: "torch.Tensor", rng: Optional[random.Random] = None) -> "torch.Tensor":
    """Apply random augmentations to create positive pairs for contrastive learning.

    Args:
        x: (n_channels, n_samples) tensor — single EEG epoch.
        rng: Random number generator for reproducibility.

    Returns:
        Augmented copy of x (same shape).
    """
    if rng is None:
        rng = random.Random()

    aug = x.clone()

    # Gaussian noise (SNR ~20dB)
    if rng.random() > 0.5:
        signal_power = aug.pow(2).mean().clamp(min=1e-10)
        noise_power = signal_power / (10 ** (20.0 / 10.0))
        aug = aug + torch.randn_like(aug) * noise_power.sqrt()

    # Temporal shift (+/- 50 samples)
    if rng.random() > 0.5:
        shift = rng.randint(-50, 50)
        aug = torch.roll(aug, shift, dims=-1)

    # Channel dropout (zero one random channel)
    if rng.random() > 0.3:
        ch = rng.randint(0, aug.shape[0] - 1)
        aug[ch] = 0.0

    # Amplitude scaling (0.8 - 1.2x)
    if rng.random() > 0.5:
        scale = 0.8 + rng.random() * 0.4
        aug = aug * scale

    return aug


# ---------- Contrastive Encoder ----------


if TORCH_AVAILABLE:

    class EEGContrastiveEncoder(nn.Module):
        """Self-supervised 1D-CNN encoder for 4-channel EEG.

        Produces a fixed-size embedding from variable-length (but typically 1024-sample)
        EEG epochs. The projection head is only used during contrastive pretraining;
        for downstream tasks, use encode() to get the representation.

        Args:
            n_channels: Number of EEG channels (4 for Muse 2).
            n_samples: Expected number of time samples per epoch (1024 = 4s @ 256Hz).
            embed_dim: Dimension of the learned representation.
            proj_dim: Dimension of the projection head output (for NT-Xent loss).
        """

        def __init__(
            self,
            n_channels: int = 4,
            n_samples: int = 1024,
            embed_dim: int = 64,
            proj_dim: int = 32,
        ):
            super().__init__()
            self.n_channels = n_channels
            self.n_samples = n_samples
            self.embed_dim = embed_dim
            self.proj_dim = proj_dim

            # Encoder: 3-layer 1D-CNN with batch norm and ELU
            self.encoder = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=25, padding=12),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.AvgPool1d(4),
                nn.Conv1d(32, 64, kernel_size=15, padding=7),
                nn.BatchNorm1d(64),
                nn.ELU(),
                nn.AvgPool1d(4),
                nn.Conv1d(64, embed_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(embed_dim),
                nn.ELU(),
                nn.AdaptiveAvgPool1d(1),
            )

            # Projection head (only used during pretraining)
            self.projector = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, proj_dim),
            )

            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Full forward: encoder + projection head.

            Args:
                x: (batch, n_channels, n_samples) tensor.

            Returns:
                h: (batch, embed_dim) — representation (use for downstream tasks).
                z: (batch, proj_dim) — projection (use for contrastive loss).
            """
            h = self.encoder(x).squeeze(-1)  # (batch, embed_dim)
            z = self.projector(h)  # (batch, proj_dim)
            return h, z

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            """Encode-only forward pass (no projection head).

            Args:
                x: (batch, n_channels, n_samples) tensor.

            Returns:
                h: (batch, embed_dim) representation.
            """
            return self.encoder(x).squeeze(-1)

        def save(self, path: str | Path) -> None:
            """Save encoder + projector weights and config."""
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "config": {
                        "n_channels": self.n_channels,
                        "n_samples": self.n_samples,
                        "embed_dim": self.embed_dim,
                        "proj_dim": self.proj_dim,
                    },
                },
                path,
            )
            log.info("EEGContrastiveEncoder saved -> %s", path)

        @classmethod
        def load(cls, path: str | Path) -> "EEGContrastiveEncoder":
            """Load encoder from checkpoint."""
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = cls(**ckpt["config"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            log.info("EEGContrastiveEncoder loaded <- %s", path)
            return model

else:
    # Stub for when PyTorch is not available
    class EEGContrastiveEncoder:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.embed_dim = kwargs.get("embed_dim", 64)
            self.proj_dim = kwargs.get("proj_dim", 32)

        def encode(self, x):
            raise RuntimeError("PyTorch required for EEGContrastiveEncoder")


# ---------- NT-Xent Contrastive Loss ----------


def nt_xent_loss(
    z1: "torch.Tensor",
    z2: "torch.Tensor",
    temperature: float = 0.5,
) -> "torch.Tensor":
    """Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent / SimCLR loss).

    Args:
        z1: (batch, proj_dim) projections from augmented view 1.
        z2: (batch, proj_dim) projections from augmented view 2.
        temperature: Softmax temperature (lower = sharper).

    Returns:
        Scalar loss tensor.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.shape[0]

    # Concatenate: [z1; z2] -> (2*batch, proj_dim)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    # Mask self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pair labels: (i, i+batch) and (i+batch, i)
    labels = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ]
    )

    return F.cross_entropy(sim, labels)
