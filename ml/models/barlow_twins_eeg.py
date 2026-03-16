"""Barlow Twins self-supervised learning for EEG representation learning.

Implements Zbontar et al. (2021) "Barlow Twins: Self-Supervised Learning via
Redundancy Reduction" (https://arxiv.org/abs/2103.03230) adapted for 4-channel
Muse 2 EEG signals.

Core idea
─────────
Two augmented views of the same EEG epoch are fed through a shared encoder.
The cross-correlation matrix of the resulting embeddings is driven toward the
identity matrix:
  - On-diagonal terms → 1  (invariance: the two views agree on each dimension)
  - Off-diagonal terms → 0  (redundancy reduction: embeddings are decorrelated)

No negative pairs, no asymmetric architecture, no stop-gradient tricks.
Collapse is prevented purely by the redundancy-reduction objective.

Architecture
────────────
Input: (batch, 4, 1024) — 4 channels × 4 sec @ 256 Hz

EEGEncoder (1D-CNN backbone):
  Conv1d(4,  32, 64, stride=16) + BN + ReLU  →  (batch, 32,  64)
  Conv1d(32, 64, 16, stride=4)  + BN + ReLU  →  (batch, 64,  16)
  Conv1d(64,128,  8, stride=2)  + BN + ReLU  →  (batch, 128,  8)
  AdaptiveAvgPool1d(1) → Flatten             →  (batch, 128)

ProjectionHead (MLP):
  Linear(128, 256) + BN + ReLU
  Linear(256, 256) + BN + ReLU
  Linear(256, 128)                           →  (batch, 128)

Loss: Barlow Twins cross-correlation loss
  C_ii → 1   (on-diagonal invariance term)
  C_ij → 0   (off-diagonal redundancy term, weighted by lambda)
  Default lambda = 0.005 (from original paper)

EEG augmentations (applied independently to create two views)
─────────────────────────────────────────────────────────────
  temporal_jitter   : random crop of 90-100% signal length, pad back
  channel_dropout   : zero out one random channel with probability 0.2
  gaussian_noise    : add N(0, 0.1*std) to the signal
  amplitude_scaling : multiply by Uniform(0.8, 1.2) per channel
  time_masking      : zero out a random contiguous 10% block

After self-supervised pre-training the EEGEncoder is used as a feature extractor
for downstream tasks (emotion, sleep staging, stress detection) via transfer
learning, bypassing the need for large labeled datasets.

Usage
─────
  # Self-supervised training
  from models.barlow_twins_eeg import BarlowTwinsEEG, EEGAugmentations
  model = BarlowTwinsEEG()
  loss = model.training_step(batch_eeg)

  # Extract embeddings for downstream use
  encoder = model.encoder
  embedding = encoder(eeg_tensor)   # (batch, 128)

  # Save / load
  model.save("models/saved/barlow_twins_eeg.pt")
  model = BarlowTwinsEEG.load("models/saved/barlow_twins_eeg.pt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ── Default hyperparameters ────────────────────────────────────────────────────

_DEFAULT_N_CHANNELS = 4
_DEFAULT_N_SAMPLES = 1024        # 4 sec @ 256 Hz
_DEFAULT_ENC_DIM = 128           # encoder output dimension
_DEFAULT_PROJ_DIM = 128          # projection head output dimension
_DEFAULT_PROJ_HIDDEN = 256       # projection head hidden dimension
_DEFAULT_LAMBDA = 0.005          # off-diagonal weight (original paper value)


# ══════════════════════════════════════════════════════════════════════════════
# EEG-specific data augmentations
# ══════════════════════════════════════════════════════════════════════════════

class EEGAugmentations:
    """Stochastic augmentation pipeline for self-supervised EEG learning.

    Each call to __call__ produces an independently augmented view of the input.
    All augmentations operate on tensors of shape (batch, n_channels, n_samples).

    Augmentation rationale
    ──────────────────────
    temporal_jitter   : simulates variable epoch alignment / boundary jitter
    channel_dropout   : teaches the encoder to rely on multi-channel patterns
                        rather than a single electrode
    gaussian_noise    : models electrode impedance drift and amplifier noise
    amplitude_scaling : normalises amplitude variability across recording sessions
    time_masking      : SpecAugment-style masking; forces robust temporal coding
    """

    def __init__(
        self,
        p_channel_dropout: float = 0.20,
        noise_std: float = 0.10,
        amp_low: float = 0.80,
        amp_high: float = 1.20,
        mask_frac: float = 0.10,
        jitter_frac_min: float = 0.90,
    ) -> None:
        self.p_channel_dropout = p_channel_dropout
        self.noise_std = noise_std
        self.amp_low = amp_low
        self.amp_high = amp_high
        self.mask_frac = mask_frac
        self.jitter_frac_min = jitter_frac_min

    # ── Individual augmentations ───────────────────────────────────────────────

    def temporal_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Random crop of 90-100% of signal length, zero-padded back.

        Args:
            x: (batch, channels, time)

        Returns:
            Augmented tensor with the same shape as x.
        """
        n = x.shape[-1]
        frac = torch.empty(1).uniform_(self.jitter_frac_min, 1.0).item()
        crop_len = max(1, int(n * frac))
        max_start = n - crop_len
        start = torch.randint(0, max(1, max_start + 1), (1,)).item()
        cropped = x[..., start:start + crop_len]
        # Pad back to original length with zeros on the right
        pad_right = n - crop_len
        return F.pad(cropped, (0, pad_right))

    def channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out one randomly selected channel with probability p.

        Args:
            x: (batch, channels, time)

        Returns:
            Augmented tensor (same shape).
        """
        if torch.rand(1).item() >= self.p_channel_dropout:
            return x
        n_ch = x.shape[1]
        ch_idx = torch.randint(0, n_ch, (1,)).item()
        out = x.clone()
        out[:, ch_idx, :] = 0.0
        return out

    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add zero-mean Gaussian noise scaled to signal standard deviation.

        Args:
            x: (batch, channels, time)

        Returns:
            Noisy tensor (same shape).
        """
        sig_std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        noise = torch.randn_like(x) * (self.noise_std * sig_std)
        return x + noise

    def amplitude_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply each channel by an independent Uniform(amp_low, amp_high) scalar.

        Args:
            x: (batch, channels, time)

        Returns:
            Scaled tensor (same shape).
        """
        batch, n_ch, n_t = x.shape
        scale = torch.empty(batch, n_ch, 1, device=x.device).uniform_(
            self.amp_low, self.amp_high
        )
        return x * scale

    def time_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out a random contiguous block of mask_frac × n_samples.

        Args:
            x: (batch, channels, time)

        Returns:
            Masked tensor (same shape).
        """
        n = x.shape[-1]
        mask_len = max(1, int(n * self.mask_frac))
        max_start = n - mask_len
        start = torch.randint(0, max(1, max_start + 1), (1,)).item()
        out = x.clone()
        out[..., start:start + mask_len] = 0.0
        return out

    # ── Composed pipeline ──────────────────────────────────────────────────────

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations in sequence.

        Args:
            x: EEG tensor (batch, channels, time)

        Returns:
            Augmented view with the same shape.
        """
        x = self.temporal_jitter(x)
        x = self.channel_dropout(x)
        x = self.gaussian_noise(x)
        x = self.amplitude_scaling(x)
        x = self.time_masking(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 1D-CNN Encoder
# ══════════════════════════════════════════════════════════════════════════════

class EEGEncoder(nn.Module):
    """1D convolutional encoder for 4-channel Muse 2 EEG.

    Operates directly on raw or lightly preprocessed EEG.

    Input:  (batch, n_channels=4, n_samples=1024)
    Output: (batch, enc_dim=128)

    Three strided conv blocks progressively compress the time axis:
      Conv1d(4,  32, 64, s=16)  →  64 time steps   (~250 ms granularity)
      Conv1d(32, 64, 16, s=4)   →  16 time steps   (~1 sec granularity)
      Conv1d(64,128,  8, s=2)   →   8 time steps   (~2 sec granularity)
      AdaptiveAvgPool → global pooling               →  (batch, 128)

    Kernel sizes are chosen to roughly match the dominant EEG rhythms:
      - 64-sample kernel at 16× stride ≈ captures 250 ms windows (theta/alpha)
      - 16-sample kernel at 4× stride  ≈ captures 62.5 ms segments (beta)
      -  8-sample kernel at 2× stride  ≈ 31 ms (fine temporal features)
    """

    def __init__(
        self,
        n_channels: int = _DEFAULT_N_CHANNELS,
        n_samples: int = _DEFAULT_N_SAMPLES,
        enc_dim: int = _DEFAULT_ENC_DIM,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.enc_dim = enc_dim

        self.block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=64, stride=16, padding=24, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=16, stride=4, padding=6, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, enc_dim, kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(enc_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode EEG to a fixed-size embedding.

        Args:
            x: (batch, n_channels, n_samples)

        Returns:
            (batch, enc_dim) L2-normalised embedding.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)          # (batch, enc_dim)
        return F.normalize(x, dim=-1)          # unit-sphere projection


# ══════════════════════════════════════════════════════════════════════════════
# Projection Head
# ══════════════════════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """Three-layer MLP projector used only during Barlow Twins training.

    Original paper: projector dimensionality matters more than depth.
    The projector is discarded after pre-training; only the encoder is kept.

    Input:  (batch, enc_dim=128)
    Output: (batch, proj_dim=128)
    """

    def __init__(
        self,
        enc_dim: int = _DEFAULT_ENC_DIM,
        hidden_dim: int = _DEFAULT_PROJ_HIDDEN,
        proj_dim: int = _DEFAULT_PROJ_DIM,
    ) -> None:
        super().__init__()
        self.enc_dim = enc_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim

        self.net = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            # No BN on the output layer — Barlow Twins operates on the
            # cross-correlation matrix, not on normalised embeddings.
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Barlow Twins loss
# ══════════════════════════════════════════════════════════════════════════════

def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    lambda_: float = _DEFAULT_LAMBDA,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the Barlow Twins cross-correlation loss.

    Zbontar et al. (2021), Equation 1:

      L = sum_i (1 - C_ii)^2  +  lambda * sum_{i!=j} C_ij^2

    where C is the cross-correlation matrix of batch-normalised projections:

      C_ij = sum_b z_A_bi * z_B_bj
             ─────────────────────────────────────────────────────
             sqrt(sum_b z_A_bi^2) * sqrt(sum_b z_B_bj^2)

    Args:
        z_a:     Projection of view A, shape (batch, proj_dim).
        z_b:     Projection of view B, shape (batch, proj_dim).
        lambda_: Weight for off-diagonal terms. Default 0.005 (paper default).

    Returns:
        loss: scalar tensor
        info: dict with {"loss_inv", "loss_red", "loss_total"} as Python floats
    """
    batch_size, proj_dim = z_a.shape

    # Batch-normalise along the batch dimension (not feature-wise BN)
    z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-6)
    z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-6)

    # Cross-correlation matrix: (proj_dim, proj_dim)
    C = (z_a_norm.T @ z_b_norm) / batch_size

    # Invariance loss: drive on-diagonal toward 1
    loss_inv = (torch.diagonal(C) - 1).pow(2).sum()

    # Redundancy-reduction loss: drive off-diagonal toward 0
    # Create mask that zeros the diagonal
    diag_mask = torch.eye(proj_dim, device=z_a.device, dtype=torch.bool)
    off_diag = C.masked_fill(diag_mask, 0.0)
    loss_red = off_diag.pow(2).sum()

    loss = loss_inv + lambda_ * loss_red

    return loss, {
        "loss_inv":   float(loss_inv.item()),
        "loss_red":   float(loss_red.item()),
        "loss_total": float(loss.item()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BarlowTwinsEEG — full self-supervised model
# ══════════════════════════════════════════════════════════════════════════════

class BarlowTwinsEEG(nn.Module):
    """Full Barlow Twins model for EEG self-supervised representation learning.

    Combines EEGEncoder + ProjectionHead with data augmentation.

    Workflow
    ────────
    Training:
      1. Load unlabeled EEG epochs (batch, 4, 1024).
      2. Apply augmentations twice → views A and B.
      3. Encode both → embeddings.
      4. Project both → projections.
      5. Compute Barlow Twins loss on projections.
      6. Backprop, update encoder + projector.

    Inference (after training):
      1. Load saved checkpoint.
      2. Use model.encoder to extract 128-dim embeddings.
      3. Feed embeddings to a lightweight linear classifier.

    Args:
        n_channels: number of EEG input channels (default 4 for Muse 2)
        n_samples:  EEG samples per epoch (default 1024 = 4 sec × 256 Hz)
        enc_dim:    encoder output dimension (default 128)
        proj_dim:   projection head output dimension (default 128)
        proj_hidden: projection head hidden dimension (default 256)
        lambda_:    off-diagonal loss weight (default 0.005)
        aug_params: keyword arguments passed to EEGAugmentations
    """

    def __init__(
        self,
        n_channels: int = _DEFAULT_N_CHANNELS,
        n_samples: int = _DEFAULT_N_SAMPLES,
        enc_dim: int = _DEFAULT_ENC_DIM,
        proj_dim: int = _DEFAULT_PROJ_DIM,
        proj_hidden: int = _DEFAULT_PROJ_HIDDEN,
        lambda_: float = _DEFAULT_LAMBDA,
        **aug_params,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.enc_dim = enc_dim
        self.proj_dim = proj_dim
        self.proj_hidden = proj_hidden
        self.lambda_ = lambda_

        self.encoder = EEGEncoder(n_channels, n_samples, enc_dim)
        self.projector = ProjectionHead(enc_dim, proj_hidden, proj_dim)
        self.augmentations = EEGAugmentations(**aug_params)

    # ── Forward / training ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode EEG to embedding (inference mode — no augmentation).

        Args:
            x: (batch, n_channels, n_samples)

        Returns:
            (batch, enc_dim) L2-normalised embedding from encoder.
        """
        return self.encoder(x)

    def training_step(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Generate two augmented views and compute Barlow Twins loss.

        Args:
            x: Unlabeled EEG batch (batch, n_channels, n_samples).

        Returns:
            (loss, info_dict) where loss is a differentiable scalar tensor
            and info_dict contains {"loss_inv", "loss_red", "loss_total"}.
        """
        view_a = self.augmentations(x)
        view_b = self.augmentations(x)

        emb_a = self.encoder(view_a)
        emb_b = self.encoder(view_b)

        z_a = self.projector(emb_a)
        z_b = self.projector(emb_b)

        return barlow_twins_loss(z_a, z_b, self.lambda_)

    # ── Embedding extraction (downstream use) ─────────────────────────────────

    def extract_embedding(self, eeg: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Extract encoder embedding from a raw EEG array.

        Args:
            eeg:    numpy array, shape (n_channels, n_samples) or (n_samples,)
            device: torch device string

        Returns:
            1D numpy array of shape (enc_dim,).
        """
        if eeg.ndim == 1:
            eeg = eeg[np.newaxis, :]          # (1, n_samples)

        # Pad / trim to model's expected n_samples
        n_ch, n_t = eeg.shape
        if n_t < self.n_samples:
            pad = self.n_samples - n_t
            eeg = np.pad(eeg, ((0, 0), (0, pad)), mode="edge")
        elif n_t > self.n_samples:
            eeg = eeg[:, :self.n_samples]

        # Normalise each channel to zero-mean unit-variance
        mu = eeg.mean(axis=-1, keepdims=True)
        sd = eeg.std(axis=-1, keepdims=True) + 1e-7
        eeg = (eeg - mu) / sd

        x = torch.from_numpy(eeg.astype(np.float32)).unsqueeze(0).to(device)

        self.eval()
        with torch.no_grad():
            emb = self.encoder(x)

        return emb.squeeze(0).cpu().numpy()

    def extract_embeddings_batch(
        self, eeg_batch: np.ndarray, batch_size: int = 64, device: str = "cpu"
    ) -> np.ndarray:
        """Batch embedding extraction for datasets.

        Args:
            eeg_batch: (N, n_channels, n_samples)
            batch_size: mini-batch size for GPU memory management
            device:    torch device string

        Returns:
            (N, enc_dim) numpy array.
        """
        self.eval()
        all_embs = []
        n = len(eeg_batch)
        for start in range(0, n, batch_size):
            chunk = eeg_batch[start:start + batch_size].astype(np.float32)
            # Per-epoch per-channel normalisation
            mu = chunk.mean(axis=-1, keepdims=True)
            sd = chunk.std(axis=-1, keepdims=True) + 1e-7
            chunk = (chunk - mu) / sd
            x = torch.from_numpy(chunk).to(device)
            with torch.no_grad():
                emb = self.encoder(x)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    # ── Serialisation ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save full model checkpoint (encoder + projector + config)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "n_channels": self.n_channels,
                    "n_samples":  self.n_samples,
                    "enc_dim":    self.enc_dim,
                    "proj_dim":   self.proj_dim,
                    "proj_hidden": self.proj_hidden,
                    "lambda_":    self.lambda_,
                },
            },
            path,
        )
        log.info("BarlowTwinsEEG saved → %s", path)

    def save_encoder(self, path: str | Path) -> None:
        """Save only the encoder weights (for downstream transfer)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_state": self.encoder.state_dict(),
                "enc_dim":       self.enc_dim,
                "n_channels":    self.n_channels,
                "n_samples":     self.n_samples,
            },
            path,
        )
        log.info("BarlowTwinsEEG encoder saved → %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "BarlowTwinsEEG":
        """Load a full Barlow Twins checkpoint.

        Args:
            path:   path to .pt file saved with BarlowTwinsEEG.save()
            device: torch device string

        Returns:
            Loaded BarlowTwinsEEG in eval mode.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model.to(device)
        log.info("BarlowTwinsEEG loaded ← %s", path)
        return model

    @classmethod
    def load_encoder_only(cls, path: str | Path, device: str = "cpu") -> EEGEncoder:
        """Load only the EEGEncoder from a checkpoint.

        Supports both full checkpoints (saved with save()) and encoder-only
        checkpoints (saved with save_encoder()).

        Args:
            path:   path to .pt file
            device: torch device string

        Returns:
            EEGEncoder in eval mode, ready for downstream classification.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if "encoder_state" in ckpt:
            # Encoder-only checkpoint
            encoder = EEGEncoder(
                n_channels=ckpt["n_channels"],
                n_samples=ckpt["n_samples"],
                enc_dim=ckpt["enc_dim"],
            )
            encoder.load_state_dict(ckpt["encoder_state"])
        else:
            # Full checkpoint — extract encoder weights
            cfg = ckpt["config"]
            encoder = EEGEncoder(
                n_channels=cfg["n_channels"],
                n_samples=cfg["n_samples"],
                enc_dim=cfg["enc_dim"],
            )
            enc_state = {
                k.replace("encoder.", "", 1): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("encoder.")
            }
            encoder.load_state_dict(enc_state)

        encoder.eval()
        encoder.to(device)
        log.info("BarlowTwinsEEG encoder-only loaded ← %s", path)
        return encoder

    # ── Parameter counts ──────────────────────────────────────────────────────

    def param_count(self) -> Dict[str, int]:
        """Return parameter counts for encoder, projector, and total."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        proj_params = sum(p.numel() for p in self.projector.parameters())
        return {
            "encoder":   enc_params,
            "projector": proj_params,
            "total":     enc_params + proj_params,
        }
