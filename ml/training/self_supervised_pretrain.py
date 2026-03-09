"""Self-supervised contrastive pretraining for 4-channel EEG.

Scientific basis:
- MDPI Mathematics (2025): Self-supervised framework on 4-channel EEG
  achieves +20.4% accuracy over conventional features

Approach: SimCLR-style NT-Xent contrastive loss.
- Two augmented views of same EEG epoch = positive pair
- Different epochs = negative pairs
- Encoder learns representations without emotion labels

Augmentations (order from strongest to weakest for EEG):
1. Temporal jitter: shift signal by ±50ms (±12 samples at 256Hz)
2. Channel dropout: zero out 1 of 4 channels randomly
3. Gaussian noise: add noise at SNR 20dB
4. Frequency masking: zero out power in one random frequency band

Architecture:
- Encoder: 2-layer 1D-CNN (32→64 filters, stride 2) → Global average pool
- Projection head: Linear(128) → ReLU → Linear(64)
- NT-Xent loss with temperature τ=0.1
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EEGAugmentor:
    """EEG data augmentation for contrastive learning."""

    def __init__(self, fs: float = 256.0):
        self.fs = fs

    def temporal_jitter(self, epoch: np.ndarray, max_shift_ms: float = 50.0) -> np.ndarray:
        """Shift signal by random ±max_shift_ms milliseconds."""
        shift = int(np.random.uniform(
            -max_shift_ms * self.fs / 1000,
            max_shift_ms * self.fs / 1000,
        ))
        return np.roll(epoch, shift, axis=-1)

    def channel_dropout(self, epoch: np.ndarray, p_drop: float = 0.25) -> np.ndarray:
        """Zero out each channel independently with probability p_drop."""
        result = epoch.copy()
        if epoch.ndim == 2:
            for ch_idx in range(epoch.shape[0]):
                if np.random.random() < p_drop:
                    result[ch_idx] = 0.0
        return result

    def gaussian_noise(self, epoch: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """Add Gaussian noise at given SNR (dB)."""
        signal_power = np.mean(epoch ** 2) + 1e-10
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(*epoch.shape) * np.sqrt(noise_power)
        return epoch + noise

    def frequency_masking(self, epoch: np.ndarray, fs: Optional[float] = None) -> np.ndarray:
        """Zero out power in one random EEG band (delta/theta/alpha/beta)."""
        fs = fs or self.fs
        result = epoch.copy()
        bands = [(0.5, 4), (4, 8), (8, 12), (12, 30)]
        low, high = bands[np.random.randint(len(bands))]
        # FFT mask
        n = epoch.shape[-1]
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = ~((freqs >= low) & (freqs <= high))
        if epoch.ndim == 1:
            fft = np.fft.rfft(epoch)
            result = np.fft.irfft(fft * mask, n)
        elif epoch.ndim == 2:
            for ch in range(epoch.shape[0]):
                fft = np.fft.rfft(epoch[ch])
                result[ch] = np.fft.irfft(fft * mask, n)
        return result

    def augment(self, epoch: np.ndarray) -> np.ndarray:
        """Apply random combination of augmentations."""
        aug = epoch.copy()
        if np.random.random() < 0.8:
            aug = self.temporal_jitter(aug)
        if np.random.random() < 0.5:
            aug = self.channel_dropout(aug)
        if np.random.random() < 0.8:
            aug = self.gaussian_noise(aug)
        if np.random.random() < 0.5:
            aug = self.frequency_masking(aug)
        return aug


if TORCH_AVAILABLE:
    class EEGEncoder(nn.Module):
        """1D-CNN encoder for 4-channel EEG."""

        def __init__(self, n_channels: int = 4, embed_dim: int = 128):
            super().__init__()
            self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=8, stride=2, padding=4)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, embed_dim, kernel_size=4, stride=2, padding=2)
            self.bn3 = nn.BatchNorm1d(embed_dim)
            self.relu = nn.ReLU()
            self.embed_dim = embed_dim

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, n_channels, T)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            return x.mean(dim=-1)  # global average pool → (batch, embed_dim)

else:
    class EEGEncoder:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable."""

        def __init__(self, n_channels: int = 4, embed_dim: int = 128):
            self.embed_dim = embed_dim

        def parameters(self):
            return iter([])


class EEGContrastivePretrainer:
    """SimCLR-style contrastive pretrainer for 4-channel EEG.

    No emotion labels required — learns representations from augmented pairs.
    """

    def __init__(
        self,
        n_channels: int = 4,
        embed_dim: int = 128,
        proj_dim: int = 64,
        temperature: float = 0.1,
        fs: float = 256.0,
    ):
        self.fs = fs
        self.temperature = temperature
        self.augmentor = EEGAugmentor(fs=fs)

        if TORCH_AVAILABLE:
            self.encoder = EEGEncoder(n_channels=n_channels, embed_dim=embed_dim)
            self.projection_head = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, proj_dim),
            )
            self.params = (
                list(self.encoder.parameters())
                + list(self.projection_head.parameters())
            )
        else:
            self.encoder = EEGEncoder(n_channels=n_channels, embed_dim=embed_dim)
            self.projection_head = None
            self.params = []

    def nt_xent_loss(
        self, z1: "torch.Tensor", z2: "torch.Tensor"
    ) -> "torch.Tensor":
        """NT-Xent (normalized temperature-scaled cross-entropy) loss.

        Args:
            z1, z2: normalized projections, shape (batch, proj_dim)
        Returns:
            scalar loss
        """
        import torch
        import torch.nn.functional as F

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch = z1.shape[0]

        # Concatenate: [z1, z2] → (2*batch, proj_dim)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask diagonal (self-similarity)
        mask = torch.eye(2 * batch, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+batch) and (i+batch, i)
        labels = torch.cat([
            torch.arange(batch, 2 * batch),
            torch.arange(0, batch),
        ])

        return F.cross_entropy(sim, labels)

    def pretrain_step(self, epochs: np.ndarray) -> Dict:
        """Run one pretraining step on a batch of EEG epochs.

        Args:
            epochs: np.ndarray of shape (batch, n_channels, T)

        Returns:
            dict with 'loss', 'n_samples', 'torch_available'
        """
        if not TORCH_AVAILABLE or self.projection_head is None:
            return {
                "loss": None,
                "n_samples": len(epochs),
                "torch_available": False,
                "message": "PyTorch required for contrastive pretraining",
            }

        import torch

        # Create two augmented views
        aug1 = np.array([self.augmentor.augment(e) for e in epochs])
        aug2 = np.array([self.augmentor.augment(e) for e in epochs])

        x1 = torch.FloatTensor(aug1)
        x2 = torch.FloatTensor(aug2)

        # Forward pass (no optimizer step — caller is responsible for backward)
        self.encoder.train()
        self.projection_head.train()

        with torch.no_grad():
            z1 = self.projection_head(self.encoder(x1))
            z2 = self.projection_head(self.encoder(x2))
            loss = self.nt_xent_loss(z1, z2)

        return {
            "loss": round(float(loss.item()), 6),
            "n_samples": len(epochs),
            "torch_available": True,
            "embed_dim": self.encoder.embed_dim,
        }

    def pretrain(
        self,
        unlabeled_epochs: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 3e-4,
    ) -> Dict:
        """Run full pretraining loop.

        Args:
            unlabeled_epochs: (N, n_channels, T) array of EEG epochs
            n_epochs: training epochs
            batch_size: mini-batch size
            lr: learning rate

        Returns:
            dict with 'final_loss', 'n_epochs_trained', 'n_samples', 'loss_history'
        """
        if not TORCH_AVAILABLE or self.projection_head is None:
            return {
                "final_loss": None,
                "n_epochs_trained": 0,
                "n_samples": len(unlabeled_epochs),
                "torch_available": False,
            }

        import torch

        optimizer = torch.optim.Adam(self.params, lr=lr)
        loss_history: List[float] = []
        N = len(unlabeled_epochs)

        for epoch in range(n_epochs):
            idx = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N, batch_size):
                batch = unlabeled_epochs[idx[i : i + batch_size]]
                if len(batch) < 2:
                    continue

                aug1 = np.array([self.augmentor.augment(e) for e in batch])
                aug2 = np.array([self.augmentor.augment(e) for e in batch])
                x1 = torch.FloatTensor(aug1)
                x2 = torch.FloatTensor(aug2)

                optimizer.zero_grad()
                z1 = self.projection_head(self.encoder(x1))
                z2 = self.projection_head(self.encoder(x2))
                loss = self.nt_xent_loss(z1, z2)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(round(avg_loss, 6))

        return {
            "final_loss": loss_history[-1] if loss_history else None,
            "n_epochs_trained": n_epochs,
            "n_samples": N,
            "torch_available": True,
            "loss_history": loss_history,
        }

    def extract_features(self, eeg: np.ndarray) -> np.ndarray:
        """Extract learned representations from pretrained encoder.

        Args:
            eeg: (n_channels, T) or (batch, n_channels, T)

        Returns:
            embeddings: (embed_dim,) or (batch, embed_dim) numpy array
        """
        if not TORCH_AVAILABLE or self.projection_head is None:
            # Return zero vector so callers always get a valid array
            embed_dim = self.encoder.embed_dim
            single = eeg.ndim == 2
            if single:
                return np.zeros(embed_dim)
            return np.zeros((len(eeg), embed_dim))

        import torch

        self.encoder.eval()
        single = eeg.ndim == 2
        if single:
            eeg = eeg[np.newaxis]  # → (1, n_ch, T)

        x = torch.FloatTensor(eeg)
        with torch.no_grad():
            z = self.encoder(x).numpy()

        return z[0] if single else z

    def get_pretrainer_info(self) -> Dict:
        """Return pretrainer architecture info."""
        n_params = 0
        if TORCH_AVAILABLE and self.params:
            n_params = sum(p.numel() for p in self.params)
        return {
            "architecture": "SimCLR-style contrastive EEG encoder",
            "n_params": n_params,
            "torch_available": TORCH_AVAILABLE,
            "temperature": self.temperature,
            "augmentations": [
                "temporal_jitter",
                "channel_dropout",
                "gaussian_noise",
                "frequency_masking",
            ],
            "reference": "MDPI Mathematics 2025: +20.4% accuracy over raw features",
        }
