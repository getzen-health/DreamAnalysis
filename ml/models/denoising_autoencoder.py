"""EEG Denoising Autoencoder — learned neural denoiser for robust EEG processing.

A 1D convolutional autoencoder that learns to remove artifacts from EEG signals.
Trained on paired clean/noisy data (EEGdenoiseNet or synthetic pairs).

Architecture:
    Encoder: Conv1D layers with stride-2 downsampling (compresses signal)
    Bottleneck: Compact latent representation of clean signal features
    Decoder: ConvTranspose1D layers with stride-2 upsampling (reconstructs clean signal)

Usage:
    denoiser = EEGDenoiser(fs=256)
    denoiser.load("models/saved/denoiser_model.pt")
    clean_signal = denoiser.denoise(noisy_signal)
"""

import numpy as np
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Conv1DAutoencoder(nn.Module):
    """1D Convolutional Autoencoder for EEG denoising.

    Encoder compresses temporal signal, decoder reconstructs clean version.
    Skip connections (U-Net style) preserve fine temporal detail.
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 32):
        super().__init__()

        # Encoder: progressively compress
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.LeakyReLU(0.1),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_filters * 2),
            nn.LeakyReLU(0.1),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_filters * 4),
            nn.LeakyReLU(0.1),
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(base_filters * 4, base_filters * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 8),
            nn.LeakyReLU(0.1),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters * 8),
            nn.LeakyReLU(0.1),
        )

        # Decoder: progressively upsample (with skip connections)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(base_filters * 8, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.LeakyReLU(0.1),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(base_filters * 4 + base_filters * 4, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 2),
            nn.LeakyReLU(0.1),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(base_filters * 2 + base_filters * 2, base_filters, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm1d(base_filters),
            nn.LeakyReLU(0.1),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(base_filters + base_filters, in_channels, kernel_size=6, stride=2, padding=2),
        )

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = self._match_and_cat(d4, e3)
        d3 = self.dec3(d4)
        d3 = self._match_and_cat(d3, e2)
        d2 = self.dec2(d3)
        d2 = self._match_and_cat(d2, e1)
        d1 = self.dec1(d2)

        # Match output length to input
        if d1.shape[-1] != x.shape[-1]:
            d1 = nn.functional.interpolate(d1, size=x.shape[-1], mode="linear", align_corners=False)

        return d1

    @staticmethod
    def _match_and_cat(x, skip):
        """Match temporal dimensions and concatenate skip connection."""
        if x.shape[-1] != skip.shape[-1]:
            min_len = min(x.shape[-1], skip.shape[-1])
            x = x[..., :min_len]
            skip = skip[..., :min_len]
        return torch.cat([x, skip], dim=1)


class EEGDenoiser:
    """High-level EEG denoising interface.

    Wraps the Conv1D autoencoder with preprocessing, normalization, and
    fallback to classical filtering when model is not available.
    """

    def __init__(self, fs: float = 256.0, model_path: Optional[str] = None):
        self.fs = fs
        self.model = None
        self.device = "cpu"
        self.scaler_mean = 0.0
        self.scaler_std = 1.0

        if TORCH_AVAILABLE:
            self.model = Conv1DAutoencoder(in_channels=1, base_filters=32)
            if model_path and Path(model_path).exists():
                self.load(model_path)

    def load(self, model_path: str):
        """Load a trained denoiser model."""
        if not TORCH_AVAILABLE:
            return

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint.get("model_state", checkpoint))
            self.scaler_mean = checkpoint.get("scaler_mean", 0.0)
            self.scaler_std = checkpoint.get("scaler_std", 1.0)
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def save(self, model_path: str):
        """Save the denoiser model."""
        if not TORCH_AVAILABLE or self.model is None:
            return

        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
        }, model_path)

    def denoise(self, signal: np.ndarray) -> np.ndarray:
        """Denoise a single-channel EEG signal.

        Args:
            signal: 1D numpy array of raw EEG data.

        Returns:
            Denoised signal as 1D numpy array.
        """
        if not TORCH_AVAILABLE or self.model is None:
            return self._fallback_denoise(signal)

        self.model.eval()

        # Normalize
        sig_mean = np.mean(signal)
        sig_std = np.std(signal) + 1e-8
        normalized = (signal - sig_mean) / sig_std

        # Pad to multiple of 16 for conv/deconv alignment
        orig_len = len(normalized)
        pad_len = (16 - orig_len % 16) % 16
        if pad_len > 0:
            normalized = np.pad(normalized, (0, pad_len), mode="reflect")

        # Convert to tensor: (1, 1, n_samples)
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            cleaned = self.model(x)

        # Convert back
        result = cleaned.squeeze().numpy()[:orig_len]

        # Denormalize
        result = result * sig_std + sig_mean

        return result

    def denoise_multichannel(self, signals: np.ndarray) -> np.ndarray:
        """Denoise multi-channel EEG data.

        Args:
            signals: 2D array (n_channels, n_samples).

        Returns:
            Denoised signals (n_channels, n_samples).
        """
        result = np.zeros_like(signals)
        for ch in range(signals.shape[0]):
            result[ch] = self.denoise(signals[ch])
        return result

    def train_on_pairs(
        self,
        noisy: np.ndarray,
        clean: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> dict:
        """Train the denoiser on paired noisy/clean data.

        Args:
            noisy: Array of noisy signals (n_samples, signal_length).
            clean: Array of corresponding clean signals.
            epochs: Number of training epochs.
            batch_size: Batch size.
            lr: Learning rate.

        Returns:
            Training stats dict.
        """
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        # Normalize globally
        self.scaler_mean = float(np.mean(clean))
        self.scaler_std = float(np.std(clean)) + 1e-8
        noisy_norm = (noisy - self.scaler_mean) / self.scaler_std
        clean_norm = (clean - self.scaler_mean) / self.scaler_std

        # Pad signals to multiple of 16
        sig_len = noisy_norm.shape[-1]
        pad_len = (16 - sig_len % 16) % 16
        if pad_len > 0:
            noisy_norm = np.pad(noisy_norm, ((0, 0), (0, pad_len)), mode="reflect")
            clean_norm = np.pad(clean_norm, ((0, 0), (0, pad_len)), mode="reflect")

        # Convert to tensors
        X = torch.tensor(noisy_norm, dtype=torch.float32).unsqueeze(1)
        Y = torch.tensor(clean_norm, dtype=torch.float32).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Combined loss: L1 + spectral loss
        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()

        self.model.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for noisy_batch, clean_batch in loader:
                optimizer.zero_grad()

                pred = self.model(noisy_batch)

                # L1 reconstruction loss (robust to outliers)
                loss_l1 = l1_loss(pred, clean_batch)

                # Spectral loss (preserve frequency content)
                pred_fft = torch.fft.rfft(pred.squeeze(1))
                clean_fft = torch.fft.rfft(clean_batch.squeeze(1))
                loss_spectral = mse_loss(
                    torch.abs(pred_fft), torch.abs(clean_fft)
                ) * 0.1

                loss = loss_l1 + loss_spectral
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}")

        self.model.eval()

        return {
            "final_loss": losses[-1] if losses else 0,
            "loss_history": losses,
            "n_samples": len(noisy),
            "epochs": epochs,
        }

    def compute_snr_improvement(
        self, noisy: np.ndarray, clean: np.ndarray
    ) -> dict:
        """Compute SNR improvement after denoising.

        Args:
            noisy: Noisy signal(s).
            clean: Ground truth clean signal(s).

        Returns:
            Dict with SNR before, after, and improvement in dB.
        """
        denoised = self.denoise(noisy) if noisy.ndim == 1 else np.array([
            self.denoise(noisy[i]) for i in range(len(noisy))
        ])

        def _snr_db(signal, noise):
            sig_power = np.mean(signal ** 2) + 1e-10
            noise_power = np.mean(noise ** 2) + 1e-10
            return 10 * np.log10(sig_power / noise_power)

        if noisy.ndim == 1:
            snr_before = _snr_db(clean, noisy - clean)
            snr_after = _snr_db(clean, denoised - clean)
        else:
            snr_before = np.mean([_snr_db(clean[i], noisy[i] - clean[i]) for i in range(len(clean))])
            snr_after = np.mean([_snr_db(clean[i], denoised[i] - clean[i]) for i in range(len(clean))])

        return {
            "snr_before_db": float(snr_before),
            "snr_after_db": float(snr_after),
            "improvement_db": float(snr_after - snr_before),
        }

    def _fallback_denoise(self, signal: np.ndarray) -> np.ndarray:
        """Classical bandpass + notch filter fallback when no model available."""
        from scipy import signal as scipy_signal

        # Bandpass 0.5-50 Hz
        nyq = self.fs / 2
        b, a = scipy_signal.butter(4, [0.5 / nyq, min(50.0 / nyq, 0.99)], btype="band")
        filtered = scipy_signal.filtfilt(b, a, signal)

        # Notch at 50 and 60 Hz
        for notch_freq in [50.0, 60.0]:
            if notch_freq < nyq:
                b_notch, a_notch = scipy_signal.iirnotch(notch_freq, Q=30.0, fs=self.fs)
                filtered = scipy_signal.filtfilt(b_notch, a_notch, filtered)

        return filtered
