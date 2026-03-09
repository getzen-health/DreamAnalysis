"""CNN-KAN-F2CA sparse-channel emotion classifier for 4-channel EEG.

Inspired by CNN-KAN-F2CA (89% DEAP 4-class, 4 channels).
Architecture: 1D-CNN temporal extraction → KAN-inspired polynomial feature
expansion → emotion classification.

KAN layer uses Chebyshev polynomial basis (degree 4) as learnable activation
functions, which replaces standard linear layers for better generalization on
small datasets.

Reference: CNN-KAN-F2CA paper achieving 89% DEAP 4-class accuracy with only
4 EEG channels via learnable Kolmogorov-Arnold Network activation functions.

The four emotion classes correspond to Russell's 2D valence-arousal space
(DEAP-style quadrant labeling):
  - low_valence_low_arousal   — sad, depressed, bored
  - high_valence_low_arousal  — calm, content, serene
  - low_valence_high_arousal  — stressed, angry, fearful
  - high_valence_high_arousal — happy, excited, elated
"""

import logging
import threading
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Emotion class definitions ─────────────────────────────────────────────────
EMOTION_CLASSES = [
    "low_valence_low_arousal",    # LVLA: sad/depressed
    "high_valence_low_arousal",   # HVLA: calm/content
    "low_valence_high_arousal",   # LVHA: stressed/angry
    "high_valence_high_arousal",  # HVHA: happy/excited
]

# Valence/arousal centroids for each quadrant (used in feature-based fallback)
_QUADRANT_CENTROIDS = {
    "low_valence_low_arousal":   (-0.6, -0.6),
    "high_valence_low_arousal":  ( 0.6, -0.4),
    "low_valence_high_arousal":  (-0.5,  0.6),
    "high_valence_high_arousal": ( 0.6,  0.6),
}

# ── PyTorch availability check ────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.info("PyTorch not available — CNN-KAN will use feature-based fallback")


# ── Chebyshev KAN layer ───────────────────────────────────────────────────────

def _poly_basis(x: "torch.Tensor", degree: int = 4) -> "torch.Tensor":
    """Chebyshev polynomial basis T_0..T_degree evaluated at x.

    Uses the recurrence relation: T_n(x) = 2x·T_{n-1}(x) - T_{n-2}(x)
    which is numerically stable and equivalent to cos(n·arccos(x)) for x∈[-1,1].

    Args:
        x: Input tensor of shape (...,). Values should be in [-1, 1].
        degree: Maximum Chebyshev degree (inclusive). Default 4.

    Returns:
        Tensor of shape (..., degree+1) — one column per T_n.
    """
    # Clamp to valid domain for arccos-based interpretation
    x = torch.clamp(x, -1.0, 1.0)
    polys = [torch.ones_like(x), x]
    for n in range(2, degree + 1):
        t_n = 2.0 * x * polys[-1] - polys[-2]
        polys.append(t_n)
    return torch.stack(polys, dim=-1)  # (..., degree+1)


if _TORCH_AVAILABLE:
    class _KANLinear(nn.Module):
        """KAN-inspired linear layer using Chebyshev polynomial basis functions.

        Instead of computing output = W·x + b (standard linear), this layer:
        1. Normalises input features to [-1, 1] via tanh.
        2. Expands each input feature into (degree+1) Chebyshev basis values.
        3. Computes output as a learned weighted sum over basis terms.

        This gives each input-output connection a learnable nonlinear activation
        (a degree-4 polynomial approximated via Chebyshev coefficients), which
        is the core idea of KAN (Kolmogorov-Arnold Networks).

        Parameter count: in_features × out_features × (degree+1) + out_features bias
        """

        def __init__(self, in_features: int, out_features: int, degree: int = 4):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.degree = degree
            self.n_basis = degree + 1

            # Learnable Chebyshev coefficients: shape (out, in, n_basis)
            self.coefficients = nn.Parameter(
                torch.empty(out_features, in_features, self.n_basis)
            )
            self.bias = nn.Parameter(torch.zeros(out_features))

            # Initialize with small random values for stable gradient flow
            nn.init.kaiming_uniform_(
                self.coefficients.view(out_features, -1),
                a=0.01,
                mode="fan_in",
                nonlinearity="linear",
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (batch, in_features)

            Returns:
                (batch, out_features)
            """
            # Normalise to [-1, 1] so Chebyshev basis is well-defined
            x_norm = torch.tanh(x)  # (batch, in)

            # Evaluate basis at each input feature: (batch, in, n_basis)
            basis = _poly_basis(x_norm, self.degree)

            # Weighted sum over input features and basis terms
            # coefficients: (out, in, n_basis)
            # basis:        (batch, in, n_basis)
            # → einsum(b_i_k, o_i_k → b_o)
            out = torch.einsum("bik,oik->bo", basis, self.coefficients) + self.bias
            return out

        def extra_repr(self) -> str:
            return (
                f"in={self.in_features}, out={self.out_features}, "
                f"degree={self.degree}, params="
                f"{self.in_features * self.out_features * self.n_basis + self.out_features}"
            )

    class _CNNKANModel(nn.Module):
        """Full CNN + KAN model for 4-channel EEG 4-class emotion classification.

        Architecture:
            Input: (batch, n_channels, n_samples)
            Conv1d(n_channels, 32, k=16) → BN → ReLU
            Conv1d(32, 64, k=8)          → BN → ReLU
            AdaptiveAvgPool1d(16)        → flatten → (batch, 1024)
            _KANLinear(1024, 128)        → ReLU → Dropout(0.3)
            _KANLinear(128, n_classes)
        """

        def __init__(self, n_channels: int = 4, n_classes: int = 4):
            super().__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes

            # 1D-CNN temporal feature extraction
            self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=16, padding=8)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=8, padding=4)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool = nn.AdaptiveAvgPool1d(16)

            # KAN-inspired feature combination layers
            cnn_out_dim = 64 * 16  # 1024
            self.kan1 = _KANLinear(cnn_out_dim, 128, degree=4)
            self.dropout = nn.Dropout(p=0.3)
            self.kan2 = _KANLinear(128, n_classes, degree=4)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (batch, n_channels, n_samples)

            Returns:
                logits: (batch, n_classes)
            """
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)               # (batch, 64, 16)
            x = x.flatten(start_dim=1)    # (batch, 1024)
            x = torch.relu(self.kan1(x))
            x = self.dropout(x)
            x = self.kan2(x)              # (batch, n_classes)
            return x

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Processing helpers (scipy preferred, numpy fallback) ─────────────────────

def _extract_band_power(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """Estimate power in a frequency band via Welch PSD."""
    try:
        from scipy.signal import welch
        nperseg = min(len(signal), int(fs * 2))
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        mask = (freqs >= fmin) & (freqs <= fmax)
        return float(np.mean(psd[mask])) if mask.any() else 1e-10
    except Exception:
        # Fallback: rough power via FFT
        n = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal)) ** 2 / n
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= fmin) & (freqs <= fmax)
        return float(np.mean(fft_vals[mask])) if mask.any() else 1e-10


def _compute_faa(signals: np.ndarray, fs: float) -> float:
    """Frontal Alpha Asymmetry: ln(AF8_alpha) - ln(AF7_alpha).

    Expects signals shape (4, n_samples) with BrainFlow Muse 2 order:
    ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10.
    """
    if signals.ndim < 2 or signals.shape[0] < 3:
        return 0.0
    af7_alpha = _extract_band_power(signals[1], fs, 8.0, 13.0)
    af8_alpha = _extract_band_power(signals[2], fs, 8.0, 13.0)
    af7_alpha = max(af7_alpha, 1e-12)
    af8_alpha = max(af8_alpha, 1e-12)
    return float(np.log(af8_alpha) - np.log(af7_alpha))


def _feature_based_predict(signals: np.ndarray, fs: float) -> Dict[str, Any]:
    """Threshold-based 4-quadrant emotion classification from band powers.

    Used when PyTorch is unavailable or signals are too short for CNN.
    Computes valence from FAA + alpha/beta ratio, arousal from beta ratio.
    """
    # Use first channel (AF7) as primary signal; use all channels if available
    primary = signals[0] if signals.ndim == 2 else signals
    n_channels = signals.shape[0] if signals.ndim == 2 else 1

    # Band powers from primary channel
    delta = _extract_band_power(primary, fs, 0.5, 4.0)
    theta = _extract_band_power(primary, fs, 4.0, 8.0)
    alpha = _extract_band_power(primary, fs, 8.0, 13.0)
    beta  = _extract_band_power(primary, fs, 13.0, 30.0)
    high_beta = _extract_band_power(primary, fs, 20.0, 30.0)

    # Protect against zero
    denom = alpha + beta
    denom = denom if denom > 1e-12 else 1e-12

    # Arousal: beta/(alpha+beta) — higher beta → more aroused
    arousal_raw = float(beta / denom)
    arousal = float(np.clip(2.0 * arousal_raw - 1.0, -1.0, 1.0))

    # Valence: alpha/beta ratio term (band-based)
    ab_ratio = float(alpha / max(beta, 1e-12))
    valence_ab = float(np.tanh((ab_ratio - 0.7) * 2.0))

    # FAA component (if multichannel)
    if n_channels >= 3:
        faa = _compute_faa(signals, fs)
        faa_valence = float(np.clip(np.tanh(faa * 2.0), -1.0, 1.0))
        valence = float(np.clip(0.5 * valence_ab + 0.5 * faa_valence, -1.0, 1.0))
    else:
        valence = float(np.clip(valence_ab, -1.0, 1.0))

    # Map (valence, arousal) → quadrant
    if valence >= 0 and arousal >= 0:
        emotion = "high_valence_high_arousal"
    elif valence >= 0 and arousal < 0:
        emotion = "high_valence_low_arousal"
    elif valence < 0 and arousal >= 0:
        emotion = "low_valence_high_arousal"
    else:
        emotion = "low_valence_low_arousal"

    # Soft probabilities based on distance to quadrant centroids
    probs = _soft_probs_from_valence_arousal(valence, arousal)

    return {
        "emotion": emotion,
        "probabilities": probs,
        "valence": round(float(valence), 4),
        "arousal": round(float(arousal), 4),
        "model_type": "feature-based",
        "band_powers": {
            "delta": round(float(delta), 6),
            "theta": round(float(theta), 6),
            "alpha": round(float(alpha), 6),
            "beta":  round(float(beta), 6),
            "high_beta": round(float(high_beta), 6),
        },
    }


def _soft_probs_from_valence_arousal(valence: float, arousal: float) -> Dict[str, float]:
    """Convert (valence, arousal) to soft probability distribution over 4 classes.

    Uses inverse-distance weighting against quadrant centroids.
    """
    scores = {}
    for cls, (v_c, a_c) in _QUADRANT_CENTROIDS.items():
        dist = np.sqrt((valence - v_c) ** 2 + (arousal - a_c) ** 2)
        scores[cls] = 1.0 / (dist + 1e-6)

    total = sum(scores.values())
    return {cls: round(float(s / total), 4) for cls, s in scores.items()}


# ── Main classifier class ─────────────────────────────────────────────────────

class CNNKANEmotionClassifier:
    """CNN-KAN-F2CA sparse-channel emotion classifier for 4-channel EEG.

    Architecture: 1D-CNN temporal feature extraction followed by KAN-inspired
    Chebyshev polynomial feature combination layers. Designed for 4-channel
    consumer EEG (Muse 2 / OpenBCI Cyton in sparse mode).

    Args:
        n_channels: Number of EEG input channels (default 4 for Muse 2).
        n_classes:  Number of output emotion classes (default 4 for DEAP quadrants).
        fs:         Sampling frequency in Hz (default 256.0).
    """

    def __init__(self, n_channels: int = 4, n_classes: int = 4, fs: float = 256.0):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.fs = fs
        self._model: Optional[Any] = None

        if _TORCH_AVAILABLE:
            try:
                self._model = _CNNKANModel(n_channels=n_channels, n_classes=n_classes)
                self._model.eval()
                logger.info(
                    "CNN-KAN-F2CA initialised — %d params",
                    self._model.count_parameters(),
                )
            except Exception as exc:
                logger.warning("CNN-KAN model init failed: %s — using feature-based", exc)
                self._model = None
        else:
            logger.info("CNN-KAN-F2CA: PyTorch unavailable — feature-based mode")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict[str, Any]:
        """Classify 4-quadrant emotion from raw EEG.

        Args:
            eeg: EEG array. Shape (4, n_samples) for multi-channel or
                 (n_samples,) for single channel.
            fs:  Sampling frequency in Hz.

        Returns:
            dict with keys:
                emotion      — one of the 4 EMOTION_CLASSES strings
                probabilities — dict mapping each class to 0..1 probability
                valence      — continuous valence estimate in [-1, 1]
                arousal      — continuous arousal estimate in [-1, 1]
                model_type   — "cnn-kan" or "feature-based"
        """
        # ── Input normalisation ───────────────────────────────────────────────
        eeg = np.asarray(eeg, dtype=np.float32)
        if eeg.ndim == 1:
            signals = eeg.reshape(1, -1)  # (1, n_samples)
        elif eeg.ndim == 2:
            signals = eeg
        else:
            signals = eeg.reshape(self.n_channels, -1)

        # Need at least 32 samples for meaningful CNN processing
        if signals.shape[-1] < 32:
            return self._fallback_short(signals, fs)

        # ── Route to CNN or fallback ──────────────────────────────────────────
        if self._model is not None and _TORCH_AVAILABLE:
            return self._predict_cnn(signals, fs)
        return _feature_based_predict(signals, fs)

    def get_model_info(self) -> Dict[str, Any]:
        """Return architecture summary and parameter count."""
        if self._model is not None and _TORCH_AVAILABLE:
            n_params = self._model.count_parameters()
            kan_params = (
                self._model.kan1.in_features
                * self._model.kan1.out_features
                * self._model.kan1.n_basis
                + self._model.kan1.out_features
                + self._model.kan2.in_features
                * self._model.kan2.out_features
                * self._model.kan2.n_basis
                + self._model.kan2.out_features
            )
            return {
                "architecture": "CNN-KAN-F2CA",
                "backend": "pytorch",
                "n_channels": self.n_channels,
                "n_classes": self.n_classes,
                "fs": self.fs,
                "total_parameters": n_params,
                "cnn_parameters": n_params - kan_params,
                "kan_parameters": kan_params,
                "chebyshev_degree": 4,
                "emotion_classes": EMOTION_CLASSES,
                "description": (
                    "1D-CNN temporal extraction + KAN-inspired Chebyshev "
                    "polynomial feature combination. Optimised for 4-channel "
                    "sparse EEG. Inspired by CNN-KAN-F2CA (89% DEAP 4-class)."
                ),
                "layers": [
                    "Conv1d(n_channels→32, k=16) + BN + ReLU",
                    "Conv1d(32→64, k=8) + BN + ReLU",
                    "AdaptiveAvgPool1d(16) → flatten(1024)",
                    "_KANLinear(1024→128, degree=4) + ReLU + Dropout(0.3)",
                    "_KANLinear(128→4, degree=4)",
                ],
            }
        return {
            "architecture": "CNN-KAN-F2CA (feature-based fallback)",
            "backend": "numpy+scipy",
            "n_channels": self.n_channels,
            "n_classes": self.n_classes,
            "fs": self.fs,
            "total_parameters": 0,
            "emotion_classes": EMOTION_CLASSES,
            "description": (
                "Feature-based fallback: band powers + FAA → quadrant classification. "
                "PyTorch not available for full CNN-KAN model."
            ),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _predict_cnn(self, signals: np.ndarray, fs: float) -> Dict[str, Any]:
        """Run CNN-KAN forward pass and decode output."""
        try:
            # Pad or trim channels to match model expectation
            n_ch = self._model.n_channels
            if signals.shape[0] < n_ch:
                pad = np.zeros((n_ch - signals.shape[0], signals.shape[1]), dtype=np.float32)
                signals = np.concatenate([signals, pad], axis=0)
            elif signals.shape[0] > n_ch:
                signals = signals[:n_ch]

            # Z-score normalise per channel to handle amplitude variability
            mu = signals.mean(axis=1, keepdims=True)
            sd = signals.std(axis=1, keepdims=True) + 1e-8
            signals_norm = (signals - mu) / sd

            # Build batch tensor: (1, n_channels, n_samples)
            x = torch.from_numpy(signals_norm).unsqueeze(0)

            with torch.no_grad():
                logits = self._model(x)           # (1, n_classes)
                probs_t = torch.softmax(logits, dim=-1)[0]

            probs_np = probs_t.numpy().astype(float)
            class_idx = int(np.argmax(probs_np))
            emotion = EMOTION_CLASSES[class_idx]

            probs = {cls: round(float(p), 4) for cls, p in zip(EMOTION_CLASSES, probs_np)}

            # Derive valence/arousal from probabilities (weighted quadrant centroids)
            valence = sum(
                _QUADRANT_CENTROIDS[cls][0] * p for cls, p in probs.items()
            )
            arousal = sum(
                _QUADRANT_CENTROIDS[cls][1] * p for cls, p in probs.items()
            )

            return {
                "emotion": emotion,
                "probabilities": probs,
                "valence": round(float(valence), 4),
                "arousal": round(float(arousal), 4),
                "model_type": "cnn-kan",
            }

        except Exception as exc:
            logger.warning("CNN forward pass failed: %s — falling back", exc)
            return _feature_based_predict(signals, fs)

    def _fallback_short(self, signals: np.ndarray, fs: float) -> Dict[str, Any]:
        """Return a neutral result for signals too short for any meaningful processing."""
        probs = {cls: round(1.0 / len(EMOTION_CLASSES), 4) for cls in EMOTION_CLASSES}
        return {
            "emotion": "high_valence_low_arousal",  # neutral/calm default
            "probabilities": probs,
            "valence": 0.0,
            "arousal": 0.0,
            "model_type": "feature-based",
            "note": f"Signal too short ({signals.shape[-1]} samples); returning neutral",
        }


# ── Singleton factory ─────────────────────────────────────────────────────────

_instances: Dict[str, CNNKANEmotionClassifier] = {}
_instance_lock = threading.Lock()


def get_cnn_kan_classifier(user_id: str = "default") -> CNNKANEmotionClassifier:
    """Return a per-user (or shared default) CNNKANEmotionClassifier instance.

    Instances are cached so that the CNN model weights are initialised only once
    per user. For the default user_id, a single shared instance is used.

    Args:
        user_id: Identifier for the user; used to cache separate instances.

    Returns:
        CNNKANEmotionClassifier ready for inference.
    """
    with _instance_lock:
        if user_id not in _instances:
            _instances[user_id] = CNNKANEmotionClassifier()
        return _instances[user_id]
