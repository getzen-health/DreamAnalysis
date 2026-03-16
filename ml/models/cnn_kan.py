"""CNN-KAN hybrid model for 4-channel EEG emotion classification.

Architecture:
    Input: (batch, 4, 1024) — 4 channels × 4 sec @ 256 Hz
      → Conv1D(4, 32, kernel=64, stride=16) + BatchNorm + ReLU
      → Conv1D(32, 64, kernel=16, stride=4) + BatchNorm + ReLU
      → AdaptiveAvgPool1d(8)
      → Flatten → 512 features
      → KANLinear(512, 64, grid_size=5)  — learnable B-spline activations
      → KANLinear(64, 3)                 — 3-class: positive/neutral/negative

KAN (Kolmogorov-Arnold Network) replaces traditional MLP layers with learnable
activation functions parameterised as B-spline basis expansions. Each
input-output edge has its own spline, giving richer function approximation
with fewer parameters than a deep MLP.

Reference:
    Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024).
    arXiv:2404.19756

Classes (3-class valence):
    0 — positive  (happy, excited, calm)
    1 — neutral   (alert, focused, baseline)
    2 — negative  (sad, fearful, stressed)
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Emotion class definitions ────────────────────────────────────────────────

EMOTION_CLASSES: List[str] = ["positive", "neutral", "negative"]

# Valence centroids used in the feature-based fallback and for soft-prob decoding
_VALENCE_CENTROIDS = {
    "positive": 0.65,
    "neutral":  0.0,
    "negative": -0.65,
}

# ── PyTorch guard ────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    log.info("PyTorch not available — CNN-KAN will use feature-based fallback")


# ── B-spline helpers ─────────────────────────────────────────────────────────

def _b_spline_basis(
    x: "torch.Tensor",
    grid: "torch.Tensor",
    spline_order: int,
) -> "torch.Tensor":
    """Evaluate B-spline basis functions on a uniform grid via Cox-de Boor recursion.

    Standard B-spline construction:
    - Order-0 basis: indicator B_i^0(x) = 1 if t_i <= x < t_{i+1}, else 0.
    - Order-k basis: B_i^k(x) = (x - t_i)/(t_{i+k} - t_i) * B_i^{k-1}(x)
                               + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1}^{k-1}(x)

    Each recursion step reduces the number of basis functions by 1.
    After spline_order steps: n_basis = n_knots - 1 - spline_order = grid_size + spline_order.

    Args:
        x:            (..., in_features) — input values.  Must be clamped to grid domain
                      before calling (caller responsibility).
        grid:         (in_features, n_knots) — knot positions per input feature.
                      n_knots = grid_size + 2*spline_order + 1 (extended knot vector).
        spline_order: Polynomial order of the B-spline (3 = cubic, default).

    Returns:
        basis: (..., in_features, grid_size + spline_order).
    """
    # Expand x for broadcasting against the knot dimension: (..., in, 1)
    x = x.unsqueeze(-1)

    # Order-0: indicator functions — shape (..., in_features, n_knots-1)
    basis = ((x >= grid[..., :-1]) & (x < grid[..., 1:])).float()

    # Cox-de Boor recursion: at each step k, current basis has n functions.
    # Result has n-1 functions (one fewer per step).
    for k in range(1, spline_order + 1):
        n = basis.shape[-1]  # current number of basis functions

        # Left term coefficients: (x - t_i) / (t_{i+k} - t_i)
        # i ranges over 0..n-2; need grid slices of length n-1.
        t_i    = grid[..., : n - 1]           # (in, n-1)
        t_ik   = grid[..., k : n + k - 1]     # (in, n-1)
        left_d = t_ik - t_i
        left_d = torch.where(left_d.abs() < 1e-8, torch.ones_like(left_d), left_d)
        left_c = (x - t_i) / left_d           # (..., in, n-1)

        # Right term coefficients: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1})
        t_ik1  = grid[..., k + 1 : n + k]     # (in, n-1)
        t_i1   = grid[..., 1 : n]             # (in, n-1)
        right_d = t_ik1 - t_i1
        right_d = torch.where(right_d.abs() < 1e-8, torch.ones_like(right_d), right_d)
        right_c = (t_ik1 - x) / right_d       # (..., in, n-1)

        basis = left_c * basis[..., :-1] + right_c * basis[..., 1:]  # (..., in, n-1)

    return basis  # (..., in_features, grid_size + spline_order)


# ── KANLinear layer ──────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:

    class KANLinear(nn.Module):
        """KAN linear layer using learnable B-spline basis functions.

        Replaces a standard linear layer with per-edge learnable activations.
        Each connection (input_i → output_j) has its own B-spline whose shape
        is determined by `grid_size` learnable knot intervals.

        Forward pass:
            y_j = Σ_i [ base(x_i) * base_weight[j,i]
                      + spline(x_i) @ spline_weight[j,i] ]

        where base(x) = SiLU(x) and spline(x) is a B-spline basis expansion.

        Args:
            in_features:   Number of input dimensions.
            out_features:  Number of output dimensions.
            grid_size:     Number of B-spline intervals (controls expressiveness).
                           Default 5.
            spline_order:  Polynomial order of the spline basis. Default 3.
            grid_range:    (lo, hi) — domain of the spline grid. Default (-1, 1).
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            grid_size: int = 5,
            spline_order: int = 3,
            grid_range: Tuple[float, float] = (-1.0, 1.0),
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.grid_size = grid_size
            self.spline_order = spline_order
            self.grid_range = grid_range

            # n_basis = grid_size + spline_order  (number of B-spline basis functions)
            self.n_basis = grid_size + spline_order

            # Uniform knot grid: (grid_size + 2*spline_order + 1) knots per feature
            # Extended by spline_order knots on each side for boundary conditions
            lo, hi = grid_range
            base_knots = torch.linspace(lo, hi, grid_size + 1)
            step = (hi - lo) / grid_size
            left_ext = torch.tensor(
                [lo - step * k for k in range(spline_order, 0, -1)]
            )
            right_ext = torch.tensor(
                [hi + step * k for k in range(1, spline_order + 1)]
            )
            grid = torch.cat([left_ext, base_knots, right_ext])  # (n_knots,)

            # Expand to per-feature grid: (in_features, n_knots)
            self.register_buffer(
                "grid", grid.unsqueeze(0).expand(in_features, -1).contiguous()
            )

            # Spline weights: (out_features, in_features, n_basis)
            self.spline_weight = nn.Parameter(
                torch.empty(out_features, in_features, self.n_basis)
            )
            # Residual linear (base activation) weights: (out_features, in_features)
            self.base_weight = nn.Parameter(
                torch.empty(out_features, in_features)
            )

            self._reset_parameters()

        def _reset_parameters(self) -> None:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (batch, in_features) — input activations.

            Returns:
                y: (batch, out_features).
            """
            # ── Base (residual SiLU) branch ───────────────────────────────
            # base_weight: (out, in)  ×  x^T: (in, batch)  → (out, batch)
            base_out = F.linear(F.silu(x), self.base_weight)  # (batch, out)

            # ── B-spline branch ───────────────────────────────────────────
            # Clamp x to grid domain before evaluating basis
            lo, hi = self.grid_range
            x_clamped = x.clamp(lo, hi)  # (batch, in)

            # Evaluate B-spline basis: (batch, in, n_basis)
            basis = _b_spline_basis(x_clamped, self.grid, self.spline_order)

            # Contract with spline weights:
            # basis:          (batch, in, n_basis)
            # spline_weight:  (out, in, n_basis)
            # → einsum "b i k, o i k → b o"
            spline_out = torch.einsum("bik,oik->bo", basis, self.spline_weight)

            return base_out + spline_out

        def extra_repr(self) -> str:
            return (
                f"in={self.in_features}, out={self.out_features}, "
                f"grid_size={self.grid_size}, spline_order={self.spline_order}, "
                f"n_basis={self.n_basis}"
            )

    # ── Full CNN-KAN model ───────────────────────────────────────────────────

    class CNNKANModel(nn.Module):
        """CNN-KAN hybrid for 4-channel Muse 2 EEG emotion classification.

        CNN backbone extracts multi-scale temporal features; KAN head performs
        the final classification using learnable B-spline activations.

        Input shape:  (batch, 4, 1024)  — 4 channels × 1024 samples (4 s @ 256 Hz)
        Output shape: (batch, 3)        — logits over [positive, neutral, negative]
        """

        def __init__(
            self,
            n_channels: int = 4,
            n_classes: int = 3,
            grid_size: int = 5,
            spline_order: int = 3,
        ) -> None:
            super().__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes

            # ── CNN backbone ──────────────────────────────────────────────
            # Stage 1: broad temporal filter (64-sample kernel ≈ 250 ms)
            self.conv1 = nn.Conv1d(
                n_channels, 32, kernel_size=64, stride=16, padding=24
            )
            self.bn1 = nn.BatchNorm1d(32)

            # Stage 2: finer temporal filter
            self.conv2 = nn.Conv1d(32, 64, kernel_size=16, stride=4, padding=6)
            self.bn2 = nn.BatchNorm1d(64)

            # Collapse time dimension to fixed size
            self.pool = nn.AdaptiveAvgPool1d(8)

            # ── KAN classification head ───────────────────────────────────
            cnn_flat_dim = 64 * 8  # 512
            self.kan1 = KANLinear(
                cnn_flat_dim, 64,
                grid_size=grid_size,
                spline_order=spline_order,
            )
            self.dropout = nn.Dropout(p=0.3)
            self.kan2 = KANLinear(
                64, n_classes,
                grid_size=grid_size,
                spline_order=spline_order,
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (batch, n_channels, n_samples)

            Returns:
                logits: (batch, n_classes)
            """
            x = F.relu(self.bn1(self.conv1(x)))   # (batch, 32, ~64)
            x = F.relu(self.bn2(self.conv2(x)))   # (batch, 64, ~16)
            x = self.pool(x)                       # (batch, 64, 8)
            x = x.flatten(start_dim=1)             # (batch, 512)
            x = F.relu(self.kan1(x))               # (batch, 64)
            x = self.dropout(x)
            return self.kan2(x)                    # (batch, n_classes)

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def save(self, path: "Path | str") -> None:
            """Save model weights and config as a .pt checkpoint."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "n_channels": self.n_channels,
                    "n_classes": self.n_classes,
                    "grid_size": self.kan1.grid_size,
                    "spline_order": self.kan1.spline_order,
                },
                path,
            )

        @classmethod
        def load(cls, path: "Path | str") -> "CNNKANModel":
            """Load a saved checkpoint."""
            ckpt = torch.load(path, map_location="cpu")
            model = cls(
                n_channels=ckpt.get("n_channels", 4),
                n_classes=ckpt.get("n_classes", 3),
                grid_size=ckpt.get("grid_size", 5),
                spline_order=ckpt.get("spline_order", 3),
            )
            model.load_state_dict(ckpt["state_dict"])
            return model


# ── Band-power helper (scipy preferred, numpy fallback) ──────────────────────

def _band_power(signal: np.ndarray, fs: float, flo: float, fhi: float) -> float:
    try:
        from scipy.signal import welch
        nperseg = min(len(signal), int(fs * 2))
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        mask = (freqs >= flo) & (freqs <= fhi)
        return float(np.mean(psd[mask])) if mask.any() else 1e-10
    except Exception:
        n = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal)) ** 2 / n
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= flo) & (freqs <= fhi)
        return float(np.mean(fft_vals[mask])) if mask.any() else 1e-10


def _faa(signals: np.ndarray, fs: float) -> float:
    """Frontal Alpha Asymmetry: ln(AF8_alpha) − ln(AF7_alpha).

    BrainFlow Muse 2 channel order: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10.
    """
    if signals.ndim < 2 or signals.shape[0] < 3:
        return 0.0
    af7_a = max(_band_power(signals[1], fs, 8.0, 13.0), 1e-12)
    af8_a = max(_band_power(signals[2], fs, 8.0, 13.0), 1e-12)
    return float(np.log(af8_a) - np.log(af7_a))


def _feature_based_predict(signals: np.ndarray, fs: float) -> Dict[str, Any]:
    """3-class valence classification from band powers + FAA.

    Used when PyTorch is unavailable or the signal is too short for CNN.
    """
    primary = signals[0] if signals.ndim == 2 else signals
    n_ch = signals.shape[0] if signals.ndim == 2 else 1

    alpha = _band_power(primary, fs, 8.0, 13.0)
    beta  = _band_power(primary, fs, 13.0, 30.0)
    high_beta = _band_power(primary, fs, 20.0, 30.0)

    denom = max(alpha + beta, 1e-12)
    ab_ratio = alpha / max(beta, 1e-12)
    valence_ab = float(np.tanh((ab_ratio - 0.7) * 2.0))

    if n_ch >= 3:
        faa_v = float(np.clip(np.tanh(_faa(signals, fs) * 2.0), -1.0, 1.0))
        valence = float(np.clip(0.5 * valence_ab + 0.5 * faa_v, -1.0, 1.0))
    else:
        valence = float(np.clip(valence_ab, -1.0, 1.0))

    # Map valence to 3 classes
    if valence >= 0.2:
        emotion = "positive"
    elif valence <= -0.2:
        emotion = "negative"
    else:
        emotion = "neutral"

    probs = _soft_probs(valence)
    return {
        "emotion": emotion,
        "probabilities": probs,
        "valence": round(valence, 4),
        "model_type": "feature-based",
    }


def _soft_probs(valence: float) -> Dict[str, float]:
    """Convert scalar valence into soft probability distribution over 3 classes."""
    scores = {
        cls: 1.0 / (abs(valence - v_c) + 1e-6)
        for cls, v_c in _VALENCE_CENTROIDS.items()
    }
    total = sum(scores.values())
    return {cls: round(float(s / total), 4) for cls, s in scores.items()}


# ── Public classifier wrapper ────────────────────────────────────────────────

class CNNKANClassifier:
    """CNN-KAN emotion classifier for 4-channel Muse 2 EEG.

    Wraps CNNKANModel with:
    - automatic weight loading from models/saved/cnn_kan_emotion.pt
    - per-channel z-score normalisation
    - feature-based fallback when PyTorch is unavailable or signal is too short
    - thread-safe singleton factory via get_cnn_kan_classifier()

    Args:
        model_path: Optional path to a saved .pt checkpoint. If None, attempts
                    to auto-load from the default saved-models directory.
        n_channels: Number of EEG channels (default 4 for Muse 2).
        n_classes:  Number of output emotion classes (default 3).
        fs:         Sampling frequency in Hz (default 256.0).
    """

    DEFAULT_MODEL_NAME = "cnn_kan_emotion.pt"

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_channels: int = 4,
        n_classes: int = 3,
        fs: float = 256.0,
    ) -> None:
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.fs = fs
        self._model: Optional[Any] = None

        if not _TORCH_AVAILABLE:
            log.info("CNN-KAN: PyTorch unavailable — feature-based mode only")
            return

        # Resolve model path
        if model_path is None:
            saved_dir = Path(__file__).parent / "saved"
            model_path_obj = saved_dir / self.DEFAULT_MODEL_NAME
        else:
            model_path_obj = Path(model_path)

        if model_path_obj.exists():
            try:
                self._model = CNNKANModel.load(model_path_obj)
                self._model.eval()
                log.info(
                    "CNN-KAN loaded from %s — %d params",
                    model_path_obj,
                    self._model.count_parameters(),
                )
            except Exception as exc:
                log.warning("CNN-KAN load failed: %s — untrained model", exc)
                self._model = self._init_untrained(n_channels, n_classes)
        else:
            log.info("CNN-KAN: no saved weights at %s — using untrained model", model_path_obj)
            self._model = self._init_untrained(n_channels, n_classes)

    @staticmethod
    def _init_untrained(n_channels: int, n_classes: int) -> "CNNKANModel":
        model = CNNKANModel(n_channels=n_channels, n_classes=n_classes)
        model.eval()
        return model

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(self, eeg: np.ndarray, fs: Optional[float] = None) -> Dict[str, Any]:
        """Classify emotion valence from raw EEG.

        Args:
            eeg: (4, n_samples) multi-channel or (n_samples,) single-channel.
            fs:  Sampling frequency in Hz. Defaults to self.fs.

        Returns:
            dict with:
                emotion       — "positive" | "neutral" | "negative"
                probabilities — {class: float} summing to 1.0
                valence       — continuous estimate in [-1, 1]
                model_type    — "cnn-kan" | "feature-based"
        """
        fs = fs or self.fs
        eeg = np.asarray(eeg, dtype=np.float32)

        if eeg.ndim == 1:
            signals = eeg[np.newaxis, :]
        elif eeg.ndim == 2:
            signals = eeg
        else:
            signals = eeg.reshape(self.n_channels, -1)

        # Minimum length: at least 64 samples for CNN stage 1
        if signals.shape[-1] < 64:
            return self._neutral_result(signals)

        if self._model is not None and _TORCH_AVAILABLE:
            return self._cnn_predict(signals)

        return _feature_based_predict(signals, fs)

    def get_model_info(self) -> Dict[str, Any]:
        """Return architecture summary."""
        if self._model is not None and _TORCH_AVAILABLE:
            return {
                "architecture": "CNN-KAN",
                "backend": "pytorch",
                "n_channels": self.n_channels,
                "n_classes": self.n_classes,
                "fs": self.fs,
                "total_parameters": self._model.count_parameters(),
                "kan_grid_size": self._model.kan1.grid_size,
                "kan_spline_order": self._model.kan1.spline_order,
                "kan_n_basis": self._model.kan1.n_basis,
                "emotion_classes": EMOTION_CLASSES,
                "description": (
                    "1D-CNN temporal extraction + KAN classification head "
                    "(B-spline basis). Designed for 4-channel Muse 2 EEG, "
                    "4-second epochs @ 256 Hz. 3-class valence output."
                ),
                "layers": [
                    "Conv1d(4→32, k=64, s=16) + BN + ReLU",
                    "Conv1d(32→64, k=16, s=4) + BN + ReLU",
                    "AdaptiveAvgPool1d(8) → flatten(512)",
                    f"KANLinear(512→64, grid={self._model.kan1.grid_size}) + ReLU + Dropout",
                    f"KANLinear(64→3, grid={self._model.kan2.grid_size})",
                ],
            }
        return {
            "architecture": "CNN-KAN (feature-based fallback)",
            "backend": "numpy+scipy",
            "n_channels": self.n_channels,
            "n_classes": self.n_classes,
            "emotion_classes": EMOTION_CLASSES,
            "description": "Band-power + FAA → 3-class valence. PyTorch unavailable.",
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _cnn_predict(self, signals: np.ndarray) -> Dict[str, Any]:
        try:
            import torch

            n_ch = self._model.n_channels
            # Pad / trim channels to match model
            if signals.shape[0] < n_ch:
                pad = np.zeros(
                    (n_ch - signals.shape[0], signals.shape[1]), dtype=np.float32
                )
                signals = np.concatenate([signals, pad], axis=0)
            elif signals.shape[0] > n_ch:
                signals = signals[:n_ch]

            # Z-score normalise per channel
            mu = signals.mean(axis=1, keepdims=True)
            sd = signals.std(axis=1, keepdims=True) + 1e-8
            signals_norm = (signals - mu) / sd

            x = torch.from_numpy(signals_norm).unsqueeze(0)  # (1, ch, samples)

            with torch.no_grad():
                logits = self._model(x)                        # (1, 3)
                probs_t = torch.softmax(logits, dim=-1)[0]

            probs_np = probs_t.numpy().astype(float)
            class_idx = int(np.argmax(probs_np))
            emotion = EMOTION_CLASSES[class_idx]

            probs = {
                cls: round(float(p), 4)
                for cls, p in zip(EMOTION_CLASSES, probs_np)
            }

            # Weighted-centroid valence estimate
            valence = float(
                sum(_VALENCE_CENTROIDS[cls] * p for cls, p in probs.items())
            )

            return {
                "emotion": emotion,
                "probabilities": probs,
                "valence": round(valence, 4),
                "model_type": "cnn-kan",
            }

        except Exception as exc:
            log.warning("CNN-KAN forward pass failed: %s — falling back", exc)
            return _feature_based_predict(signals, self.fs)

    def _neutral_result(self, signals: np.ndarray) -> Dict[str, Any]:
        probs = {cls: round(1.0 / len(EMOTION_CLASSES), 4) for cls in EMOTION_CLASSES}
        return {
            "emotion": "neutral",
            "probabilities": probs,
            "valence": 0.0,
            "model_type": "feature-based",
            "note": (
                f"Signal too short ({signals.shape[-1]} samples); returning neutral. "
                "Need ≥ 64 samples."
            ),
        }


# ── Singleton factory ────────────────────────────────────────────────────────

_instances: Dict[str, CNNKANClassifier] = {}
_lock = threading.Lock()


def get_cnn_kan_classifier(user_id: str = "default") -> CNNKANClassifier:
    """Return a cached CNNKANClassifier instance for the given user.

    Instances are created once and reused so model weights are loaded only
    once per user. Thread-safe.

    Args:
        user_id: Identifier for per-user instance caching.

    Returns:
        CNNKANClassifier ready for inference.
    """
    with _lock:
        if user_id not in _instances:
            _instances[user_id] = CNNKANClassifier()
        return _instances[user_id]
