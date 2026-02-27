"""EEGNet — Compact CNN for EEG-based Emotion Classification.

Reference: Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural Network
for EEG-based Brain-Computer Interfaces." Journal of Neural Engineering.
https://doi.org/10.1088/1741-2552/aace8c

Why EEGNet for this project:
──────────────────────────────────────────────────────────────────────────────
1. DEVICE-AGNOSTIC: Takes (n_channels, n_samples) as input. Works with 4-channel
   Muse 2 and 8-channel OpenBCI Cyton and 16-channel Cyton+Daisy without
   retraining the architecture — just pass the right n_channels at init time.

2. DATA-EFFICIENT: Only ~2,548 parameters for the 4-channel version. Works well
   with small datasets (our pilot study target: ~6,000 labeled epochs).

3. LEARNS DIRECTLY FROM RAW EEG: No hand-crafted features needed. The temporal
   convolution learns frequency decomposition; the depthwise convolution learns
   spatial patterns (like FAA) automatically.

4. ONNX-EXPORTABLE: PyTorch → ONNX export → on-device mobile inference via
   onnxruntime-web (already in the frontend stack). No server needed for basic
   emotion classification on iOS/Android.

Architecture (default params, 4-channel Muse 2, 256 Hz, 4-sec epochs = 1024 samples):
──────────────────────────────────────────────────────────────────────────────
Input:           (batch, 1, 4, 1024)

Block 1 — Temporal convolution (learns frequency filters):
  Conv2d(1, F1=8, kernel=(1, 64), padding=(0, 32))  → (batch, 8, 4, 1024)
  BatchNorm2d(8)
  DepthwiseConv2d(8, 8*D=16, kernel=(n_channels, 1), groups=8) → (batch, 16, 1, 1024)
  BatchNorm2d(16)
  ELU
  AvgPool2d(kernel=(1, 4))                           → (batch, 16, 1, 256)
  Dropout(0.25)

Block 2 — Separable convolution (learns temporal summaries):
  SeparableConv2d: depthwise (1, 16) + pointwise (1, 1) → (batch, F2=16, 1, 256)
  BatchNorm2d(16)
  ELU
  AvgPool2d(kernel=(1, 8))                           → (batch, 16, 1, 32)
  Dropout(0.25)

Classifier:
  Flatten                                            → (batch, 512)
  Linear(512, n_classes)                             → (batch, 6)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── Architecture helpers ───────────────────────────────────────────────────

class _DepthwiseSeparableConv(nn.Module):
    """Depthwise + pointwise convolution (SeparableConv2d)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int]):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=(0, kernel_size[1] // 2), groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, (1, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


# ── Main EEGNet model ──────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """Compact convolutional neural network for EEG emotion classification.

    Parameters
    ----------
    n_classes   : number of output classes (6 for emotions)
    n_channels  : number of EEG channels (4 = Muse 2, 8 = OpenBCI Cyton, 16 = Cyton+Daisy)
    n_samples   : number of time samples per epoch (256 Hz × 4 sec = 1024)
    F1          : number of temporal filters (default 8)
    D           : depth multiplier for depthwise conv (F2 = F1 * D, default 2 → F2=16)
    dropout_rate: dropout probability (default 0.25)
    """

    def __init__(
        self,
        n_classes: int = 6,
        n_channels: int = 4,
        n_samples: int = 1024,
        F1: int = 8,
        D: int = 2,
        dropout_rate: float = 0.25,
    ):
        super().__init__()

        F2 = F1 * D
        kernel_length = max(32, n_samples // 16)  # ≈250 ms at 256 Hz → 64 samples

        # ── Block 1: Temporal + Depthwise Spatial ─────────────────────────
        self.temporal_conv = nn.Conv2d(
            1, F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise_conv = nn.Conv2d(
            F1, F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # ── Block 2: Separable Temporal ───────────────────────────────────
        sep_kernel = max(16, n_samples // 64)   # ≈62.5 ms at 256 Hz → 16 samples
        self.separable_conv = _DepthwiseSeparableConv(F2, F2, kernel_size=(1, sep_kernel))
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # ── Compute flatten size dynamically ─────────────────────────────
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat_size = self._forward_features(dummy).shape[1]

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Linear(flat_size, n_classes)

        # Store for serialisation
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.F1 = F1
        self.D = D
        self.dropout_rate = dropout_rate

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature extraction blocks (shared by forward and flatten-size probe)."""
        # Block 1
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        # Block 2
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch, n_channels, n_samples)  OR  (batch, 1, n_channels, n_samples)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # → (batch, 1, n_channels, n_samples)
        features = self._forward_features(x)
        return self.classifier(features)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Return softmax probabilities as numpy array."""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return F.softmax(logits, dim=-1).cpu().numpy()

    # ── Serialisation helpers ─────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model weights + architecture params to a single .pt file."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "n_classes":    self.n_classes,
                "n_channels":   self.n_channels,
                "n_samples":    self.n_samples,
                "F1":           self.F1,
                "D":            self.D,
                "dropout_rate": self.dropout_rate,
            },
        }, path)
        log.info("EEGNet saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "EEGNet":
        """Load EEGNet from a .pt file saved with save()."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        log.info("EEGNet loaded ← %s  (%d-ch)", path, model.n_channels)
        return model

    def export_onnx(self, path: str | Path) -> None:
        """Export to ONNX for mobile inference via onnxruntime-web."""
        self.eval()
        dummy = torch.zeros(1, self.n_channels, self.n_samples)
        torch.onnx.export(
            self, dummy, str(path),
            input_names=["eeg"],
            output_names=["logits"],
            dynamic_axes={"eeg": {0: "batch"}},
            opset_version=17,
        )
        log.info("EEGNet ONNX exported → %s", path)


# ── Wrapper that plugs into EmotionClassifier ──────────────────────────────

class EEGNetEmotionClassifier:
    """Thin wrapper around EEGNet that matches the EmotionClassifier._predict_* interface.

    Handles:
    - Loading the right channel-count model file automatically
    - Raw-EEG → softmax probabilities → emotion dict conversion
    - EMA smoothing (same alpha as EmotionClassifier)
    - Graceful fallback when model file doesn't exist
    """

    # Model files: ml/models/saved/eegnet_emotion_{n_ch}ch.pt
    _SAVE_DIR = Path(__file__).parent / "saved"
    _EMA_ALPHA = 0.35

    EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

    def __init__(self) -> None:
        self._models: dict[int, EEGNet] = {}      # keyed by n_channels
        self._benchmarks: dict[int, float] = {}   # accuracy per channel count
        self._ema: dict[int, Optional[np.ndarray]] = {}

    # ── Model loading ─────────────────────────────────────────────────────

    def _model_path(self, n_channels: int) -> Path:
        return self._SAVE_DIR / f"eegnet_emotion_{n_channels}ch.pt"

    def _benchmark_path(self, n_channels: int) -> Path:
        return self._SAVE_DIR / f"eegnet_emotion_{n_channels}ch_benchmark.txt"

    def load_for_channels(self, n_channels: int) -> bool:
        """Load (or cache-hit) EEGNet for the given channel count. Returns True if loaded."""
        if n_channels in self._models:
            return True
        pt_path = self._model_path(n_channels)
        if not pt_path.exists():
            return False
        try:
            self._models[n_channels] = EEGNet.load(pt_path)
            # Load benchmark
            bm_path = self._benchmark_path(n_channels)
            if bm_path.exists():
                self._benchmarks[n_channels] = float(bm_path.read_text().strip())
            else:
                self._benchmarks[n_channels] = 0.0
            self._ema[n_channels] = None
            log.info(
                "EEGNet loaded for %d-ch (accuracy=%.2f%%)",
                n_channels, self._benchmarks[n_channels] * 100,
            )
            return True
        except Exception as exc:
            log.warning("EEGNet load failed for %d-ch: %s", n_channels, exc)
            return False

    def is_available(self, n_channels: int, min_accuracy: float = 0.60) -> bool:
        """True if a model for this channel count is loaded AND meets accuracy threshold."""
        if not self.load_for_channels(n_channels):
            return False
        return self._benchmarks.get(n_channels, 0.0) >= min_accuracy

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> dict:
        """Predict emotion from raw EEG.

        Parameters
        ----------
        eeg : 2D array (n_channels, n_samples)
        fs  : sampling rate in Hz
        """
        n_channels, n_samples = eeg.shape
        model = self._models[n_channels]

        # Normalise each channel to zero-mean unit-variance
        eeg_norm = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-7)

        # Pad or trim to model's expected n_samples
        target = model.n_samples
        if n_samples < target:
            pad = target - n_samples
            eeg_norm = np.pad(eeg_norm, ((0, 0), (0, pad)), mode="edge")
        elif n_samples > target:
            eeg_norm = eeg_norm[:, :target]

        x = torch.from_numpy(eeg_norm.astype(np.float32)).unsqueeze(0)  # (1, ch, t)
        probs = model.predict_proba(x)[0]   # (n_classes,)

        # EMA smoothing
        ema = self._ema.get(n_channels)
        if ema is None:
            ema = probs
        else:
            ema = self._EMA_ALPHA * probs + (1 - self._EMA_ALPHA) * ema
        self._ema[n_channels] = ema
        smoothed = ema / (ema.sum() + 1e-10)

        emotion_idx = int(np.argmax(smoothed))
        emotion = self.EMOTIONS[emotion_idx]

        # Derive valence / arousal from class probabilities
        # happy=0, sad=1, angry=2, fearful=3, relaxed=4, focused=5
        valence = (
            smoothed[0] * 0.9    # happy → very positive
            + smoothed[4] * 0.4  # relaxed → mildly positive
            + smoothed[5] * 0.2  # focused → slightly positive
            - smoothed[1] * 0.9  # sad → very negative
            - smoothed[2] * 0.7  # angry → negative
            - smoothed[3] * 0.8  # fearful → negative
        )
        arousal = (
            smoothed[0] * 0.7    # happy → moderate-high arousal
            + smoothed[2] * 0.9  # angry → very high arousal
            + smoothed[3] * 0.8  # fearful → high arousal
            + smoothed[5] * 0.7  # focused → moderate arousal
            - smoothed[1] * 0.5  # sad → lower arousal
            - smoothed[4] * 0.8  # relaxed → low arousal
        )
        arousal = float(np.clip(arousal + 0.5, 0.0, 1.0))  # shift to 0–1 range
        valence = float(np.clip(valence, -1.0, 1.0))

        # Stress: high beta + negative emotions
        stress_index = float(np.clip(
            smoothed[2] * 0.85 + smoothed[3] * 0.8 - smoothed[4] * 0.7, 0.0, 1.0
        ))
        focus_index = float(np.clip(smoothed[5] * 0.9 + smoothed[0] * 0.3, 0.0, 1.0))
        relax_index = float(np.clip(smoothed[4] * 0.9 + smoothed[1] * 0.2, 0.0, 1.0))

        return {
            "emotion":            emotion,
            "probabilities":      {e: round(float(p), 4) for e, p in zip(self.EMOTIONS, smoothed)},
            "valence":            round(valence, 3),
            "arousal":            round(arousal, 3),
            "stress_index":       round(stress_index, 3),
            "focus_index":        round(focus_index, 3),
            "relaxation_index":   round(relax_index, 3),
            "model_type":         f"eegnet_{n_channels}ch",
        }
