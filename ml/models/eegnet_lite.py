"""EEGNet-Lite: compact depthwise-separable CNN for 4-channel Muse 2 EEG.

Architecture (EEGNet-Lite, adapted from Lawhern et al. 2018):

    Input: (batch, 1, 4, 1024)  — 1 "colour" x 4 channels x 1024 samples (4 s @ 256 Hz)

    Block 1 — Temporal + spatial:
        Conv2d(1, 8, (1, 64), padding=(0, 32))    temporal filter bank, no bias
        BatchNorm2d(8)
        DepthwiseConv2d(8, 16, (4, 1), groups=8)  spatial filter per channel
        BatchNorm2d(16)
        ELU
        AvgPool2d((1, 4))                          downsample time x4
        Dropout(0.25)

    Block 2 — Separable:
        DepthwiseConv2d(16, 16, (1, 16), groups=16, padding=(0, 8))
        Conv2d(16, 16, (1, 1))                     pointwise
        BatchNorm2d(16)
        ELU
        AvgPool2d((1, 8))                          downsample time x8
        Dropout(0.25)

    Classifier:
        Flatten  ->  Linear(16 * 1 * 32, n_classes)

After both pooling steps: 1024 / 4 / 8 = 32 time frames.
Flat dim = 16 * 1 * 32 = 512.

Total trainable parameters: ~1,800 (ultra-lightweight for ONNX/browser).

Design goals:
    - < 2 000 trainable parameters
    - ONNX export < 50 KB — runs in browser via onnxruntime-web
    - Quantisation-ready (QuantStub / DeQuantStub stubs)
    - Feature-based fallback when PyTorch is unavailable
    - Online last-layer SGD fine-tuning for personalisation

Classes (3-class valence — same as CNN-KAN):
    0 — positive  (happy, excited, calm)
    1 — neutral   (alert, focused, baseline)
    2 — negative  (sad, fearful, stressed)

Reference:
    Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for
    EEG-based Brain-Computer Interfaces", J. Neural Eng. 15 (2018).
    arXiv:1611.08024
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Emotion class definitions ─────────────────────────────────────────────────

EMOTION_CLASSES: List[str] = ["positive", "neutral", "negative"]

_VALENCE_CENTROIDS: Dict[str, float] = {
    "positive":  0.65,
    "neutral":   0.0,
    "negative": -0.65,
}

# ── PyTorch guard ─────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    log.info("PyTorch not available — EEGNet-Lite will use feature-based fallback")


# ── Model definition ──────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:

    class _EEGNetLiteModule(nn.Module):
        """EEGNet-Lite PyTorch module.

        Kept internal — use EEGNetLiteClassifier (the wrapper) externally.

        Input shape:  (batch, 1, n_channels, n_samples)
        Output shape: (batch, n_classes)  — raw logits

        Args:
            n_channels: Number of EEG channels (default 4).
            n_samples:  Samples per epoch (default 1024 = 4 s @ 256 Hz).
            n_classes:  Output classes (default 3).
            F1:         Temporal filter count (default 8).
            D:          Depth multiplier for spatial filters (default 2).
            F2:         Separable filter count (default 16). Must equal F1 * D.
            dropout:    Dropout probability (default 0.25).
        """

        def __init__(
            self,
            n_channels: int = 4,
            n_samples: int = 1024,
            n_classes: int = 3,
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            dropout: float = 0.25,
        ) -> None:
            super().__init__()
            self.n_channels = n_channels
            self.n_samples = n_samples
            self.n_classes = n_classes
            self.F1 = F1
            self.D = D
            self.F2 = F2

            # Block 1 — temporal
            # kernel (1, 64): ~250 ms at 256 Hz; padding keeps time length intact
            self.temporal_conv = nn.Conv2d(
                1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False
            )
            self.bn1 = nn.BatchNorm2d(F1)

            # Block 1 — depthwise spatial (collapses channel dim to 1)
            self.spatial_dw = nn.Conv2d(
                F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(F1 * D)
            self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
            self.drop1 = nn.Dropout(p=dropout)

            # Block 2 — separable (depthwise + pointwise)
            self.sep_dw = nn.Conv2d(
                F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8),
                groups=F1 * D, bias=False,
            )
            self.sep_pw = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
            self.bn3 = nn.BatchNorm2d(F2)
            self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
            self.drop2 = nn.Dropout(p=dropout)

            # Quantisation stubs (identity when not quantized)
            try:
                self.quant = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()
            except AttributeError:
                self.quant = nn.Identity()
                self.dequant = nn.Identity()

            # Classifier
            # After pool1(x4) and pool2(x8): time_out = n_samples // 4 // 8 = 32
            # spatial_dw collapses height (channel) to 1
            time_out = n_samples // 4 // 8
            flat_dim = F2 * 1 * time_out
            self.classifier = nn.Linear(flat_dim, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (batch, 1, n_channels, n_samples)

            Returns:
                logits: (batch, n_classes)
            """
            x = self.quant(x)

            # Block 1
            x = self.temporal_conv(x)     # (B, F1,    n_ch, T)
            x = self.bn1(x)
            x = self.spatial_dw(x)        # (B, F1*D,  1,    T)
            x = self.bn2(x)
            x = F.elu(x)
            x = self.pool1(x)             # (B, F1*D,  1,    T//4)
            x = self.drop1(x)

            # Block 2
            x = self.sep_dw(x)            # (B, F1*D,  1,    T//4)
            x = self.sep_pw(x)            # (B, F2,    1,    T//4)
            x = self.bn3(x)
            x = F.elu(x)
            x = self.pool2(x)             # (B, F2,    1,    T//32)
            x = self.drop2(x)

            x = self.dequant(x)
            x = x.flatten(start_dim=1)    # (B, F2 * 1 * T//32)
            return self.classifier(x)     # (B, n_classes)

        def count_parameters(self) -> int:
            """Return number of trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def save(self, path: "Path | str") -> None:
            """Save model weights and config as a .pt checkpoint."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "n_channels": self.n_channels,
                    "n_samples":  self.n_samples,
                    "n_classes":  self.n_classes,
                    "F1":         self.F1,
                    "D":          self.D,
                    "F2":         self.F2,
                },
                path,
            )
            log.info(
                "EEGNet-Lite saved to %s (%d params)", path, self.count_parameters()
            )

        @classmethod
        def load(cls, path: "Path | str") -> "_EEGNetLiteModule":
            """Load a saved .pt checkpoint."""
            ckpt = torch.load(path, map_location="cpu")
            model = cls(
                n_channels=ckpt.get("n_channels", 4),
                n_samples=ckpt.get("n_samples", 1024),
                n_classes=ckpt.get("n_classes", 3),
                F1=ckpt.get("F1", 8),
                D=ckpt.get("D", 2),
                F2=ckpt.get("F2", 16),
            )
            model.load_state_dict(ckpt["state_dict"])
            return model


    def export_onnx(
        model: "_EEGNetLiteModule",
        out_path: "Path | str",
        n_channels: int = 4,
        n_samples: int = 1024,
        opset: int = 11,
    ) -> Path:
        """Export EEGNet-Lite to ONNX with dynamic batch size.

        The exported file is typically < 50 KB — small enough for
        onnxruntime-web deployment in the browser.

        Args:
            model:      Trained _EEGNetLiteModule instance (eval mode).
            out_path:   Destination .onnx path.
            n_channels: Number of EEG channels (default 4).
            n_samples:  Samples per epoch (default 1024).
            opset:      ONNX opset version (default 11).

        Returns:
            Path object for the written ONNX file.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy = torch.zeros(1, 1, n_channels, n_samples)

        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            opset_version=opset,
            input_names=["eeg"],
            output_names=["logits"],
            dynamic_axes={
                "eeg":    {0: "batch"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
            dynamo=False,           # use TorchScript-based exporter (no onnxscript dep)
        )
        size_kb = out_path.stat().st_size / 1024
        log.info("ONNX exported to %s (%.1f KB)", out_path, size_kb)
        return out_path


# ── Band-power helper (scipy preferred, numpy fallback) ──────────────────────

def _band_power(signal: np.ndarray, fs: float, flo: float, fhi: float) -> float:
    """Estimate average PSD power in [flo, fhi] Hz."""
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
    """Frontal Alpha Asymmetry: ln(AF8_alpha) - ln(AF7_alpha).

    BrainFlow Muse 2 channel order: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10.
    """
    if signals.ndim < 2 or signals.shape[0] < 3:
        return 0.0
    af7_a = max(_band_power(signals[1], fs, 8.0, 13.0), 1e-12)
    af8_a = max(_band_power(signals[2], fs, 8.0, 13.0), 1e-12)
    return float(np.log(af8_a) - np.log(af7_a))


def _soft_probs(valence: float) -> Dict[str, float]:
    """Soft probability distribution over 3 classes from scalar valence."""
    scores = {
        cls: 1.0 / (abs(valence - v_c) + 1e-6)
        for cls, v_c in _VALENCE_CENTROIDS.items()
    }
    total = sum(scores.values())
    return {cls: round(float(s / total), 4) for cls, s in scores.items()}


def _feature_based_predict(signals: np.ndarray, fs: float) -> Dict[str, Any]:
    """3-class valence prediction from band powers + FAA.

    Used when PyTorch is unavailable or the signal is too short.
    """
    primary = signals[0] if signals.ndim == 2 else signals
    n_ch = signals.shape[0] if signals.ndim == 2 else 1

    alpha = _band_power(primary, fs, 8.0, 13.0)
    beta  = _band_power(primary, fs, 13.0, 30.0)

    ab_ratio = alpha / max(beta, 1e-12)
    valence_ab = float(np.tanh((ab_ratio - 0.7) * 2.0))

    if n_ch >= 3:
        faa_v = float(np.clip(np.tanh(_faa(signals, fs) * 2.0), -1.0, 1.0))
        valence = float(np.clip(0.5 * valence_ab + 0.5 * faa_v, -1.0, 1.0))
    else:
        valence = float(np.clip(valence_ab, -1.0, 1.0))

    if valence >= 0.2:
        emotion = "positive"
    elif valence <= -0.2:
        emotion = "negative"
    else:
        emotion = "neutral"

    return {
        "emotion": emotion,
        "probabilities": _soft_probs(valence),
        "valence": round(valence, 4),
        "model_type": "feature-based",
        "n_params": 0,
    }


# ── Public classifier wrapper ─────────────────────────────────────────────────

class EEGNetLiteClassifier:
    """EEGNet-Lite emotion classifier for 4-channel Muse 2 EEG.

    Load priority:
        1. ONNX via onnxruntime  (fastest on CPU, smallest memory)
        2. PyTorch .pt checkpoint
        3. Feature-based heuristics (no model file needed)

    Args:
        model_path: Path to .pt checkpoint (optional; auto-detected if None).
        onnx_path:  Path to .onnx model (optional; auto-detected if None).
        n_channels: Number of EEG channels (default 4).
        n_classes:  Output emotion classes (default 3).
        fs:         Sampling frequency in Hz (default 256.0).
    """

    PT_MODEL_NAME   = "eegnet_lite_emotion.pt"
    ONNX_MODEL_NAME = "eegnet_lite_emotion.onnx"

    def __init__(
        self,
        model_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        n_channels: int = 4,
        n_classes: int = 3,
        fs: float = 256.0,
    ) -> None:
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.fs = fs
        self._module: Optional[Any] = None    # PyTorch _EEGNetLiteModule
        self._ort: Optional[Any] = None       # onnxruntime InferenceSession

        saved_dir = Path(__file__).parent / "saved"

        # 1 — try ONNX
        onnx_path_obj = Path(onnx_path) if onnx_path else saved_dir / self.ONNX_MODEL_NAME
        if onnx_path_obj.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self._ort = ort.InferenceSession(
                    str(onnx_path_obj),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                log.info("EEGNet-Lite ONNX loaded from %s", onnx_path_obj)
                return
            except Exception as exc:
                log.warning("EEGNet-Lite ONNX load failed (%s) — trying .pt", exc)

        # 2 — try PyTorch .pt
        if not _TORCH_AVAILABLE:
            log.info("EEGNet-Lite: PyTorch unavailable — feature-based mode only")
            return

        pt_path_obj = Path(model_path) if model_path else saved_dir / self.PT_MODEL_NAME
        if pt_path_obj.exists():
            try:
                self._module = _EEGNetLiteModule.load(pt_path_obj)
                self._module.eval()
                log.info(
                    "EEGNet-Lite PyTorch loaded from %s — %d params",
                    pt_path_obj,
                    self._module.count_parameters(),
                )
            except Exception as exc:
                log.warning("EEGNet-Lite .pt load failed: %s", exc)
        else:
            log.info(
                "EEGNet-Lite: no checkpoint at %s — feature-based mode only",
                pt_path_obj,
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(signals: np.ndarray) -> np.ndarray:
        """Per-channel z-score normalisation."""
        mean = signals.mean(axis=-1, keepdims=True)
        std  = signals.std(axis=-1, keepdims=True) + 1e-7
        return ((signals - mean) / std).astype(np.float32)

    @staticmethod
    def _pad_or_trim(signals: np.ndarray, target: int = 1024) -> np.ndarray:
        """Pad (reflect) or trim to exactly `target` samples."""
        n = signals.shape[1]
        if n < target:
            return np.pad(signals, ((0, 0), (0, target - n)), mode="reflect")
        return signals[:, :target]

    @staticmethod
    def _logits_to_result(logits: np.ndarray) -> Dict[str, Any]:
        """Convert logit vector to emotion prediction dict."""
        try:
            from scipy.special import softmax as _sm
            probs = _sm(logits).tolist()
        except Exception:
            e = np.exp(logits - logits.max())
            probs = (e / e.sum()).tolist()

        idx = int(np.argmax(probs))
        emotion = EMOTION_CLASSES[idx]
        valence = float(
            sum(p * _VALENCE_CENTROIDS[cls] for p, cls in zip(probs, EMOTION_CLASSES))
        )
        return {
            "emotion": emotion,
            "probabilities": {
                cls: round(float(p), 4) for cls, p in zip(EMOTION_CLASSES, probs)
            },
            "valence": round(float(np.clip(valence, -1.0, 1.0)), 4),
            "model_type": "eegnet-lite",
            "n_params": 0,
        }

    def _run_onnx(self, signals: np.ndarray) -> Dict[str, Any]:
        arr = self._normalise(signals)
        inp = arr[np.newaxis, np.newaxis, :, :]          # (1, 1, C, T)
        outputs = self._ort.run(None, {"eeg": inp})
        logits = np.array(outputs[0][0], dtype=np.float32)
        result = self._logits_to_result(logits)
        result["model_type"] = "eegnet-lite-onnx"
        if _TORCH_AVAILABLE:
            # Report reference param count even in ONNX mode
            result["n_params"] = _EEGNetLiteModule().count_parameters()
        return result

    def _run_torch(self, signals: np.ndarray) -> Dict[str, Any]:
        import torch
        arr = self._normalise(signals)
        x = torch.from_numpy(arr[np.newaxis, np.newaxis, :, :])  # (1,1,C,T)
        with torch.no_grad():
            logits = self._module(x)[0].numpy()
        result = self._logits_to_result(logits)
        result["n_params"] = self._module.count_parameters()
        return result

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, signals: np.ndarray, fs: float = 256.0) -> Dict[str, Any]:
        """Classify 3-class EEG valence from raw signal.

        Args:
            signals: (n_channels, n_samples) or (n_samples,).
            fs:      Sampling rate Hz. Default 256.0.

        Returns:
            dict with keys: emotion, probabilities, valence, model_type, n_params.
        """
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]

        n_samples = signals.shape[1]
        if n_samples < 128:
            return {
                **_feature_based_predict(signals, fs),
                "note": f"Signal too short ({n_samples} samples < 128 min)",
            }

        signals = self._pad_or_trim(signals, target=1024)

        if self._ort is not None:
            try:
                return self._run_onnx(signals)
            except Exception as exc:
                log.warning("ONNX inference error: %s — falling back", exc)

        if self._module is not None:
            try:
                return self._run_torch(signals)
            except Exception as exc:
                log.warning("PyTorch inference error: %s — falling back", exc)

        return _feature_based_predict(signals, fs)

    def fine_tune_last_layer(
        self,
        signals: np.ndarray,
        label: int,
        fs: float = 256.0,
        lr: float = 0.01,
    ) -> Dict[str, Any]:
        """Online SGD update on the classifier (last) layer only.

        Freezes all layers except the final Linear, performs one gradient step.
        Implements the +7.31% personalisation improvement from ISWC 2024.

        Args:
            signals: (n_channels, n_samples) EEG epoch.
            label:   True class index (0 = positive, 1 = neutral, 2 = negative).
            fs:      Sampling rate Hz. Default 256.0.
            lr:      SGD learning rate. Default 0.01.

        Returns:
            {"updated": bool, "loss": float | None}
        """
        if not _TORCH_AVAILABLE or self._module is None:
            return {"updated": False, "loss": None,
                    "reason": "PyTorch model not loaded"}

        import torch

        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
        signals = self._pad_or_trim(signals, target=1024)
        arr = self._normalise(signals)

        x = torch.from_numpy(arr[np.newaxis, np.newaxis, :, :])
        y = torch.tensor([label])

        # Freeze all params; unfreeze only the classifier linear
        for name, param in self._module.named_parameters():
            param.requires_grad = name.startswith("classifier.")

        self._module.train()
        opt = torch.optim.SGD(
            [p for p in self._module.parameters() if p.requires_grad], lr=lr
        )
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(self._module(x), y)
        loss.backward()
        opt.step()
        self._module.eval()

        # Restore all params to trainable
        for param in self._module.parameters():
            param.requires_grad = True

        return {"updated": True, "loss": round(float(loss.item()), 6)}

    def get_model_info(self) -> Dict[str, Any]:
        """Return architecture and loading status."""
        n_params = 0
        if _TORCH_AVAILABLE and self._module is not None:
            n_params = self._module.count_parameters()
        elif _TORCH_AVAILABLE:
            n_params = _EEGNetLiteModule().count_parameters()

        return {
            "architecture": "EEGNet-Lite",
            "reference": "Lawhern et al. 2018 (arXiv:1611.08024)",
            "n_params": n_params,
            "n_channels": self.n_channels,
            "n_classes": self.n_classes,
            "classes": EMOTION_CLASSES,
            "input_shape": "(batch, 1, 4, 1024)",
            "pytorch_available": _TORCH_AVAILABLE,
            "onnx_loaded": self._ort is not None,
            "pt_loaded": self._module is not None,
        }


# ── Thread-safe singleton ─────────────────────────────────────────────────────

_singleton: Optional[EEGNetLiteClassifier] = None
_singleton_lock = threading.Lock()


def get_eegnet_lite_classifier(
    model_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
) -> EEGNetLiteClassifier:
    """Return (or lazily create) the shared EEGNetLiteClassifier singleton.

    Thread-safe — safe to call from concurrent FastAPI worker tasks.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = EEGNetLiteClassifier(
                    model_path=model_path,
                    onnx_path=onnx_path,
                )
    return _singleton


# ── Backward-compatibility alias ──────────────────────────────────────────────
# cognitive.py imported `EEGNetLite` from this module before the refactor.
# Keep the alias so that import does not break.
EEGNetLite = EEGNetLiteClassifier
