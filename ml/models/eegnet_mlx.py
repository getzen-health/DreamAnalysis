"""EEGNet implemented in Apple MLX — runs on Metal + Apple Neural Engine.

This file is the MLX counterpart of eegnet.py (PyTorch). The two implementations
are kept cleanly separated — no mixing in the same forward pass.

Requirements:
  pip install mlx>=0.18.0

How to run:
  cd ml
  pip install mlx>=0.18.0
  python -m training.train_eegnet_mlx --synthetic

Architecture notes vs PyTorch version:
  - MLX Conv2d uses NHWC format: (batch, H, W, C) — NOT NCHW
  - __call__ instead of forward
  - No .cuda() / .to(device) — MLX uses Metal automatically
  - BatchNorm normalises over the LAST dimension in MLX

Requires Apple Silicon Mac (M1/M2/M3). Will raise ImportError on Intel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import mlx.nn as nn
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_mlx() -> None:
    if not _MLX_AVAILABLE:
        raise ImportError(
            "MLX is not installed. Run: pip install mlx>=0.18.0\n"
            "MLX requires Apple Silicon (M1/M2/M3)."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _avg_pool_hw(x, pool_h: int, pool_w: int):
    """Manual average pooling over (H, W) in NHWC format.

    x: (batch, H, W, C)
    Returns: (batch, H//pool_h, W//pool_w, C)
    """
    batch, H, W, C = x.shape
    H2 = H // pool_h
    W2 = W // pool_w
    # Reshape and mean — fold pool_h into H axis, pool_w into W axis
    x = x[:, :H2 * pool_h, :W2 * pool_w, :]          # trim to divisible size
    x = x.reshape(batch, H2, pool_h, W2 * pool_w, C)  # (B, H2, ph, W2*pw, C)
    x = x.mean(axis=2)                                  # (B, H2, W2*pw, C)
    x = x.reshape(batch, H2, W2, pool_w, C)
    x = x.mean(axis=3)                                  # (B, H2, W2, C)
    return x


# ---------------------------------------------------------------------------
# Depthwise-separable conv helper (MLX)
# ---------------------------------------------------------------------------

class _DepthwiseSeparableConvMLX(nn.Module if _MLX_AVAILABLE else object):
    """Depthwise conv (groups=in_channels) + pointwise 1×1 conv — NHWC."""

    def __init__(self, in_channels: int, out_channels: int, kernel_w: int):
        if not _MLX_AVAILABLE:
            raise ImportError("MLX required")
        super().__init__()
        # Depthwise: kernel (1, kernel_w), groups=in_channels
        # MLX Conv2d weight shape: (out_channels, kH, kW, in_channels/groups)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(1, kernel_w),
            padding=(0, kernel_w // 2),
            groups=in_channels,
            bias=False,
        )
        # Pointwise: 1×1, in_channels → out_channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def __call__(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ---------------------------------------------------------------------------
# EEGNet (MLX)
# ---------------------------------------------------------------------------

class EEGNetMLX(nn.Module if _MLX_AVAILABLE else object):
    """EEGNet architecture implemented in MLX (NHWC format).

    Input convention (MLX NHWC):
      (batch, n_channels, n_samples, 1)

    This mirrors the PyTorch EEGNet which expects:
      (batch, 1, n_channels, n_samples)  [NCHW]

    Parameters
    ----------
    n_classes   : number of output classes (6 for emotions)
    n_channels  : EEG channels (4 = Muse 2, 8 = Cyton, 16 = Cyton+Daisy)
    n_samples   : time samples per epoch (256 Hz × 4 sec = 1024)
    F1          : temporal filters (default 8)
    D           : depth multiplier for depthwise conv (F2 = F1*D, default 2)
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
        _require_mlx()
        super().__init__()

        F2 = F1 * D
        kernel_length = max(32, n_samples // 16)   # ~250 ms at 256 Hz → 64
        sep_kernel = max(16, n_samples // 64)       # ~62.5 ms at 256 Hz → 16

        # Store config for serialisation
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.F1 = F1
        self.D = D
        self.dropout_rate = dropout_rate
        self._F2 = F2
        self._kernel_length = kernel_length
        self._sep_kernel = sep_kernel

        # ── Block 1: Temporal convolution ─────────────────────────────────
        # Input NHWC: (B, n_channels, n_samples, 1)
        # temporal_conv: (1, kernel_length) kernel — operates across time (W axis)
        # Output: (B, n_channels, n_samples, F1)
        self.temporal_conv = nn.Conv2d(
            1, F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm(F1)   # normalises over last dim (C=F1)

        # Depthwise: (n_channels, 1) kernel collapses spatial (H) dim
        # Input: (B, n_channels, n_samples, F1)
        # Output: (B, 1, n_samples, F2)
        self.depthwise_conv = nn.Conv2d(
            F1, F2,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm(F2)
        # pool1: AvgPool (1, 4) → (B, 1, n_samples//4, F2)
        self._pool1_h = 1
        self._pool1_w = 4
        self.drop1 = nn.Dropout(p=dropout_rate)

        # ── Block 2: Separable temporal ───────────────────────────────────
        # Input: (B, 1, n_samples//4, F2)
        self.separable_conv = _DepthwiseSeparableConvMLX(F2, F2, kernel_w=sep_kernel)
        self.bn3 = nn.BatchNorm(F2)
        # pool2: AvgPool (1, 8) → (B, 1, n_samples//32, F2)
        self._pool2_h = 1
        self._pool2_w = 8
        self.drop2 = nn.Dropout(p=dropout_rate)

        # ── Classifier ────────────────────────────────────────────────────
        # Flat size: 1 × (n_samples//32) × F2
        flat_size = 1 * (n_samples // 32) * F2
        self.classifier = nn.Linear(flat_size, n_classes)

    # ── Forward pass ──────────────────────────────────────────────────────

    def __call__(self, x, training: bool = False):
        """Forward pass.

        Parameters
        ----------
        x        : MLX array, shape (batch, n_channels, n_samples)
                   OR (batch, n_channels, n_samples, 1)
        training : whether to apply dropout

        Returns
        -------
        logits   : (batch, n_classes)
        """
        # Ensure 4D NHWC: (B, n_channels, n_samples, 1)
        if x.ndim == 3:
            x = x[:, :, :, None]   # add channel dim at end

        # Block 1 — Temporal conv
        x = self.temporal_conv(x)              # (B, n_channels, n_samples, F1)
        x = self.bn1(x)
        x = self.depthwise_conv(x)             # (B, 1, n_samples, F2)
        x = self.bn2(x)
        x = nn.elu(x)
        x = _avg_pool_hw(x, self._pool1_h, self._pool1_w)   # (B, 1, n_samples//4, F2)
        x = self.drop1(x)

        # Block 2 — Separable conv
        x = self.separable_conv(x)             # (B, 1, n_samples//4, F2)
        x = self.bn3(x)
        x = nn.elu(x)
        x = _avg_pool_hw(x, self._pool2_h, self._pool2_w)   # (B, 1, n_samples//32, F2)
        x = self.drop2(x)

        # Flatten and classify
        batch = x.shape[0]
        x = x.reshape(batch, -1)               # (B, flat_size)
        return self.classifier(x)

    # ── Serialisation ─────────────────────────────────────────────────────

    def save_weights(self, path: str | Path) -> None:
        """Save model weights to an .npz file using mx.savez."""
        _require_mlx()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = dict(self.parameters())
        # Flatten nested dict to dot-separated keys
        flat = {}
        _flatten_params(params, "", flat)
        # Save config alongside weights
        flat["__n_classes__"] = mx.array([self.n_classes])
        flat["__n_channels__"] = mx.array([self.n_channels])
        flat["__n_samples__"] = mx.array([self.n_samples])
        flat["__F1__"] = mx.array([self.F1])
        flat["__D__"] = mx.array([self.D])
        flat["__dropout_rate__"] = mx.array([self.dropout_rate])
        mx.savez(str(path), **flat)
        log.info("EEGNetMLX weights saved → %s", path)

    def load_weights(self, path: str | Path) -> None:
        """Load weights from an .npz file saved with save_weights."""
        _require_mlx()
        path = Path(path)
        flat = dict(mx.load(str(path)))
        # Remove config keys
        for k in list(flat.keys()):
            if k.startswith("__"):
                flat.pop(k)
        # Rebuild nested dict and update model parameters
        nested = _unflatten_params(flat)
        self.update(nested)
        log.info("EEGNetMLX weights loaded ← %s", path)

    @classmethod
    def from_weights(cls, path: str | Path) -> "EEGNetMLX":
        """Construct a new EEGNetMLX from a saved .npz file (reads config from file)."""
        _require_mlx()
        path = Path(path)
        flat = dict(mx.load(str(path)))
        n_classes = int(flat["__n_classes__"].item())
        n_channels = int(flat["__n_channels__"].item())
        n_samples = int(flat["__n_samples__"].item())
        F1 = int(flat["__F1__"].item())
        D = int(flat["__D__"].item())
        dropout_rate = float(flat["__dropout_rate__"].item())
        model = cls(
            n_classes=n_classes,
            n_channels=n_channels,
            n_samples=n_samples,
            F1=F1,
            D=D,
            dropout_rate=dropout_rate,
        )
        model.load_weights(path)
        return model

    def to_pytorch_state_dict(self) -> dict:
        """Export weights as a dict of numpy arrays for loading into PyTorch EEGNet.

        The returned dict maps PyTorch state_dict key names to numpy arrays so
        that the PyTorch model can be loaded and exported to ONNX without
        retraining on non-MLX machines.

        Note: Weight tensor dimension ordering differs between MLX and PyTorch:
          - MLX Conv2d weight: (out_C, kH, kW, in_C/groups)
          - PyTorch Conv2d weight: (out_C, in_C/groups, kH, kW)
        """
        _require_mlx()
        params = dict(self.parameters())
        flat = {}
        _flatten_params(params, "", flat)
        numpy_flat = {k: np.array(v) for k, v in flat.items()}

        def _conv_weight(arr: np.ndarray) -> np.ndarray:
            """Convert MLX Conv weight (out, kH, kW, in) → PyTorch (out, in, kH, kW)."""
            return arr.transpose(0, 3, 1, 2)

        def _bn_to_pt(flat_params: dict, mlx_prefix: str, pt_prefix: str, out: dict) -> None:
            """Map MLX BatchNorm → PyTorch BatchNorm2d keys."""
            # MLX BatchNorm has: weight (gamma), bias (beta), running_mean, running_var
            for mlx_k, pt_k in [
                ("weight", "weight"),
                ("bias", "bias"),
                ("running_mean", "running_mean"),
                ("running_var", "running_var"),
            ]:
                src = mlx_prefix + mlx_k
                if src in flat_params:
                    out[pt_prefix + pt_k] = flat_params[src]

        sd: dict = {}

        # temporal_conv.weight
        key = "temporal_conv.weight"
        if key in numpy_flat:
            sd["temporal_conv.weight"] = _conv_weight(numpy_flat[key])

        # bn1
        _bn_to_pt(numpy_flat, "bn1.", "bn1.", sd)

        # depthwise_conv.weight
        key = "depthwise_conv.weight"
        if key in numpy_flat:
            sd["depthwise_conv.weight"] = _conv_weight(numpy_flat[key])

        # bn2
        _bn_to_pt(numpy_flat, "bn2.", "bn2.", sd)

        # separable_conv.depthwise.weight / pointwise.weight
        key = "separable_conv.depthwise.weight"
        if key in numpy_flat:
            sd["separable_conv.depthwise.weight"] = _conv_weight(numpy_flat[key])
        key = "separable_conv.pointwise.weight"
        if key in numpy_flat:
            sd["separable_conv.pointwise.weight"] = _conv_weight(numpy_flat[key])

        # bn3
        _bn_to_pt(numpy_flat, "bn3.", "bn3.", sd)

        # classifier
        if "classifier.weight" in numpy_flat:
            sd["classifier.weight"] = numpy_flat["classifier.weight"]
        if "classifier.bias" in numpy_flat:
            sd["classifier.bias"] = numpy_flat["classifier.bias"]

        return sd


# ---------------------------------------------------------------------------
# Parameter dict utilities
# ---------------------------------------------------------------------------

def _flatten_params(d, prefix: str, out: dict) -> None:
    """Recursively flatten nested parameter dict to dot-separated keys."""
    for k, v in d.items():
        full_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            _flatten_params(v, full_key, out)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    _flatten_params(item, f"{full_key}.{i}", out)
                else:
                    out[f"{full_key}.{i}"] = item
        else:
            out[full_key] = v


def _unflatten_params(flat: dict) -> dict:
    """Reconstruct nested dict from dot-separated keys (inverse of _flatten_params)."""
    nested: dict = {}
    for key, val in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = val
    return nested


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_eegnet_mlx(
    n_classes: int = 6,
    n_channels: int = 4,
    n_samples: int = 1024,
) -> "EEGNetMLX":
    """Build an EEGNetMLX with default hyperparameters for the given channel count."""
    _require_mlx()
    model = EEGNetMLX(
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
    )
    log.info(
        "EEGNetMLX built — %d-ch, %d samples/epoch, device: %s",
        n_channels, n_samples, mx.default_device(),
    )
    return model
