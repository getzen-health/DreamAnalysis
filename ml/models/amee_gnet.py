"""AMEEGNet: Attention-enhanced Multi-scale EEGNet architecture specification.

Defines the AMEEGNet architecture as a configuration + inference wrapper,
NOT a PyTorch nn.Module directly. This enables architecture exploration,
benchmark comparison, and config-driven model instantiation without
requiring PyTorch at import time.

Architecture concept:
    Parallel EEGNet blocks with different temporal kernel sizes (multi-scale)
    capture features at different time resolutions. An efficient channel
    attention mechanism (SE-like squeeze-excitation) re-weights the fused
    multi-scale features before final classification.

    Input: (batch, 1, n_channels, n_samples)

    Branch 1 (fine):   EEGNet block with temporal kernel = 32  (125 ms @ 256 Hz)
    Branch 2 (medium): EEGNet block with temporal kernel = 64  (250 ms)
    Branch 3 (coarse): EEGNet block with temporal kernel = 128 (500 ms)

    Fusion: concatenate branch outputs along feature dimension
    Attention: SE block (squeeze → FC → ReLU → FC → sigmoid → scale)
    Classifier: Linear → n_classes

Design goals:
    - Architecture config is serializable (JSON/dict)
    - Benchmark comparison against standard EEGNet
    - Attention weights are inspectable for interpretability
    - No PyTorch dependency at import time

Reference:
    Lawhern et al., "EEGNet" (2018) -- base architecture
    Hu et al., "Squeeze-and-Excitation Networks" (2018) -- SE attention
    Ding et al., "TSception" (2022) -- multi-scale temporal convolutions
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Default architecture constants ───────────────────────────────────────────

DEFAULT_N_CHANNELS: int = 4
DEFAULT_N_CLASSES: int = 3
DEFAULT_TEMPORAL_KERNELS: List[int] = [32, 64, 128]
DEFAULT_ATTENTION_REDUCTION_RATIO: int = 4
DEFAULT_DROPOUT: float = 0.25
DEFAULT_FILTERS_PER_BRANCH: int = 8
DEFAULT_DEPTH_MULTIPLIER: int = 2
DEFAULT_POOL_1: int = 4
DEFAULT_POOL_2: int = 8
DEFAULT_N_SAMPLES: int = 1024

EMOTION_CLASSES: List[str] = ["positive", "neutral", "negative"]


# ── Architecture config ──────────────────────────────────────────────────────


def create_ameegnet_config(
    n_channels: int = DEFAULT_N_CHANNELS,
    n_classes: int = DEFAULT_N_CLASSES,
    temporal_kernels: Optional[List[int]] = None,
    attention_reduction_ratio: int = DEFAULT_ATTENTION_REDUCTION_RATIO,
    dropout: float = DEFAULT_DROPOUT,
    filters_per_branch: int = DEFAULT_FILTERS_PER_BRANCH,
    depth_multiplier: int = DEFAULT_DEPTH_MULTIPLIER,
    pool_1: int = DEFAULT_POOL_1,
    pool_2: int = DEFAULT_POOL_2,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> Dict[str, Any]:
    """Create an AMEEGNet architecture configuration.

    The config fully specifies the architecture and can be serialized to JSON
    for experiment tracking or used to instantiate a PyTorch model.

    Args:
        n_channels:               Number of EEG channels.
        n_classes:                Number of output classes.
        temporal_kernels:         List of temporal kernel sizes for each branch.
        attention_reduction_ratio: SE block channel reduction ratio.
        dropout:                  Dropout probability.
        filters_per_branch:       Number of temporal filters per branch.
        depth_multiplier:         Depthwise conv multiplier (spatial filters).
        pool_1:                   First average pooling kernel size.
        pool_2:                   Second average pooling kernel size.
        n_samples:                Expected input time samples.

    Returns:
        dict with full architecture specification.
    """
    if temporal_kernels is None:
        temporal_kernels = list(DEFAULT_TEMPORAL_KERNELS)

    n_branches = len(temporal_kernels)
    spatial_filters = filters_per_branch * depth_multiplier

    # Compute feature dimension per branch after pooling
    # After Block 1 pooling: n_samples / pool_1
    # After Block 2 pooling: (n_samples / pool_1) / pool_2
    time_after_pool = n_samples // pool_1 // pool_2
    features_per_branch = spatial_filters * time_after_pool
    total_fused_features = features_per_branch * n_branches

    # SE attention intermediate dimension
    se_intermediate = max(1, total_fused_features // attention_reduction_ratio)

    # Parameter count estimates
    params_per_branch = _estimate_branch_params(
        n_channels, filters_per_branch, depth_multiplier, temporal_kernels[0]
    )
    total_branch_params = sum(
        _estimate_branch_params(
            n_channels, filters_per_branch, depth_multiplier, k
        )
        for k in temporal_kernels
    )
    se_params = (
        total_fused_features * se_intermediate
        + se_intermediate
        + se_intermediate * total_fused_features
        + total_fused_features
    )
    classifier_params = total_fused_features * n_classes + n_classes
    total_params = total_branch_params + se_params + classifier_params

    config = {
        "architecture": "AMEEGNet",
        "version": "1.0",
        "n_channels": n_channels,
        "n_classes": n_classes,
        "n_samples": n_samples,
        "temporal_kernels": temporal_kernels,
        "n_branches": n_branches,
        "filters_per_branch": filters_per_branch,
        "depth_multiplier": depth_multiplier,
        "spatial_filters": spatial_filters,
        "pool_1": pool_1,
        "pool_2": pool_2,
        "dropout": dropout,
        "attention": {
            "type": "squeeze-excitation",
            "reduction_ratio": attention_reduction_ratio,
            "input_dim": total_fused_features,
            "intermediate_dim": se_intermediate,
        },
        "classifier": {
            "input_dim": total_fused_features,
            "output_dim": n_classes,
        },
        "estimated_params": {
            "total": total_params,
            "branches": total_branch_params,
            "attention": se_params,
            "classifier": classifier_params,
        },
        "time_after_pool": time_after_pool,
        "features_per_branch": features_per_branch,
        "total_fused_features": total_fused_features,
        "emotion_classes": EMOTION_CLASSES[:n_classes],
        "description": (
            f"AMEEGNet: {n_branches}-branch multi-scale EEGNet "
            f"with SE attention. Temporal kernels: {temporal_kernels}. "
            f"~{total_params} params."
        ),
    }

    return config


# ── Attention computation ────────────────────────────────────────────────────


def compute_attention_weights(
    features: np.ndarray,
    reduction_ratio: int = DEFAULT_ATTENTION_REDUCTION_RATIO,
) -> np.ndarray:
    """Compute SE-like channel attention weights from feature vector.

    Simulates the squeeze-excitation forward pass:
        squeeze: global average pooling (already 1-D if features are flat)
        excite:  FC(d -> d/r) -> ReLU -> FC(d/r -> d) -> sigmoid

    Uses random-initialized weights for demonstration / config validation.
    Real inference uses trained weights.

    Args:
        features: (n_features,) or (batch, n_features) feature vector.
        reduction_ratio: SE reduction ratio.

    Returns:
        weights: Same shape as features, values in [0, 1].
    """
    features = np.asarray(features, dtype=np.float64)
    was_1d = features.ndim == 1
    if was_1d:
        features = features[np.newaxis, :]

    _, d = features.shape
    d_reduced = max(1, d // reduction_ratio)

    # Deterministic pseudo-weights for reproducibility
    rng = np.random.RandomState(42)
    w1 = rng.randn(d, d_reduced) * 0.1
    b1 = np.zeros(d_reduced)
    w2 = rng.randn(d_reduced, d) * 0.1
    b2 = np.zeros(d)

    # Squeeze: global average (features already pooled)
    z = features

    # Excite: FC -> ReLU -> FC -> sigmoid
    h = np.maximum(0, z @ w1 + b1)  # ReLU
    s = _sigmoid(h @ w2 + b2)

    if was_1d:
        return s[0]
    return s


def multi_scale_feature_fusion(
    branch_features: List[np.ndarray],
) -> np.ndarray:
    """Fuse features from multiple temporal-scale branches.

    Concatenates branch outputs along the feature dimension. In the full
    model, this is followed by the SE attention block.

    Args:
        branch_features: List of (batch, features_per_branch) arrays.
                         For single sample, each is (features_per_branch,).

    Returns:
        fused: Concatenated features. Shape (batch, total_features) or
               (total_features,) for single sample.
    """
    if not branch_features:
        raise ValueError("branch_features must not be empty")

    arrays = [np.asarray(f, dtype=np.float64) for f in branch_features]

    # Check dimensionality consistency
    ndims = {a.ndim for a in arrays}
    if len(ndims) > 1:
        raise ValueError(
            f"All branch features must have same ndim, got {ndims}"
        )

    ndim = arrays[0].ndim
    if ndim == 1:
        return np.concatenate(arrays, axis=0)
    elif ndim == 2:
        return np.concatenate(arrays, axis=1)
    else:
        raise ValueError(f"Expected 1-D or 2-D arrays, got {ndim}-D")


def compare_architectures(
    ameegnet_config: Optional[Dict[str, Any]] = None,
    eegnet_params: Optional[int] = None,
    eegnet_accuracy: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare AMEEGNet against standard EEGNet architecture.

    Produces a side-by-side comparison for architecture selection.

    Args:
        ameegnet_config: AMEEGNet config dict (from create_ameegnet_config).
                         If None, uses defaults.
        eegnet_params:   Parameter count of baseline EEGNet. If None,
                         estimates from standard Muse 2 config (~2707).
        eegnet_accuracy: Known EEGNet cross-validation accuracy (0-1).
                         If None, uses benchmark value (0.85).

    Returns:
        dict with comparison metrics and recommendation.
    """
    if ameegnet_config is None:
        ameegnet_config = create_ameegnet_config()

    if eegnet_params is None:
        eegnet_params = 2707  # EEGNet-Lite default for Muse 2

    if eegnet_accuracy is None:
        eegnet_accuracy = 0.85  # 85% CV from CLAUDE.md

    ameeg_params = ameegnet_config["estimated_params"]["total"]
    param_ratio = ameeg_params / max(eegnet_params, 1)

    comparison = {
        "eegnet": {
            "architecture": "EEGNet",
            "params": eegnet_params,
            "accuracy": eegnet_accuracy,
            "temporal_scales": 1,
            "has_attention": False,
            "description": "Single-scale depthwise-separable 2D CNN",
        },
        "ameegnet": {
            "architecture": "AMEEGNet",
            "params": ameeg_params,
            "accuracy": None,  # Not yet trained
            "temporal_scales": ameegnet_config["n_branches"],
            "has_attention": True,
            "attention_type": "squeeze-excitation",
            "description": ameegnet_config["description"],
        },
        "comparison": {
            "param_ratio": round(param_ratio, 2),
            "ameegnet_overhead_percent": round((param_ratio - 1.0) * 100, 1),
            "expected_accuracy_gain": "3-7% (from multi-scale + attention, per literature)",
            "trade_off": (
                "AMEEGNet adds multi-scale temporal processing and channel "
                "attention at the cost of ~{:.1f}x more parameters. Expected "
                "accuracy gain of 3-7% based on TSception and SE-Net ablations."
            ).format(param_ratio),
        },
        "recommendation": (
            "AMEEGNet is recommended when accuracy is prioritized over model "
            "size. For edge/browser deployment where size is critical, "
            "EEGNet-Lite remains the better choice."
        ),
    }

    return comparison


def config_to_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert AMEEGNet config to a JSON-serializable dictionary.

    Ensures all values (including numpy types) are Python-native.

    Args:
        config: AMEEGNet configuration dict.

    Returns:
        JSON-serializable dict.
    """
    result: Dict[str, Any] = {}

    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = config_to_dict(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [
                int(v) if isinstance(v, (np.integer,)) else
                float(v) if isinstance(v, (np.floating,)) else v
                for v in value
            ]
        elif isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.floating):
            result[key] = float(value)
        else:
            result[key] = value

    return result


# ── Private helpers ──────────────────────────────────────────────────────────


def _estimate_branch_params(
    n_channels: int,
    filters: int,
    depth_mult: int,
    kernel_size: int,
) -> int:
    """Estimate trainable parameter count for one EEGNet branch.

    Block 1:
        Conv2d(1, filters, (1, kernel_size)) : filters * kernel_size params
        BatchNorm(filters)                   : 2 * filters
        DepthwiseConv2d(filters, filters*dm, (n_ch, 1)) : filters * dm * n_ch
        BatchNorm(filters * dm)              : 2 * filters * dm

    Block 2:
        DepthwiseConv2d(f*dm, f*dm, (1, 16), groups=f*dm) : f*dm * 16
        Conv2d(f*dm, f*dm, (1, 1))           : (f*dm)^2
        BatchNorm(f*dm)                      : 2 * f*dm
    """
    f = filters
    dm = depth_mult
    sf = f * dm  # spatial filters

    # Block 1
    b1_conv = f * kernel_size  # no bias
    b1_bn1 = 2 * f
    b1_dw = f * dm * n_channels
    b1_bn2 = 2 * sf

    # Block 2
    b2_dw = sf * 16  # depthwise kernel (1, 16)
    b2_pw = sf * sf  # pointwise 1x1
    b2_bn = 2 * sf

    return b1_conv + b1_bn1 + b1_dw + b1_bn2 + b2_dw + b2_pw + b2_bn


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
