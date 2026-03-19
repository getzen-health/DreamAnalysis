"""DSP-MCF dual-stream pre-training API (#409).

Exposes endpoints for:
  - Creating DSP-MCF architecture configurations
  - Computing dual-stream features (spatial + temporal + fused)
  - Checking pipeline status

Routes
------
  POST /dsp-mcf/config    — create architecture config
  POST /dsp-mcf/features  — compute dual-stream features
  GET  /dsp-mcf/status    — pipeline status
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.dsp_mcf import (
    DSPMCFConfig,
    compute_spatial_features,
    compute_temporal_features,
    config_to_dict,
    create_dsp_mcf_config,
    fuse_dual_stream,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dsp-mcf", tags=["dsp-mcf"])

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_current_config: Optional[DSPMCFConfig] = None
_feature_computation_count: int = 0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DSPMCFConfigRequest(BaseModel):
    n_channels: int = Field(default=4, ge=1)
    n_samples: int = Field(default=1024, ge=1)
    fs: float = Field(default=256.0, gt=0)
    spatial_feature_dim: int = Field(default=64, ge=1)
    temporal_feature_dim: int = Field(default=64, ge=1)
    fused_dim: int = Field(default=128, ge=1)
    window_sizes: Optional[List[int]] = Field(
        default=None,
        description="STFT window sizes in samples. Defaults to [64, 128, 256].",
    )
    mask_fraction: float = Field(default=0.15, gt=0, lt=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    n_epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=64, ge=1)


class DSPMCFFeaturesRequest(BaseModel):
    signals: List[List[float]] = Field(
        ...,
        description="EEG signals: list of channels, each a list of samples.",
    )
    fs: float = Field(default=256.0, gt=0, description="Sampling frequency in Hz.")
    user_id: str = Field(default="default")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/config")
async def create_config(req: DSPMCFConfigRequest):
    """Create a DSP-MCF architecture configuration.

    Validates parameters and stores the config for subsequent feature
    computation and training operations.
    """
    global _current_config

    wsizes: Optional[Tuple[int, ...]] = None
    if req.window_sizes is not None:
        wsizes = tuple(req.window_sizes)

    try:
        config = create_dsp_mcf_config(
            n_channels=req.n_channels,
            n_samples=req.n_samples,
            fs=req.fs,
            spatial_feature_dim=req.spatial_feature_dim,
            temporal_feature_dim=req.temporal_feature_dim,
            fused_dim=req.fused_dim,
            window_sizes=wsizes,
            mask_fraction=req.mask_fraction,
            learning_rate=req.learning_rate,
            n_epochs=req.n_epochs,
            batch_size=req.batch_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _current_config = config
    return {
        "status": "created",
        "config": config_to_dict(config),
    }


@router.post("/features")
async def compute_features(req: DSPMCFFeaturesRequest):
    """Compute dual-stream features from raw EEG signals.

    Extracts spatial features (inter-channel correlation), temporal features
    (multi-scale STFT band powers), and produces an attention-weighted fused
    representation.
    """
    global _feature_computation_count

    if not req.signals:
        raise HTTPException(status_code=400, detail="signals must not be empty")

    signals = np.array(req.signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    # Compute dual-stream features
    spatial = compute_spatial_features(signals, _current_config)
    temporal = compute_temporal_features(signals, fs=req.fs)
    fused = fuse_dual_stream(spatial, temporal)

    _feature_computation_count += 1

    return {
        "user_id": req.user_id,
        "spatial_features": spatial.tolist(),
        "spatial_dim": len(spatial),
        "temporal_features": temporal.tolist(),
        "temporal_dim": len(temporal),
        "fused_features": fused.tolist(),
        "fused_dim": len(fused),
        "n_channels": signals.shape[0],
        "n_samples": signals.shape[1],
        "feature_computation_count": _feature_computation_count,
        "processed_at": time.time(),
    }


@router.get("/status")
async def dsp_mcf_status():
    """Return DSP-MCF pipeline status."""
    return {
        "pipeline": "dsp-mcf",
        "config_loaded": _current_config is not None,
        "config": config_to_dict(_current_config) if _current_config else None,
        "feature_computation_count": _feature_computation_count,
        "status": "ready",
    }
