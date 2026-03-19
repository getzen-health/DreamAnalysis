"""AMEEGNet attention-enhanced EEGNet architecture endpoint.

Exposes the AMEEGNet architecture specification and benchmark comparison
as a FastAPI sub-router mounted at /ameegnet.

Endpoints:
    POST /ameegnet/config — generate AMEEGNet architecture configuration
    GET  /ameegnet/status — architecture info and comparison with EEGNet
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/ameegnet", tags=["ameegnet"])

# ── Lazy module loading ──────────────────────────────────────────────────────

_module = None


def _get_module():
    global _module
    if _module is None:
        try:
            from models import amee_gnet

            _module = amee_gnet
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"AMEEGNet module unavailable: {exc}",
            )
    return _module


# ── Schemas ──────────────────────────────────────────────────────────────────


class AMEEGNetConfigRequest(BaseModel):
    """Request to generate an AMEEGNet architecture configuration.

    Attributes:
        n_channels:               Number of EEG channels. Default 4 (Muse 2).
        n_classes:                Number of output classes. Default 3.
        temporal_kernels:         List of temporal kernel sizes per branch.
                                  Default [32, 64, 128].
        attention_reduction_ratio: SE block reduction ratio. Default 4.
        dropout:                  Dropout probability. Default 0.25.
        include_comparison:       Whether to include EEGNet comparison. Default True.
    """

    n_channels: int = Field(4, ge=1, le=256, description="Number of EEG channels.")
    n_classes: int = Field(3, ge=2, le=10, description="Number of output classes.")
    temporal_kernels: Optional[List[int]] = Field(
        None, description="Temporal kernel sizes per branch. Default [32, 64, 128]."
    )
    attention_reduction_ratio: int = Field(
        4, ge=1, le=16, description="SE attention reduction ratio."
    )
    dropout: float = Field(0.25, ge=0.0, le=0.9, description="Dropout probability.")
    include_comparison: bool = Field(
        True, description="Include EEGNet benchmark comparison."
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/config")
async def ameegnet_config(req: AMEEGNetConfigRequest):
    """Generate an AMEEGNet architecture configuration.

    Returns a complete architecture specification including layer dimensions,
    estimated parameter counts, and optionally a side-by-side comparison
    with standard EEGNet.

    **AMEEGNet** extends EEGNet with:
    - Multi-scale temporal convolutions (multiple parallel EEGNet branches
      with different kernel sizes)
    - Squeeze-Excitation channel attention for adaptive feature re-weighting
    - Feature fusion via concatenation + attention gating

    The architecture config is JSON-serializable and can be used to
    instantiate the model in a training script.
    """
    mod = _get_module()

    config = mod.create_ameegnet_config(
        n_channels=req.n_channels,
        n_classes=req.n_classes,
        temporal_kernels=req.temporal_kernels,
        attention_reduction_ratio=req.attention_reduction_ratio,
        dropout=req.dropout,
    )

    # Ensure JSON-serializable
    config_dict = mod.config_to_dict(config)

    result: Dict[str, Any] = {
        "config": config_dict,
        "generated_at": time.time(),
    }

    if req.include_comparison:
        comparison = mod.compare_architectures(ameegnet_config=config)
        result["comparison"] = comparison

    return result


@router.get("/status")
async def ameegnet_status():
    """Return AMEEGNet architecture status and summary.

    Reports the default configuration, estimated parameter count,
    and a comparison with baseline EEGNet.
    """
    mod = _get_module()

    default_config = mod.create_ameegnet_config()
    comparison = mod.compare_architectures()

    return {
        "status": "ok",
        "architecture": "AMEEGNet",
        "version": default_config.get("version", "1.0"),
        "default_config": {
            "n_channels": default_config["n_channels"],
            "n_classes": default_config["n_classes"],
            "temporal_kernels": default_config["temporal_kernels"],
            "n_branches": default_config["n_branches"],
            "attention_type": default_config["attention"]["type"],
            "estimated_total_params": default_config["estimated_params"]["total"],
        },
        "description": default_config["description"],
        "comparison_with_eegnet": comparison["comparison"],
        "recommendation": comparison["recommendation"],
    }
