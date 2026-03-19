"""DFF-Net domain adaptation API for cross-subject EEG emotion (#400).

Endpoints:
  POST /dff-net/config   -- create architecture config
  POST /dff-net/evaluate -- evaluate domain discrepancy
  GET  /dff-net/status   -- pipeline status
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.dff_net import (
    DFFNetConfig,
    create_dff_net_config,
    compute_domain_discrepancy,
    compute_mmd,
    setup_few_shot_adaptation,
    evaluate_cross_subject,
    config_to_dict,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/dff-net", tags=["dff-net"])

# -- In-memory state ----------------------------------------------------------

_current_config: Optional[DFFNetConfig] = None
_last_evaluation: Optional[Dict] = None

# -- Pydantic schemas ----------------------------------------------------------


class DFFNetConfigRequest(BaseModel):
    n_channels: int = Field(4, ge=1, le=128)
    n_classes: int = Field(6, ge=2, le=20)
    adaptation_method: str = Field("mmd", pattern="^(mmd|adversarial)$")
    k_shot: int = Field(5, ge=1, le=100)
    mmd_bandwidth: float = Field(1.0, gt=0)
    lambda_domain: float = Field(0.1, ge=0, le=1)
    learning_rate: float = Field(0.001, gt=0)


class DFFNetEvaluateRequest(BaseModel):
    source_features: List[List[float]] = Field(
        ..., description="Source domain features (n_source x d)"
    )
    target_features: List[List[float]] = Field(
        ..., description="Target domain features (n_target x d)"
    )
    source_labels: Optional[List[int]] = Field(
        None, description="Source labels (for few-shot eval)"
    )
    target_labels: Optional[List[int]] = Field(
        None, description="Target labels (for few-shot eval)"
    )


# -- Endpoints -----------------------------------------------------------------


@router.post("/config")
async def create_config(req: DFFNetConfigRequest):
    """Create a DFF-Net architecture configuration."""
    global _current_config
    _current_config = create_dff_net_config(
        n_channels=req.n_channels,
        n_classes=req.n_classes,
        adaptation_method=req.adaptation_method,
        k_shot=req.k_shot,
        mmd_bandwidth=req.mmd_bandwidth,
        lambda_domain=req.lambda_domain,
        learning_rate=req.learning_rate,
    )
    return {
        "status": "ok",
        "config": config_to_dict(_current_config),
        "created_at": time.time(),
    }


@router.post("/evaluate")
async def evaluate_discrepancy(req: DFFNetEvaluateRequest):
    """Evaluate domain discrepancy between source and target features.

    If labels are provided, also runs few-shot adaptation evaluation.
    """
    global _last_evaluation

    source = np.array(req.source_features, dtype=np.float64)
    target = np.array(req.target_features, dtype=np.float64)

    if source.ndim != 2 or target.ndim != 2:
        raise HTTPException(422, "Features must be 2D matrices (n_samples x d)")
    if source.shape[1] != target.shape[1]:
        raise HTTPException(
            422,
            f"Feature dimensions must match: source={source.shape[1]}, target={target.shape[1]}",
        )

    config = _current_config or DFFNetConfig()

    has_labels = req.source_labels is not None and req.target_labels is not None

    if has_labels:
        s_labels = np.array(req.source_labels, dtype=int)
        t_labels = np.array(req.target_labels, dtype=int)
        if len(s_labels) != source.shape[0]:
            raise HTTPException(
                422,
                f"source_labels length ({len(s_labels)}) != source_features rows ({source.shape[0]})",
            )
        if len(t_labels) != target.shape[0]:
            raise HTTPException(
                422,
                f"target_labels length ({len(t_labels)}) != target_features rows ({target.shape[0]})",
            )
        result = evaluate_cross_subject(source, s_labels, target, t_labels, config)
    else:
        da_result = compute_domain_discrepancy(source, target, config)
        from dataclasses import asdict
        result = {
            "domain_adaptation": asdict(da_result),
            "few_shot": None,
            "summary": {
                "mmd_score": da_result.mmd_score,
                "alignment_score": da_result.alignment_score,
                "n_source": da_result.n_source_samples,
                "n_target": da_result.n_target_samples,
            },
        }

    _last_evaluation = result
    return result


@router.get("/status")
async def get_status():
    """Get current DFF-Net pipeline status."""
    return {
        "configured": _current_config is not None,
        "config": config_to_dict(_current_config) if _current_config else None,
        "last_evaluation": _last_evaluation is not None,
        "last_evaluation_summary": (
            _last_evaluation.get("summary") if _last_evaluation else None
        ),
    }
