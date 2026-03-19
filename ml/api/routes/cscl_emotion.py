"""CSCL contrastive learning API for cross-subject EEG emotion (#401).

Endpoints:
  POST /cscl-emotion/config   -- create CSCL training config
  POST /cscl-emotion/evaluate -- evaluate representation quality
  GET  /cscl-emotion/status   -- pipeline status
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.cscl_emotion import (
    CSCLConfig,
    create_cscl_config,
    compute_nt_xent_loss,
    evaluate_representation_quality,
    config_to_dict,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/cscl-emotion", tags=["cscl-emotion"])

# -- In-memory state ----------------------------------------------------------

_current_config: Optional[CSCLConfig] = None
_last_evaluation: Optional[Dict] = None

# -- Pydantic schemas ----------------------------------------------------------


class CSCLConfigRequest(BaseModel):
    n_channels: int = Field(4, ge=1, le=128)
    temperature: float = Field(0.07, gt=0, le=1.0)
    projection_dim: int = Field(32, ge=2, le=512)
    n_classes: int = Field(6, ge=2, le=20)
    aug_temporal_crop: float = Field(0.8, ge=0, le=1)
    aug_amplitude_jitter: float = Field(0.8, ge=0, le=1)
    aug_channel_dropout: float = Field(0.25, ge=0, le=1)
    aug_frequency_perturb: float = Field(0.5, ge=0, le=1)
    learning_rate: float = Field(0.001, gt=0)


class CSCLEvaluateRequest(BaseModel):
    features: List[List[float]] = Field(
        ..., description="Feature/embedding matrix (n_samples x d)"
    )
    labels: List[int] = Field(
        ..., description="Integer class labels (n_samples,)"
    )
    subject_ids: Optional[List[str]] = Field(
        None, description="Subject identifiers for cross-subject metric"
    )


# -- Endpoints -----------------------------------------------------------------


@router.post("/config")
async def create_config(req: CSCLConfigRequest):
    """Create a CSCL training configuration."""
    global _current_config
    _current_config = create_cscl_config(
        n_channels=req.n_channels,
        temperature=req.temperature,
        projection_dim=req.projection_dim,
        n_classes=req.n_classes,
        aug_temporal_crop=req.aug_temporal_crop,
        aug_amplitude_jitter=req.aug_amplitude_jitter,
        aug_channel_dropout=req.aug_channel_dropout,
        aug_frequency_perturb=req.aug_frequency_perturb,
        learning_rate=req.learning_rate,
    )
    return {
        "status": "ok",
        "config": config_to_dict(_current_config),
        "created_at": time.time(),
    }


@router.post("/evaluate")
async def evaluate_representations(req: CSCLEvaluateRequest):
    """Evaluate representation quality of learned embeddings."""
    global _last_evaluation

    features = np.array(req.features, dtype=np.float64)
    labels = np.array(req.labels, dtype=int)

    if features.ndim != 2:
        raise HTTPException(422, "Features must be a 2D matrix (n_samples x d)")
    if len(labels) != features.shape[0]:
        raise HTTPException(
            422,
            f"labels length ({len(labels)}) != features rows ({features.shape[0]})",
        )

    subject_ids = None
    if req.subject_ids is not None:
        if len(req.subject_ids) != features.shape[0]:
            raise HTTPException(
                422,
                f"subject_ids length ({len(req.subject_ids)}) != features rows ({features.shape[0]})",
            )
        subject_ids = np.array(req.subject_ids)

    result = evaluate_representation_quality(features, labels, subject_ids)

    from dataclasses import asdict
    result_dict = asdict(result)
    _last_evaluation = result_dict
    return result_dict


@router.get("/status")
async def get_status():
    """Get current CSCL pipeline status."""
    return {
        "configured": _current_config is not None,
        "config": config_to_dict(_current_config) if _current_config else None,
        "last_evaluation": _last_evaluation,
    }
