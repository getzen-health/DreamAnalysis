"""Multi-dataset joint pre-training (mdJPT) API (#408).

Exposes endpoints for:
  - Creating pre-training configurations
  - Harmonizing dataset features to a common space
  - Checking pipeline status

Routes
------
  POST /mdjpt/config     — create pre-training config
  POST /mdjpt/harmonize  — harmonize dataset features
  GET  /mdjpt/status     — pipeline status
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.mdjpt_pretraining import (
    SAMPLING_STRATEGIES,
    SUPPORTED_DATASETS,
    UNIFIED_LABELS,
    PretrainingConfig,
    align_labels,
    compute_dataset_statistics,
    config_to_dict,
    create_pretraining_config,
    harmonize_datasets,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mdjpt", tags=["mdjpt"])

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_current_config: Optional[PretrainingConfig] = None
_harmonization_count: int = 0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class MdJPTConfigRequest(BaseModel):
    datasets: Optional[List[str]] = Field(
        default=None,
        description="Datasets to include. Defaults to all supported.",
    )
    sampling_strategy: str = Field(
        default="proportional",
        description="Batch sampling strategy: proportional, equal, or curriculum.",
    )
    shared_encoder_dim: int = Field(default=128, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    n_epochs: int = Field(default=100, ge=1)
    warmup_epochs: int = Field(default=5, ge=0)
    freeze_encoder_for_transfer: bool = Field(default=False)


class DatasetFeaturesInput(BaseModel):
    dataset_name: str = Field(
        ..., description="Dataset name (DEAP, SEED, GAMEEMO, DREAMER)."
    )
    features: List[List[float]] = Field(
        ..., description="Feature matrix: list of samples, each a list of floats."
    )


class HarmonizeRequest(BaseModel):
    datasets: List[DatasetFeaturesInput] = Field(
        ..., description="List of dataset feature matrices to harmonize."
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/config")
async def create_config(req: MdJPTConfigRequest):
    """Create a multi-dataset joint pre-training configuration.

    Validates dataset names and sampling strategy, then stores the config
    for subsequent harmonization and training operations.
    """
    global _current_config

    try:
        config = create_pretraining_config(
            datasets=req.datasets,
            sampling_strategy=req.sampling_strategy,
            shared_encoder_dim=req.shared_encoder_dim,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            n_epochs=req.n_epochs,
            warmup_epochs=req.warmup_epochs,
            freeze_encoder_for_transfer=req.freeze_encoder_for_transfer,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _current_config = config
    return {
        "status": "created",
        "config": config_to_dict(config),
    }


@router.post("/harmonize")
async def harmonize(req: HarmonizeRequest):
    """Harmonize dataset features to a common feature space.

    Applies per-dataset z-score normalisation so that features from
    different EEG datasets are comparable.
    """
    global _harmonization_count

    if not req.datasets:
        raise HTTPException(status_code=400, detail="datasets must not be empty")

    dataset_features: Dict[str, np.ndarray] = {}
    for ds in req.datasets:
        if ds.dataset_name not in SUPPORTED_DATASETS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown dataset '{ds.dataset_name}'. "
                       f"Supported: {list(SUPPORTED_DATASETS)}",
            )
        arr = np.array(ds.features, dtype=np.float64)
        if arr.size == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Empty features for dataset '{ds.dataset_name}'",
            )
        dataset_features[ds.dataset_name] = arr

    try:
        harmonised = harmonize_datasets(dataset_features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _harmonization_count += 1

    # Build response: per-dataset statistics of harmonised features
    result: Dict[str, dict] = {}
    for name, arr in harmonised.items():
        stats = compute_dataset_statistics(arr, dataset_name=name)
        result[name] = {
            "n_samples": stats.n_samples,
            "n_features": stats.n_features,
            "feature_means_sample": stats.feature_means[:5] if stats.feature_means else [],
            "feature_stds_sample": stats.feature_stds[:5] if stats.feature_stds else [],
        }

    return {
        "status": "harmonized",
        "n_datasets": len(harmonised),
        "datasets": result,
        "harmonization_count": _harmonization_count,
        "processed_at": time.time(),
    }


@router.get("/status")
async def mdjpt_status():
    """Return mdJPT pipeline status."""
    return {
        "pipeline": "mdjpt",
        "config_loaded": _current_config is not None,
        "config": config_to_dict(_current_config) if _current_config else None,
        "supported_datasets": list(SUPPORTED_DATASETS),
        "sampling_strategies": list(SAMPLING_STRATEGIES),
        "unified_labels": list(UNIFIED_LABELS),
        "harmonization_count": _harmonization_count,
        "status": "ready",
    }
