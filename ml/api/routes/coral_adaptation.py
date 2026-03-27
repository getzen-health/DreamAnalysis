"""CORAL domain adaptation API.

Full-covariance CORAL alignment that closes the research-to-consumer EEG
distribution gap by whitening source covariance and re-coloring with target.

Endpoints:
  POST /domain-adaptation/coral/fit      -- fit adapter from source + target matrices
  GET  /domain-adaptation/coral/status   -- check if a fitted adapter is available

GitHub issue: #541
"""

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ._shared import _numpy_safe
from processing.domain_adaptation import CORALAdapter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["domain-adaptation"])

# Module-level singleton -- persists across requests within the same process.
_coral_adapter: Optional[CORALAdapter] = None


class CORALFitInput(BaseModel):
    source_features: List[List[float]] = Field(
        ...,
        description=(
            "Feature matrix from research dataset (e.g. DEAP): "
            "[[f1, f2, ...], ...] — one row per sample."
        ),
    )
    target_features: List[List[float]] = Field(
        ...,
        description=(
            "Feature matrix from consumer device (e.g. Muse 2): "
            "[[f1, f2, ...], ...] — one row per sample."
        ),
    )
    reg: float = Field(
        1e-5,
        description="Regularization term added to covariance diagonal for stability.",
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/domain-adaptation/coral/fit")
async def fit_coral(data: CORALFitInput):
    """Fit CORAL adapter from uploaded source + target feature matrices.

    Both matrices must have the same number of features (columns).  The adapter
    is stored in-memory and can be queried via the /status endpoint.

    Returns n_features, n_source_samples, n_target_samples on success.
    """
    global _coral_adapter

    source = np.array(data.source_features, dtype=np.float64)
    target = np.array(data.target_features, dtype=np.float64)

    try:
        adapter = CORALAdapter(reg=data.reg)
        adapter.fit(source, target)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    _coral_adapter = adapter
    logger.info(
        "CORAL adapter fitted: %d source, %d target, %d features",
        source.shape[0],
        target.shape[0],
        source.shape[1],
    )

    return {
        "status": "ok",
        "n_features": source.shape[1],
        "n_source_samples": source.shape[0],
        "n_target_samples": target.shape[0],
    }


@router.get("/domain-adaptation/coral/status")
async def coral_status():
    """Check if a fitted CORAL adapter is available.

    Returns fitted (bool), and if fitted: n_features and reg.
    """
    if _coral_adapter is None or not _coral_adapter.fitted:
        return {"fitted": False}

    return _numpy_safe(
        {
            "fitted": True,
            "n_features": int(len(_coral_adapter.source_mean)),
            "reg": _coral_adapter.reg,
        }
    )
