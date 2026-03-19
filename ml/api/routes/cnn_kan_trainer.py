"""CNN-KAN DEAP training pipeline endpoint.

Exposes the CNN-KAN training pipeline configuration (data preprocessing,
cross-validation setup, metric tracking) as a FastAPI sub-router mounted
at /cnn-kan-trainer.

Endpoints:
    POST /cnn-kan-trainer/config — create training pipeline configuration
    GET  /cnn-kan-trainer/status — pipeline status and DEAP dataset info
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/cnn-kan-trainer", tags=["cnn-kan-trainer"])

# ── Lazy module loading ──────────────────────────────────────────────────────

_module = None


def _get_module():
    global _module
    if _module is None:
        try:
            from models import cnn_kan_trainer as mod

            _module = mod
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"CNN-KAN trainer module unavailable: {exc}",
            )
    return _module


# ── Schemas ──────────────────────────────────────────────────────────────────


class TrainerConfigRequest(BaseModel):
    """Request to create a CNN-KAN training pipeline configuration.

    Attributes:
        learning_rate:           Initial learning rate. Default 0.001.
        batch_size:              Training batch size. Default 32.
        epochs:                  Maximum training epochs. Default 100.
        early_stopping_patience: Patience for early stopping. Default 10.
        cv_strategy:             Cross-validation strategy:
                                 "5-fold", "loso", "within-subject". Default "5-fold".
        n_classes:               Number of output classes (2 or 3). Default 3.
        window_seconds:          Epoch window in seconds. Default 4.0.
        overlap:                 Window overlap fraction. Default 0.5.
    """

    learning_rate: float = Field(1e-3, gt=0.0, description="Initial learning rate.")
    batch_size: int = Field(32, ge=1, le=512, description="Training batch size.")
    epochs: int = Field(100, ge=1, le=1000, description="Maximum epochs.")
    early_stopping_patience: int = Field(
        10, ge=1, le=100, description="Early stopping patience."
    )
    cv_strategy: str = Field(
        "5-fold",
        description="CV strategy: '5-fold', 'loso', 'within-subject'.",
    )
    n_classes: int = Field(3, ge=2, le=5, description="Number of output classes.")
    window_seconds: float = Field(4.0, ge=0.5, le=30.0, description="Window length in seconds.")
    overlap: float = Field(0.5, ge=0.0, le=0.9, description="Window overlap fraction.")


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/config")
async def cnn_kan_trainer_config(req: TrainerConfigRequest):
    """Create a CNN-KAN training pipeline configuration for DEAP 4-channel data.

    Generates a complete training config including:
    - Data preprocessing parameters (4-channel extraction, resampling, filtering)
    - Feature extraction pipeline (DE + PSD -> pseudo-RGB)
    - Training hyperparameters (optimizer, scheduler, early stopping)
    - Cross-validation setup (LOSO, 5-fold, or within-subject)
    - Estimated dataset statistics

    The config can be passed to a training script to begin model training.
    """
    mod = _get_module()

    # Validate CV strategy
    valid_strategies = {"5-fold", "loso", "within-subject"}
    if req.cv_strategy not in valid_strategies:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid cv_strategy: {req.cv_strategy!r}. "
                f"Must be one of: {sorted(valid_strategies)}"
            ),
        )

    config = mod.create_training_config(
        learning_rate=req.learning_rate,
        batch_size=req.batch_size,
        epochs=req.epochs,
        early_stopping_patience=req.early_stopping_patience,
        cv_strategy=req.cv_strategy,
        n_classes=req.n_classes,
        window_seconds=req.window_seconds,
        overlap=req.overlap,
    )

    # Also generate the CV fold assignments
    cv_setup = mod.setup_cross_validation(
        n_subjects=mod.DEAP_N_SUBJECTS,
        strategy=req.cv_strategy,
    )

    return {
        "config": config,
        "cross_validation": {
            "strategy": cv_setup["strategy"],
            "n_folds": cv_setup["n_folds"],
            "n_subjects": cv_setup["n_subjects"],
        },
        "generated_at": time.time(),
    }


@router.get("/status")
async def cnn_kan_trainer_status():
    """Return CNN-KAN training pipeline status and DEAP dataset information.

    Reports DEAP dataset parameters, channel mapping to Muse 2 positions,
    supported cross-validation strategies, and frequency band configuration.
    """
    mod = _get_module()

    return {
        "status": "ok",
        "pipeline": "CNN-KAN DEAP 4-channel training",
        "dataset": {
            "name": "DEAP",
            "n_subjects": mod.DEAP_N_SUBJECTS,
            "n_trials_per_subject": mod.DEAP_N_TRIALS,
            "original_fs": mod.DEAP_ORIGINAL_FS,
            "original_channels": mod.DEAP_N_CHANNELS,
        },
        "target": {
            "n_channels": mod.TARGET_N_CHANNELS,
            "fs": mod.TARGET_FS,
            "window_samples": mod.TARGET_WINDOW_SAMPLES,
            "channel_names": mod.CHANNEL_NAMES,
            "emotion_classes": mod.EMOTION_CLASSES,
        },
        "channel_mapping": mod.DEAP_CHANNEL_MAP,
        "frequency_bands": mod.FREQUENCY_BANDS,
        "cv_strategies": ["5-fold", "loso", "within-subject"],
    }
