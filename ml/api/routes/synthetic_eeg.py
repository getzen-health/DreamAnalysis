"""Synthetic EEG generation and augmentation API (#445).

Endpoints:
  POST /synthetic-eeg/generate   -- generate synthetic EEG signals
  POST /synthetic-eeg/augment    -- augment existing EEG data
  POST /synthetic-eeg/validate   -- validate quality of synthetic data
  GET  /synthetic-eeg/status     -- service readiness check
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ._shared import _numpy_safe

from models.synthetic_eeg import (
    BANDS,
    generate_synthetic_eeg,
    generate_emotion_conditioned_eeg,
    inject_artifacts,
    augment_eeg,
    validate_synthetic_quality,
    stats_to_dict,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/synthetic-eeg", tags=["synthetic-eeg"])


# ── Pydantic request / response schemas ──────────────────────────────────────

class GenerateRequest(BaseModel):
    duration: float = Field(default=4.0, ge=0.5, le=60.0, description="Duration in seconds")
    fs: float = Field(default=256.0, ge=64.0, le=1024.0, description="Sampling rate (Hz)")
    n_channels: int = Field(default=4, ge=1, le=64, description="Number of EEG channels")
    band_powers: Optional[Dict[str, float]] = Field(
        default=None,
        description="Target relative band powers {delta, theta, alpha, beta, gamma}. "
                    "Values are normalised internally.",
    )
    emotion: Optional[str] = Field(
        default=None,
        description="Generate EEG conditioned on this emotion label "
                    "(happy, sad, angry, fear, neutral, surprise, relaxed, focused). "
                    "If set, band_powers is ignored.",
    )
    amplitude_uv: float = Field(default=20.0, ge=1.0, le=500.0, description="RMS amplitude in uV")
    inject_blinks: float = Field(default=0.0, ge=0.0, le=5.0, description="Eye blink rate (per second)")
    inject_muscle: float = Field(default=0.0, ge=0.0, le=5.0, description="Muscle artifact rate")
    inject_electrode_pop: float = Field(default=0.0, ge=0.0, le=5.0, description="Electrode pop rate")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    validate: bool = Field(default=True, description="Run quality validation on generated signal")


class GenerateResponse(BaseModel):
    signals: List[List[float]]
    n_channels: int
    n_samples: int
    fs: float
    duration: float
    emotion: Optional[str] = None
    band_profile: Optional[Dict[str, float]] = None
    artifacts_injected: int = 0
    validation: Optional[Dict[str, Any]] = None
    generated_at: float


class AugmentRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="Original EEG signals (channels x samples)")
    fs: float = Field(default=256.0, ge=64.0, le=1024.0)
    n_augmentations: int = Field(default=5, ge=1, le=50, description="Number of augmented copies")
    time_shift: bool = Field(default=True)
    amplitude_scale: bool = Field(default=True)
    additive_noise: bool = Field(default=True)
    band_perturbation: bool = Field(default=True)
    seed: Optional[int] = None


class AugmentResponse(BaseModel):
    augmentations: List[Dict[str, Any]]
    n_augmentations: int
    original_shape: List[int]
    fs: float
    generated_at: float


class ValidateRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals to validate (channels x samples)")
    fs: float = Field(default=256.0, ge=64.0, le=1024.0)


class ValidateResponse(BaseModel):
    is_valid: bool
    n_signals: int
    n_passed: int
    n_failed: int
    mean_band_powers: Dict[str, float]
    power_in_range: Dict[str, bool]
    total_power: float
    total_power_valid: bool
    spectral_entropy: float
    failure_reasons: List[str]
    validated_at: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def synthetic_eeg_status():
    """Return service readiness and available emotion profiles."""
    from models.synthetic_eeg import _EMOTION_PROFILES
    return {
        "status": "ready",
        "available_emotions": sorted(_EMOTION_PROFILES.keys()),
        "available_bands": sorted(BANDS.keys()),
        "band_ranges_hz": {k: list(v) for k, v in BANDS.items()},
    }


@router.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    """Generate synthetic EEG with controllable spectral properties.

    Supports two modes:
      1. **Band-power controlled**: provide ``band_powers`` dict with
         target relative powers per band.
      2. **Emotion conditioned**: provide ``emotion`` label and the
         system uses a literature-derived spectral profile.

    Optionally inject realistic artifacts (blinks, muscle, electrode pops)
    and run quality validation on the output.
    """
    try:
        if req.emotion:
            result = generate_emotion_conditioned_eeg(
                emotion=req.emotion,
                duration=req.duration,
                fs=req.fs,
                n_channels=req.n_channels,
                amplitude_uv=req.amplitude_uv,
                seed=req.seed,
            )
            signals = result["signals"]
            emotion_label = result["emotion"]
            band_profile = result["band_profile"]
        else:
            signals = generate_synthetic_eeg(
                duration=req.duration,
                fs=req.fs,
                n_channels=req.n_channels,
                band_powers=req.band_powers,
                amplitude_uv=req.amplitude_uv,
                seed=req.seed,
            )
            emotion_label = None
            band_profile = req.band_powers

        # Inject artifacts if requested
        n_artifacts = 0
        if req.inject_blinks > 0 or req.inject_muscle > 0 or req.inject_electrode_pop > 0:
            art_result = inject_artifacts(
                signals,
                fs=req.fs,
                blink_rate=req.inject_blinks,
                muscle_rate=req.inject_muscle,
                electrode_pop_rate=req.inject_electrode_pop,
                seed=req.seed,
            )
            signals = art_result["signals"]
            n_artifacts = art_result["n_artifacts"]

        # Validation
        validation = None
        if req.validate:
            stats = validate_synthetic_quality(signals, fs=req.fs)
            validation = stats_to_dict(stats)

        return GenerateResponse(
            signals=_numpy_safe(signals.tolist()),
            n_channels=int(signals.shape[0]),
            n_samples=int(signals.shape[1]),
            fs=req.fs,
            duration=req.duration,
            emotion=emotion_label,
            band_profile=_numpy_safe(band_profile),
            artifacts_injected=n_artifacts,
            validation=validation,
            generated_at=time.time(),
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("Synthetic EEG generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")


@router.post("/augment", response_model=AugmentResponse)
async def augment_endpoint(req: AugmentRequest):
    """Augment existing EEG data with random transforms.

    Takes real EEG signals and produces N augmented copies using a
    combination of: time shift, amplitude scaling, additive Gaussian
    noise, and band-power perturbation.
    """
    if not req.signals or not req.signals[0]:
        raise HTTPException(status_code=400, detail="signals must not be empty")

    try:
        signals = np.array(req.signals, dtype=np.float64)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        augmented = augment_eeg(
            signals,
            fs=req.fs,
            n_augmentations=req.n_augmentations,
            time_shift=req.time_shift,
            amplitude_scale=req.amplitude_scale,
            additive_noise=req.additive_noise,
            band_perturbation=req.band_perturbation,
            seed=req.seed,
        )

        # Convert numpy arrays to lists for JSON response
        response_augmentations = []
        for aug in augmented:
            response_augmentations.append({
                "signals": _numpy_safe(aug["signals"].tolist()),
                "transforms": aug["transforms"],
                "augmentation_index": aug["augmentation_index"],
            })

        return AugmentResponse(
            augmentations=response_augmentations,
            n_augmentations=len(response_augmentations),
            original_shape=list(signals.shape),
            fs=req.fs,
            generated_at=time.time(),
        )

    except Exception as exc:
        log.exception("EEG augmentation failed")
        raise HTTPException(status_code=500, detail=f"Augmentation failed: {exc}")


@router.post("/validate", response_model=ValidateResponse)
async def validate_endpoint(req: ValidateRequest):
    """Validate spectral quality of EEG data (real or synthetic).

    Checks that relative band powers fall within physiological ranges,
    total power is realistic, and spectral entropy is reasonable.
    """
    if not req.signals or not req.signals[0]:
        raise HTTPException(status_code=400, detail="signals must not be empty")

    try:
        signals = np.array(req.signals, dtype=np.float64)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        stats = validate_synthetic_quality(signals, fs=req.fs)
        d = stats_to_dict(stats)

        return ValidateResponse(
            is_valid=d["is_valid"],
            n_signals=d["n_signals"],
            n_passed=d["n_passed"],
            n_failed=d["n_failed"],
            mean_band_powers=d["mean_band_powers"],
            power_in_range=d["power_in_range"],
            total_power=d["total_power"],
            total_power_valid=d["total_power_valid"],
            spectral_entropy=d["spectral_entropy"],
            failure_reasons=d["failure_reasons"],
            validated_at=time.time(),
        )

    except Exception as exc:
        log.exception("EEG validation failed")
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}")
