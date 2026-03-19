"""Voice emotion data augmentation API (#384).

Endpoints:
  POST /voice-augmentation/augment  -- augment a voice sample
  GET  /voice-augmentation/status   -- service readiness check
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/voice-augmentation", tags=["voice-augmentation"])


# ── Pydantic request / response schemas ──────────────────────────────────────

class AugmentRequest(BaseModel):
    audio: List[float] = Field(..., description="1-D audio waveform as a list of floats")
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Sample rate in Hz")
    noise_prob: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability of noise injection")
    pitch_prob: float = Field(default=0.3, ge=0.0, le=1.0, description="Probability of pitch shift")
    stretch_prob: float = Field(default=0.3, ge=0.0, le=1.0, description="Probability of time stretch")
    noise_type: str = Field(default="white", description="Noise type: white, pink, or babble")
    snr_db: float = Field(default=20.0, ge=0.0, le=60.0, description="SNR for noise injection (dB)")
    max_semitones: float = Field(default=2.0, ge=0.0, le=2.0, description="Max pitch shift (semitones)")
    max_stretch_deviation: float = Field(
        default=0.1, ge=0.0, le=0.1, description="Max time stretch deviation from 1.0"
    )
    n_augmentations: int = Field(default=1, ge=1, le=50, description="Number of augmented copies")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class AugmentedSample(BaseModel):
    audio: List[float]
    transforms_applied: List[str]
    original_length: int
    output_length: int


class AugmentResponse(BaseModel):
    augmentations: List[AugmentedSample]
    n_augmentations: int
    original_length: int
    sample_rate: int
    generated_at: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def voice_augmentation_status() -> Dict[str, Any]:
    """Return service readiness and available augmentation types."""
    scipy_ok = False
    try:
        from scipy import signal as _  # noqa: F401
        scipy_ok = True
    except ImportError:
        pass

    return {
        "status": "ready" if scipy_ok else "degraded",
        "scipy_available": scipy_ok,
        "augmentation_types": ["noise_injection", "pitch_shift", "time_stretch", "spec_augment"],
        "noise_types": ["white", "pink", "babble"],
        "max_semitones": 2.0,
        "max_stretch_deviation": 0.1,
    }


@router.post("/augment", response_model=AugmentResponse)
async def augment_voice(req: AugmentRequest) -> Dict[str, Any]:
    """Augment a voice sample with noise, pitch shift, and/or time stretch.

    Applies a random combination of augmentations based on the configured
    probabilities. Each augmentation is applied independently with its
    own random seed for reproducibility.
    """
    if not req.audio or len(req.audio) < 2:
        raise HTTPException(status_code=400, detail="Audio must have at least 2 samples")

    try:
        from models.voice_augmentation import augment_voice_sample

        audio = np.array(req.audio, dtype=np.float64)
        rng = np.random.default_rng(req.seed)

        augmentations = []
        for i in range(req.n_augmentations):
            sample_seed = int(rng.integers(0, 2**31))
            result = augment_voice_sample(
                audio,
                sr=req.sample_rate,
                noise_prob=req.noise_prob,
                pitch_prob=req.pitch_prob,
                stretch_prob=req.stretch_prob,
                noise_type=req.noise_type,
                snr_db=req.snr_db,
                max_semitones=req.max_semitones,
                max_stretch_deviation=req.max_stretch_deviation,
                seed=sample_seed,
            )
            augmentations.append(AugmentedSample(
                audio=result["audio"].tolist(),
                transforms_applied=result["transforms_applied"],
                original_length=result["original_length"],
                output_length=result["output_length"],
            ))

        return {
            "augmentations": augmentations,
            "n_augmentations": len(augmentations),
            "original_length": len(audio),
            "sample_rate": req.sample_rate,
            "generated_at": time.time(),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("Voice augmentation failed")
        raise HTTPException(status_code=500, detail=f"Augmentation failed: {exc}")
