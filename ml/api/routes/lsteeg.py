"""LSTEEG-style deep autoencoder for real-time EEG artifact rejection (#34).

Implements a lightweight learned signal reconstruction pipeline. When a trained
autoencoder model is available it uses it; otherwise falls back to a
conventional bandpass + threshold artifact rejection pipeline.

Based on Zhang et al. (2021) LSTEEG and related autoencoders for EEG cleaning.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/lsteeg", tags=["lsteeg"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LSteegInput(BaseModel):
    signals: List[List[float]]        # (n_channels, n_samples) raw EEG
    fs: float = 256.0
    user_id: str = Field(..., min_length=1)
    amplitude_threshold: float = 75.0  # µV — epochs above this are artifacts


class LSteegResult(BaseModel):
    user_id: str
    cleaned_signals: List[List[float]]  # denoised (n_channels, n_samples)
    artifact_mask: List[bool]           # per-sample artifact flag
    artifact_fraction: float            # 0-1 fraction of samples flagged
    snr_improvement_db: float           # estimated SNR improvement in dB
    model_used: str
    n_channels: int
    n_samples: int
    processed_at: float


class LSteegStatus(BaseModel):
    model_loaded: bool
    model_type: str
    n_channels_supported: int
    total_frames_cleaned: int


# ---------------------------------------------------------------------------
# Autoencoder model (optional)
# ---------------------------------------------------------------------------

_ae_model = None
_frames_cleaned: Dict[str, int] = defaultdict(int)

try:
    from models.denoising_autoencoder import DenoisingAutoencoder  # type: ignore
    _ae_model = DenoisingAutoencoder()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Conventional fallback: bandpass + amplitude gate
# ---------------------------------------------------------------------------

def _conventional_clean(signals: np.ndarray, fs: float,
                          threshold: float) -> tuple:
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [1.0, 45.0], btype="band", fs=fs)
    cleaned = np.zeros_like(signals)
    artifact_masks = []

    for ch in range(signals.shape[0]):
        sig = signals[ch]
        filt = filtfilt(b, a, sig)
        cleaned[ch] = filt
        artifact_masks.append(np.abs(filt) > threshold)

    combined_mask = np.any(np.vstack(artifact_masks), axis=0)
    # Zero-out artifact samples as simple rejection
    cleaned[:, combined_mask] = 0.0
    return cleaned, combined_mask


def _estimate_snr_improvement(raw: np.ndarray, cleaned: np.ndarray,
                                artifact_mask: np.ndarray) -> float:
    """Estimate dB SNR improvement from artifact removal."""
    if not artifact_mask.any():
        return 0.0
    artifact_power  = float(np.mean(raw[:, artifact_mask] ** 2)) + 1e-12
    clean_power     = float(np.mean(cleaned[:, ~artifact_mask] ** 2)) + 1e-12
    return float(10 * np.log10(clean_power / artifact_power))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/denoise", response_model=LSteegResult)
async def lsteeg_denoise(req: LSteegInput):
    """Denoise EEG signals using LSTEEG autoencoder or conventional fallback."""
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    if _ae_model is not None:
        try:
            cleaned = _ae_model.denoise(signals)
            model_used = "lsteeg_autoencoder"
            artifact_mask = np.abs(signals - cleaned).mean(axis=0) > req.amplitude_threshold * 0.3
        except Exception:
            cleaned, artifact_mask = _conventional_clean(signals, req.fs, req.amplitude_threshold)
            model_used = "conventional_fallback"
    else:
        cleaned, artifact_mask = _conventional_clean(signals, req.fs, req.amplitude_threshold)
        model_used = "conventional_fallback"

    artifact_frac = float(artifact_mask.mean())
    snr_db = _estimate_snr_improvement(signals, cleaned, artifact_mask)
    _frames_cleaned[req.user_id] += 1

    return LSteegResult(
        user_id=req.user_id,
        cleaned_signals=cleaned.tolist(),
        artifact_mask=artifact_mask.tolist(),
        artifact_fraction=artifact_frac,
        snr_improvement_db=snr_db,
        model_used=model_used,
        n_channels=signals.shape[0],
        n_samples=signals.shape[1],
        processed_at=time.time(),
    )


@router.get("/status", response_model=LSteegStatus)
async def lsteeg_status():
    """Return LSTEEG model status."""
    return LSteegStatus(
        model_loaded=_ae_model is not None,
        model_type="lsteeg_autoencoder" if _ae_model is not None else "conventional_fallback",
        n_channels_supported=4,
        total_frames_cleaned=sum(_frames_cleaned.values()),
    )


@router.post("/reset/{user_id}")
async def lsteeg_reset(user_id: str):
    """Reset per-user cleaning counter."""
    _frames_cleaned[user_id] = 0
    return {"user_id": user_id, "status": "reset"}
