"""CNN-KAN pseudo-RGB feature engineering endpoint.

Exposes the CNN-KAN feature engineering pipeline (DE + PSD + EVI -> pseudo-RGB)
as a FastAPI sub-router mounted at /cnn-kan-features.

Endpoints:
    POST /cnn-kan-features/transform — transform raw EEG to pseudo-RGB image
    GET  /cnn-kan-features/status    — pipeline status and configuration info
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/cnn-kan-features", tags=["cnn-kan-features"])

# ── Lazy module loading ──────────────────────────────────────────────────────

_module = None


def _get_module():
    global _module
    if _module is None:
        try:
            from models import cnn_kan_feature_engineering

            _module = cnn_kan_feature_engineering
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"CNN-KAN feature engineering module unavailable: {exc}",
            )
    return _module


# ── Schemas ──────────────────────────────────────────────────────────────────


class FeatureTransformRequest(BaseModel):
    """Raw EEG data for pseudo-RGB feature transformation.

    Attributes:
        signals:        2-D list (n_channels, n_samples). For Muse 2:
                        4 channels, 1024+ samples (4 s @ 256 Hz).
        fs:             Sampling rate in Hz. Default 256.0.
        window_seconds: Window length for feature extraction. Default 4.0.
        overlap:        Window overlap fraction (0.0-0.9). Default 0.5.
    """

    signals: List[List[float]] = Field(
        ...,
        description="2-D EEG array [[ch0...], [ch1...], ...]. Shape: (n_channels, n_samples).",
    )
    fs: float = Field(256.0, description="Sampling rate in Hz.")
    window_seconds: float = Field(4.0, ge=0.5, le=30.0, description="Window length in seconds.")
    overlap: float = Field(0.5, ge=0.0, le=0.9, description="Window overlap fraction.")


class FeatureTransformResponse(BaseModel):
    """Result of pseudo-RGB feature transformation.

    Attributes:
        image_shape:     (n_rows, n_time_windows, 3) shape of the pseudo-RGB image.
        n_rows:          Number of image rows (channels * bands).
        n_windows:       Number of time windows extracted.
        n_channels:      Input EEG channel count.
        n_bands:         Number of frequency bands.
        feature_stats:   Summary statistics of DE, PSD, EVI features.
        processed_at:    Unix timestamp.
    """

    image_shape: List[int]
    n_rows: int
    n_windows: int
    n_channels: int
    n_bands: int
    feature_stats: Dict[str, Any]
    processed_at: float


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/transform", response_model=FeatureTransformResponse)
async def cnn_kan_features_transform(req: FeatureTransformRequest):
    """Transform raw EEG signals to pseudo-RGB image representation.

    Extracts Differential Entropy (DE), Power Spectral Density (PSD), and
    Engagement/Vigilance Index (EVI) features from the EEG signal, then maps
    them to a 3-channel pseudo-RGB image suitable for CNN-KAN input.

    **Pipeline**:
    1. Segment signal into overlapping windows
    2. Per window: compute DE, PSD, EVI features per channel per band
    3. Map R=DE, G=PSD, B=EVI
    4. Normalize to [0, 1] image range

    **Input**: `signals` as a 2-D list (n_channels, n_samples).
    For Muse 2: 4 channels x 1024 samples (4 s @ 256 Hz).
    """
    # Parse and validate signals
    try:
        signals = np.array(req.signals, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot parse signals: {exc}")

    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    if signals.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"signals must be 1-D or 2-D, got shape {signals.shape}",
        )

    n_channels, n_samples = signals.shape
    if n_samples < 4:
        raise HTTPException(
            status_code=422,
            detail=f"signals too short ({n_samples} samples, need >= 4)",
        )

    mod = _get_module()

    # Run full pipeline
    result = mod.prepare_cnn_kan_input(
        signals,
        fs=req.fs,
        window_seconds=req.window_seconds,
        overlap=req.overlap,
    )

    # Compute feature stats from first window for reporting
    de = mod.compute_de_features(signals, fs=req.fs)
    psd = mod.compute_psd_features(signals, fs=req.fs)
    evi = mod.compute_evi_features(signals, fs=req.fs)
    stats = mod.feature_stats_to_dict(de, psd, evi)

    return FeatureTransformResponse(
        image_shape=list(result["shape"]),
        n_rows=result["n_rows"],
        n_windows=result["n_windows"],
        n_channels=result["n_channels"],
        n_bands=result["n_bands"],
        feature_stats=stats,
        processed_at=time.time(),
    )


@router.get("/status")
async def cnn_kan_features_status():
    """Return CNN-KAN feature engineering pipeline status and configuration.

    Reports available frequency bands, channel names, and pipeline parameters.
    """
    mod = _get_module()
    return {
        "status": "ok",
        "pipeline": "CNN-KAN pseudo-RGB feature engineering",
        "frequency_bands": mod.FREQUENCY_BANDS,
        "band_names": mod.BAND_NAMES,
        "n_bands": mod.N_BANDS,
        "channel_names": mod.CHANNEL_NAMES,
        "n_channels": mod.N_CHANNELS,
        "image_rows": mod.N_ROWS,
        "feature_types": ["DE (Differential Entropy)", "PSD (Power Spectral Density)", "EVI (Engagement/Vigilance Index)"],
        "output_format": "pseudo-RGB image (n_rows, n_time_windows, 3)",
    }
