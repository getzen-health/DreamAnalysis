"""EEG device adapter endpoints (#404).

POST /device-adapters/adapt     -- adapt EEG data from a specific device
GET  /device-adapters/devices   -- list supported devices
GET  /device-adapters/status    -- availability check
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/device-adapters", tags=["device-adapters"])


# -- request / response schemas -----------------------------------------------

class DeviceAdaptInput(BaseModel):
    device_id: str = Field(
        ..., description="Device identifier: emotiv_insight, naox_earbuds, md_neuro"
    )
    signals: List[List[float]] = Field(
        ..., description="2-D array of EEG signals (n_channels, n_samples)"
    )
    source_sr: int = Field(
        0, description="Source sampling rate. If 0, uses device default."
    )
    compute_features: bool = Field(
        True, description="Whether to compute compatible features"
    )


class DeviceAdaptResponse(BaseModel):
    device_id: str
    device_name: str = ""
    original_sr: int = 0
    target_sr: int = 256
    original_samples: int = 0
    resampled_samples: int = 0
    channels_mapped: Dict[str, str] = {}
    features: Optional[Dict[str, Any]] = None
    compatible_models: List[str] = []
    processed_at: float = 0.0


# -- endpoints ----------------------------------------------------------------

@router.post("/adapt", response_model=DeviceAdaptResponse)
async def adapt_device_data(req: DeviceAdaptInput):
    """Adapt EEG data from a specific device to the common format.

    Normalizes sampling rate to 256 Hz, maps channels to standard 10-20
    positions, and optionally computes compatible features.
    """
    from models.device_adapters import (
        get_device_profile,
        map_channels,
        normalize_sampling_rate,
        compute_compatible_features,
        device_profile_to_dict,
    )

    profile = get_device_profile(req.device_id)
    if "error" in profile:
        raise HTTPException(
            400,
            f"Unknown device '{req.device_id}'. "
            f"Supported: {profile.get('supported_devices', [])}",
        )

    signals = np.array(req.signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    source_sr = req.source_sr if req.source_sr > 0 else profile["native_sr"]
    original_samples = signals.shape[1]

    # Normalize sampling rate
    resampled = normalize_sampling_rate(signals, source_sr, target_sr=256)

    # Map channels
    mapped = map_channels(req.device_id, resampled)
    channel_map = {}
    if "mapped_channels" in mapped:
        channel_map = {k: k for k in mapped["mapped_channels"]}

    # Compute features if requested
    features = None
    if req.compute_features:
        feat_result = compute_compatible_features(req.device_id, resampled, fs=256.0)
        if "error" not in feat_result:
            # Convert numpy values for JSON
            features = {}
            for k, v in feat_result.items():
                if isinstance(v, (np.floating, np.integer)):
                    features[k] = float(v)
                elif isinstance(v, np.ndarray):
                    features[k] = v.tolist()
                else:
                    features[k] = v

    # Get compatible models
    dev_info = device_profile_to_dict(req.device_id)
    compatible = dev_info.get("compatible_models", [])

    return DeviceAdaptResponse(
        device_id=req.device_id,
        device_name=profile["name"],
        original_sr=source_sr,
        target_sr=256,
        original_samples=original_samples,
        resampled_samples=resampled.shape[1],
        channels_mapped=channel_map,
        features=features,
        compatible_models=compatible,
        processed_at=time.time(),
    )


@router.get("/devices")
async def list_devices() -> Dict[str, Any]:
    """List all supported EEG devices with their profiles."""
    from models.device_adapters import (
        get_device_profile,
        get_capability_matrix,
        device_profile_to_dict,
    )

    devices = []
    for device_id in ["emotiv_insight", "naox_earbuds", "md_neuro"]:
        info = device_profile_to_dict(device_id)
        devices.append({"device_id": device_id, **info})

    capability_matrix = get_capability_matrix()

    return {
        "devices": devices,
        "device_count": len(devices),
        "capability_matrix": capability_matrix,
    }


@router.get("/status")
async def device_adapters_status() -> Dict[str, Any]:
    """Check availability of device adapter functionality."""
    scipy_ok = False
    try:
        from scipy.signal import welch  # noqa: F401
        scipy_ok = True
    except ImportError:
        pass

    return {
        "ready": True,
        "scipy_available": scipy_ok,
        "features_available": scipy_ok,
        "supported_devices": ["emotiv_insight", "naox_earbuds", "md_neuro"],
        "target_sampling_rate": 256,
    }
