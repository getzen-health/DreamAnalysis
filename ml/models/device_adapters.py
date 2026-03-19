"""Device abstraction layer for multiple EEG hardware.

Supports mapping, normalization, and feature extraction across different
consumer EEG devices with varying channel counts and sampling rates.

Supported devices:
    - Emotiv Insight: 5 channels (AF3, AF4, T7, T8, Pz), 128 Hz
    - NAOX earbuds: 2 channels (TP9, TP10), 250 Hz
    - Master & Dynamic Neuro: 2 channels (FP1, FP2), 256 Hz

Design principles:
    - All devices normalized to 256 Hz common rate
    - Channel mapping to 10-20 standard positions
    - Graceful degradation: compute what you can with available channels
    - Capability matrix: explicitly state which models work with which devices
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Common target sampling rate (matches Muse 2)
_TARGET_SR = 256

# Device profiles: name -> specs
_DEVICE_PROFILES: Dict[str, Dict] = {
    "emotiv_insight": {
        "name": "Emotiv Insight",
        "channels": ["AF3", "AF4", "T7", "T8", "Pz"],
        "channel_count": 5,
        "native_sr": 128,
        "electrode_type": "semi-dry polymer",
        "has_frontal_pair": True,
        "has_temporal_pair": True,
        "has_parietal": True,
        "notes": "5-channel consumer headset, good for frontal asymmetry + parietal alpha",
    },
    "naox_earbuds": {
        "name": "NAOX Earbuds",
        "channels": ["TP9", "TP10"],
        "channel_count": 2,
        "native_sr": 250,
        "electrode_type": "in-ear conductive",
        "has_frontal_pair": False,
        "has_temporal_pair": True,
        "has_parietal": False,
        "notes": "2-channel ear EEG, temporal regions only, no frontal asymmetry",
    },
    "md_neuro": {
        "name": "Master & Dynamic Neuro",
        "channels": ["FP1", "FP2"],
        "channel_count": 2,
        "native_sr": 256,
        "electrode_type": "dry forehead",
        "has_frontal_pair": True,
        "has_temporal_pair": False,
        "has_parietal": False,
        "notes": "2-channel frontal headphones, supports FAA but not temporal features",
    },
}

# Standard 10-20 position mapping (device channel -> standard position)
_CHANNEL_MAPS: Dict[str, Dict[str, str]] = {
    "emotiv_insight": {
        "AF3": "AF3",
        "AF4": "AF4",
        "T7": "T7",
        "T8": "T8",
        "Pz": "Pz",
    },
    "naox_earbuds": {
        "TP9": "TP9",
        "TP10": "TP10",
    },
    "md_neuro": {
        "FP1": "FP1",
        "FP2": "FP2",
    },
}

# Model compatibility matrix: model_name -> required capabilities
_MODEL_REQUIREMENTS: Dict[str, Dict] = {
    "emotion_classifier": {
        "min_channels": 2,
        "requires_frontal": True,
        "requires_temporal": False,
        "description": "Needs frontal pair for FAA-based valence",
    },
    "sleep_staging": {
        "min_channels": 1,
        "requires_frontal": False,
        "requires_temporal": False,
        "description": "Works with any channel, uses band power ratios",
    },
    "stress_detector": {
        "min_channels": 2,
        "requires_frontal": True,
        "requires_temporal": False,
        "description": "Beta asymmetry requires frontal pair",
    },
    "meditation_classifier": {
        "min_channels": 1,
        "requires_frontal": False,
        "requires_temporal": False,
        "description": "Alpha/theta ratios work with any channel",
    },
    "drowsiness_detector": {
        "min_channels": 1,
        "requires_frontal": False,
        "requires_temporal": False,
        "description": "Theta increase detectable from any channel",
    },
    "attention_classifier": {
        "min_channels": 1,
        "requires_frontal": True,
        "requires_temporal": False,
        "description": "Frontal beta/theta ratio preferred",
    },
    "flow_state_detector": {
        "min_channels": 2,
        "requires_frontal": True,
        "requires_temporal": False,
        "description": "Needs frontal alpha/theta coherence",
    },
}


def get_device_profile(device_id: str) -> Dict:
    """Get the hardware profile for a supported device.

    Args:
        device_id: One of 'emotiv_insight', 'naox_earbuds', 'md_neuro'.

    Returns:
        Dict with device specs. Returns error dict if device unknown.
    """
    profile = _DEVICE_PROFILES.get(device_id)
    if profile is None:
        return {
            "error": "unknown_device",
            "supported_devices": list(_DEVICE_PROFILES.keys()),
        }
    return dict(profile)


def map_channels(device_id: str, data: np.ndarray) -> Dict:
    """Map device-specific channels to standard 10-20 positions.

    Args:
        device_id: Device identifier string.
        data: 2-D array (n_channels, n_samples). Channel order must match
              the device profile's channel list.

    Returns:
        Dict mapping standard position names to their signal arrays.
        Includes 'unmapped' key if channel counts don't match.
    """
    profile = _DEVICE_PROFILES.get(device_id)
    if profile is None:
        return {"error": "unknown_device"}

    channel_map = _CHANNEL_MAPS.get(device_id, {})
    expected_channels = profile["channels"]

    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_ch = data.shape[0]
    result: Dict = {"device": device_id, "mapped_channels": {}}

    for i, ch_name in enumerate(expected_channels):
        if i < n_ch:
            standard_name = channel_map.get(ch_name, ch_name)
            result["mapped_channels"][standard_name] = data[i]

    result["channel_count"] = min(n_ch, len(expected_channels))
    return result


def normalize_sampling_rate(
    data: np.ndarray, source_sr: int, target_sr: int = _TARGET_SR
) -> np.ndarray:
    """Resample all device data to a common sampling rate.

    Uses linear interpolation for resampling. For production, scipy.signal.resample
    would be preferred but this avoids hard dependency for simple cases.

    Args:
        data: 1-D or 2-D array of EEG signals.
        source_sr: Original sampling rate in Hz.
        target_sr: Target sampling rate (default 256 Hz).

    Returns:
        Resampled data array.
    """
    if source_sr == target_sr:
        return data.copy()

    if data.ndim == 1:
        n_source = len(data)
        n_target = int(n_source * target_sr / source_sr)
        x_source = np.linspace(0, 1, n_source)
        x_target = np.linspace(0, 1, n_target)
        return np.interp(x_target, x_source, data).astype(data.dtype)

    # Multi-channel
    n_channels, n_source = data.shape
    n_target = int(n_source * target_sr / source_sr)
    resampled = np.zeros((n_channels, n_target), dtype=data.dtype)
    x_source = np.linspace(0, 1, n_source)
    x_target = np.linspace(0, 1, n_target)

    for ch in range(n_channels):
        resampled[ch] = np.interp(x_target, x_source, data[ch])

    return resampled


def compute_compatible_features(
    device_id: str, data: np.ndarray, fs: float = _TARGET_SR
) -> Dict:
    """Compute emotion-relevant features for whatever channels are available.

    Graceful degradation: computes frontal asymmetry only if frontal pair
    exists, temporal features only if temporal pair exists, etc.

    Args:
        device_id: Device identifier.
        data: 2-D array (n_channels, n_samples) already at target SR.
        fs: Sampling rate.

    Returns:
        Dict of computed features + list of unavailable features.
    """
    from scipy.signal import welch

    profile = _DEVICE_PROFILES.get(device_id)
    if profile is None:
        return {"error": "unknown_device"}

    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_ch, n_samples = data.shape
    nperseg = min(n_samples, int(fs * 2))

    features: Dict = {"device": device_id, "available": [], "unavailable": []}

    # Band power for each channel
    band_defs = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (12.0, 30.0),
    }

    channel_powers: List[Dict[str, float]] = []
    for ch_idx in range(n_ch):
        freqs, psd = welch(data[ch_idx], fs=fs, nperseg=nperseg)
        ch_powers: Dict[str, float] = {}
        for band_name, (lo, hi) in band_defs.items():
            mask = (freqs >= lo) & (freqs <= hi)
            ch_powers[band_name] = float(np.mean(psd[mask])) if mask.any() else 1e-12
        channel_powers.append(ch_powers)

    # Average band powers across all channels
    for band_name in band_defs:
        avg = float(np.mean([cp[band_name] for cp in channel_powers]))
        features[f"avg_{band_name}_power"] = round(avg, 8)
    features["available"].append("band_powers")

    # Alpha/beta ratio (always available)
    avg_alpha = features["avg_alpha_power"]
    avg_beta = features["avg_beta_power"]
    features["alpha_beta_ratio"] = round(avg_alpha / max(avg_beta, 1e-12), 4)
    features["theta_beta_ratio"] = round(
        features["avg_theta_power"] / max(avg_beta, 1e-12), 4
    )
    features["available"].append("ratios")

    # Frontal asymmetry (needs frontal pair)
    if profile.get("has_frontal_pair") and n_ch >= 2:
        channels = profile["channels"]
        # Find left/right frontal indices
        left_idx, right_idx = _find_frontal_pair(channels)
        if left_idx is not None and right_idx is not None:
            freqs_l, psd_l = welch(data[left_idx], fs=fs, nperseg=nperseg)
            freqs_r, psd_r = welch(data[right_idx], fs=fs, nperseg=nperseg)
            alpha_mask = (freqs_l >= 8.0) & (freqs_l <= 12.0)
            l_alpha = float(np.mean(psd_l[alpha_mask])) if alpha_mask.any() else 1e-12
            r_alpha = float(np.mean(psd_r[alpha_mask])) if alpha_mask.any() else 1e-12
            faa = float(np.log(max(r_alpha, 1e-12)) - np.log(max(l_alpha, 1e-12)))
            features["frontal_asymmetry"] = round(faa, 6)
            features["available"].append("frontal_asymmetry")
        else:
            features["unavailable"].append("frontal_asymmetry")
    else:
        features["unavailable"].append("frontal_asymmetry")

    # Temporal features (needs temporal pair)
    if profile.get("has_temporal_pair") and n_ch >= 2:
        channels = profile["channels"]
        left_t, right_t = _find_temporal_pair(channels)
        if left_t is not None and right_t is not None:
            freqs_lt, psd_lt = welch(data[left_t], fs=fs, nperseg=nperseg)
            freqs_rt, psd_rt = welch(data[right_t], fs=fs, nperseg=nperseg)
            alpha_mask = (freqs_lt >= 8.0) & (freqs_lt <= 12.0)
            lt_alpha = float(np.mean(psd_lt[alpha_mask])) if alpha_mask.any() else 1e-12
            rt_alpha = float(np.mean(psd_rt[alpha_mask])) if alpha_mask.any() else 1e-12
            temporal_asym = float(
                np.log(max(rt_alpha, 1e-12)) - np.log(max(lt_alpha, 1e-12))
            )
            features["temporal_asymmetry"] = round(temporal_asym, 6)
            features["available"].append("temporal_asymmetry")
        else:
            features["unavailable"].append("temporal_asymmetry")
    else:
        features["unavailable"].append("temporal_asymmetry")

    return features


def get_capability_matrix() -> Dict:
    """Return which ML models are compatible with which devices.

    Returns:
        Dict mapping device_id -> list of compatible model names with notes.
    """
    matrix: Dict[str, List[Dict]] = {}

    for device_id, profile in _DEVICE_PROFILES.items():
        compatible: List[Dict] = []

        for model_name, reqs in _MODEL_REQUIREMENTS.items():
            is_compatible = True
            reasons: List[str] = []

            if profile["channel_count"] < reqs["min_channels"]:
                is_compatible = False
                reasons.append(
                    f"needs {reqs['min_channels']} channels, has {profile['channel_count']}"
                )

            if reqs.get("requires_frontal") and not profile.get("has_frontal_pair"):
                is_compatible = False
                reasons.append("requires frontal electrode pair")

            if reqs.get("requires_temporal") and not profile.get("has_temporal_pair"):
                is_compatible = False
                reasons.append("requires temporal electrode pair")

            compatible.append({
                "model": model_name,
                "compatible": is_compatible,
                "degraded": (
                    not is_compatible
                    or profile["channel_count"] < 4
                ),
                "reasons": reasons if not is_compatible else [],
                "description": reqs["description"],
            })

        matrix[device_id] = compatible

    return matrix


def device_profile_to_dict(device_id: str) -> Dict:
    """Get a JSON-safe device profile with capability info.

    Args:
        device_id: Device identifier.

    Returns:
        JSON-serializable dict with profile + capabilities.
    """
    profile = get_device_profile(device_id)
    if "error" in profile:
        return profile

    cap_matrix = get_capability_matrix()
    capabilities = cap_matrix.get(device_id, [])

    compatible_models = [c["model"] for c in capabilities if c["compatible"]]
    incompatible_models = [c["model"] for c in capabilities if not c["compatible"]]

    return {
        **profile,
        "target_sampling_rate": _TARGET_SR,
        "compatible_models": compatible_models,
        "incompatible_models": incompatible_models,
        "capability_details": capabilities,
    }


# -- internal helpers ---------------------------------------------------------


def _find_frontal_pair(channels: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """Find left and right frontal channel indices."""
    left_frontal = {"AF3", "AF7", "F3", "FP1", "F7"}
    right_frontal = {"AF4", "AF8", "F4", "FP2", "F8"}

    left_idx = None
    right_idx = None

    for i, ch in enumerate(channels):
        if ch in left_frontal and left_idx is None:
            left_idx = i
        if ch in right_frontal and right_idx is None:
            right_idx = i

    return left_idx, right_idx


def _find_temporal_pair(channels: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """Find left and right temporal channel indices."""
    left_temporal = {"TP9", "T7", "T3"}
    right_temporal = {"TP10", "T8", "T4"}

    left_idx = None
    right_idx = None

    for i, ch in enumerate(channels):
        if ch in left_temporal and left_idx is None:
            left_idx = i
        if ch in right_temporal and right_idx is None:
            right_idx = i

    return left_idx, right_idx


# -- module-level accessors ---------------------------------------------------


def get_adapter():
    """Return a dict of adapter functions."""
    return {
        "get_device_profile": get_device_profile,
        "map_channels": map_channels,
        "normalize_sampling_rate": normalize_sampling_rate,
        "compute_compatible_features": compute_compatible_features,
        "get_capability_matrix": get_capability_matrix,
        "device_profile_to_dict": device_profile_to_dict,
    }
