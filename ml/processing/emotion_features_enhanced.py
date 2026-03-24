"""Enhanced 53-dim emotion feature extraction for 4-channel consumer EEG.

Extracts a compact, interpretable, literature-proven feature set for emotion
classification from 4-channel EEG (Muse 2: TP9, AF7, AF8, TP10).

Feature breakdown (53 total):
    20 DE features:       Differential entropy per band per channel (5 bands x 4 ch)
     5 DASM features:     DE asymmetry AF8 - AF7 per band
     5 RASM features:     DE ratio AF8 / AF7 per band
     5 DCAU features:     DE asymmetry TP10 - TP9 per band (temporal asymmetry)
     1 FAA feature:       Frontal alpha asymmetry log(AF8_alpha) - log(AF7_alpha)
     1 FMT feature:       Frontal midline theta relative power
    12 Hjorth features:   Activity, mobility, complexity per channel (3 x 4 ch)
     4 Spectral entropy:  Normalized spectral entropy per channel

References:
    - CNN-KAN-F2CA (PLOS ONE 2025): DE/PSD/asymmetry features on 4-channel SEED
    - Zheng & Lu (2015): DASM/RASM features (SJTU BCMI Lab, SEED dataset)
    - Davidson (1992): Frontal alpha asymmetry for valence
    - Hjorth (1970): Time-domain EEG descriptors
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from processing.eeg_processor import (
    BANDS,
    bandpass_filter,
    compute_dasm_rasm,
    compute_frontal_asymmetry,
    compute_frontal_midline_theta,
    compute_hjorth_parameters,
    differential_entropy,
    preprocess,
    spectral_entropy,
)

log = logging.getLogger(__name__)

# 5-band layout matching DASM/RASM convention
_BANDS_5 = ["delta", "theta", "alpha", "beta", "gamma"]

# Spectral band importance weights for emotion-relevant feature extraction.
#
# Neuroscience rationale (Muse 2 @ AF7/AF8/TP9/TP10):
#   alpha (1.5x): PRIMARY emotion band. FAA is the most validated valence marker
#                  (Davidson 1992). Inversely related to cortical arousal.
#   theta (1.3x): Meditation, creativity, FMT complements FAA for valence.
#   beta  (1.0x): Active cognition, focus, stress. Neutral weight.
#   delta (0.8x): Deep sleep marker, minimal waking-state emotion relevance.
#   gamma (0.3x): At AF7/AF8, dominated by frontalis EMG noise, not neural gamma.
#                  Strongly suppressed to prevent muscle artifacts inflating features.
#
# Applied to DE features: DE_weighted = DE_raw * weight. This amplifies
# emotion-carrying bands and suppresses noise-dominated bands.
#
# These values MUST stay in sync with BAND_IMPORTANCE_WEIGHTS in
# client/src/lib/eeg-features.ts (TypeScript side).
BAND_IMPORTANCE_WEIGHTS: Dict[str, float] = {
    "delta": 0.8,
    "theta": 1.3,
    "alpha": 1.5,
    "beta": 1.0,
    "gamma": 0.3,
}

# Muse 2 channel order (BrainFlow board_id 22/38)
_CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
_N_CHANNELS = 4

# Expected feature count
ENHANCED_FEATURE_DIM = 53

# Feature name registry for interpretability
FEATURE_NAMES: List[str] = []

# 20 DE features: de_{band}_{channel}
for _band in _BANDS_5:
    for _ch in _CHANNEL_NAMES:
        FEATURE_NAMES.append(f"de_{_band}_{_ch}")

# 5 DASM: dasm_{band} (AF8 - AF7)
for _band in _BANDS_5:
    FEATURE_NAMES.append(f"dasm_{_band}")

# 5 RASM: rasm_{band} (AF8 / AF7)
for _band in _BANDS_5:
    FEATURE_NAMES.append(f"rasm_{_band}")

# 5 DCAU: dcau_{band} (TP10 - TP9)
for _band in _BANDS_5:
    FEATURE_NAMES.append(f"dcau_{_band}")

# 1 FAA + 1 FMT
FEATURE_NAMES.append("faa")
FEATURE_NAMES.append("fmt_relative")

# 12 Hjorth: hjorth_{param}_{channel}
for _ch in _CHANNEL_NAMES:
    for _param in ["activity", "mobility", "complexity"]:
        FEATURE_NAMES.append(f"hjorth_{_param}_{_ch}")

# 4 spectral entropy: se_{channel}
for _ch in _CHANNEL_NAMES:
    FEATURE_NAMES.append(f"se_{_ch}")

assert len(FEATURE_NAMES) == ENHANCED_FEATURE_DIM, (
    f"Expected {ENHANCED_FEATURE_DIM} feature names, got {len(FEATURE_NAMES)}"
)


def extract_enhanced_emotion_features(
    signals: np.ndarray,
    fs: int = 256,
) -> np.ndarray:
    """Extract the 53-dim enhanced emotion feature vector from 4-channel EEG.

    Args:
        signals: (n_channels, n_samples) raw EEG array. Must have >= 4 channels.
                 Channel order: TP9, AF7, AF8, TP10 (BrainFlow Muse 2).
        fs: Sampling rate in Hz.

    Returns:
        1D numpy array of shape (53,) with the feature vector.
        All values are guaranteed to be finite (NaN/inf replaced with 0).
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        # Single channel: pad to 4 channels by repeating
        signals = np.tile(signals[np.newaxis, :], (4, 1))
    if signals.shape[0] < 4:
        # Pad missing channels with zeros
        pad = np.zeros((4 - signals.shape[0], signals.shape[1]), dtype=np.float64)
        signals = np.vstack([signals, pad])

    n_ch = min(signals.shape[0], 4)
    features = np.zeros(ENHANCED_FEATURE_DIM, dtype=np.float64)
    idx = 0

    # Preprocess each channel once
    processed = np.array([preprocess(signals[ch], fs) for ch in range(n_ch)])

    # ---- 20 DE features: per band per channel ----
    # Apply spectral band importance weights: DE_weighted = DE_raw * weight.
    # This amplifies emotion-carrying bands (alpha 1.5x, theta 1.3x) and
    # suppresses noise-dominated bands (gamma 0.3x at AF7/AF8 = EMG artifact).
    for b_idx, band in enumerate(_BANDS_5):
        w = BAND_IMPORTANCE_WEIGHTS.get(band, 1.0)
        for ch_idx in range(n_ch):
            de = differential_entropy(processed[ch_idx], fs)
            features[idx] = de.get(band, 0.0) * w
            idx += 1

    # ---- 5 DASM features: AF8 (ch2) - AF7 (ch1) per band ----
    dasm_rasm = compute_dasm_rasm(signals, fs, left_ch=1, right_ch=2)
    for band in _BANDS_5:
        features[idx] = dasm_rasm.get(f"dasm_{band}", 0.0)
        idx += 1

    # ---- 5 RASM features: AF8 / AF7 per band ----
    for band in _BANDS_5:
        features[idx] = dasm_rasm.get(f"rasm_{band}", 0.0)
        idx += 1

    # ---- 5 DCAU features: temporal asymmetry TP10 (ch3) - TP9 (ch0) per band ----
    dcau = compute_dasm_rasm(signals, fs, left_ch=0, right_ch=3)
    for band in _BANDS_5:
        features[idx] = dcau.get(f"dasm_{band}", 0.0)
        idx += 1

    # ---- 1 FAA feature ----
    faa = compute_frontal_asymmetry(signals, fs, left_ch=1, right_ch=2)
    features[idx] = faa.get("frontal_asymmetry", 0.0)
    idx += 1

    # ---- 1 FMT feature (use AF7 channel = ch1) ----
    fmt = compute_frontal_midline_theta(processed[1], fs)
    features[idx] = fmt.get("fmt_relative", 0.0)
    idx += 1

    # ---- 12 Hjorth features: 3 params x 4 channels ----
    for ch_idx in range(n_ch):
        hjorth = compute_hjorth_parameters(processed[ch_idx])
        features[idx] = hjorth.get("activity", 0.0)
        features[idx + 1] = hjorth.get("mobility", 0.0)
        features[idx + 2] = hjorth.get("complexity", 0.0)
        idx += 3

    # ---- 4 spectral entropy features ----
    for ch_idx in range(n_ch):
        features[idx] = spectral_entropy(processed[ch_idx], fs)
        idx += 1

    assert idx == ENHANCED_FEATURE_DIM, f"Feature index mismatch: {idx} != {ENHANCED_FEATURE_DIM}"

    # Guarantee finite values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def extract_temporal_features(
    current_features: np.ndarray,
    history: Optional[List[np.ndarray]] = None,
    time_interval: float = 2.0,
) -> np.ndarray:
    """Add temporal delta features capturing emotional transitions.

    Computes how features CHANGE over time:
        delta = current - previous epoch
        rate  = delta / time_interval

    For the first epoch (no history), deltas are zero.

    Args:
        current_features: 1D array of shape (53,) from extract_enhanced_emotion_features.
        history: List of previous feature vectors. Uses the most recent entry.
                 If None or empty, returns zeros for delta features.
        time_interval: Seconds between epochs (e.g. 2.0 for 50% overlap at 4-sec windows).

    Returns:
        1D numpy array of shape (106,):
            features[0:53]   = instantaneous features (passed through)
            features[53:106] = delta features (current - previous) / time_interval
    """
    current_features = np.asarray(current_features, dtype=np.float64)
    assert current_features.shape == (ENHANCED_FEATURE_DIM,), (
        f"Expected shape ({ENHANCED_FEATURE_DIM},), got {current_features.shape}"
    )

    combined = np.zeros(ENHANCED_FEATURE_DIM * 2, dtype=np.float64)
    combined[:ENHANCED_FEATURE_DIM] = current_features

    if history and len(history) > 0:
        prev = np.asarray(history[-1], dtype=np.float64)
        if prev.shape == current_features.shape:
            delta = (current_features - prev) / max(time_interval, 1e-6)
            combined[ENHANCED_FEATURE_DIM:] = delta

    # Guarantee finite values
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

    return combined


def get_feature_names(include_temporal: bool = False) -> List[str]:
    """Return human-readable feature names.

    Args:
        include_temporal: If True, append delta_ prefixed names for temporal features.

    Returns:
        List of feature name strings.
    """
    names = list(FEATURE_NAMES)
    if include_temporal:
        names.extend([f"delta_{n}" for n in FEATURE_NAMES])
    return names
