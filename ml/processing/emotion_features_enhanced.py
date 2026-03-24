"""Enhanced 80-dim emotion feature extraction for 4-channel consumer EEG.

Extracts a compact, interpretable, literature-proven feature set for emotion
classification from 4-channel EEG (Muse 2: TP9, AF7, AF8, TP10).

Feature breakdown (80 total):
    20 DE features:       Differential entropy per band per channel (5 bands x 4 ch)
     8 DE sub-band:       DE per alpha sub-band per channel (2 sub-bands x 4 ch)
     5 DASM features:     DE asymmetry AF8 - AF7 per band (5 main bands)
     2 DASM sub-band:     DE asymmetry AF8 - AF7 per alpha sub-band
     5 RASM features:     DE ratio AF8 / AF7 per band (5 main bands)
     2 RASM sub-band:     DE ratio AF8 / AF7 per alpha sub-band
     5 DCAU features:     DE asymmetry TP10 - TP9 per band (5 main bands)
     2 DCAU sub-band:     DE asymmetry TP10 - TP9 per alpha sub-band
     1 FAA feature:       Frontal alpha asymmetry log(AF8_alpha) - log(AF7_alpha)
     1 HAA feature:       High-alpha asymmetry (10-12 Hz, more emotion-specific)
     1 FMT feature:       Frontal midline theta relative power
    12 Hjorth features:   Activity, mobility, complexity per channel (3 x 4 ch)
     4 Spectral entropy:  Normalized spectral entropy per channel
     4 HFD features:      Higuchi fractal dimension per channel (nonlinear complexity)
     4 SampEn features:   Sample entropy per channel (temporal regularity)
     4 LZC features:      Lempel-Ziv complexity per channel (algorithmic complexity)

Alpha sub-band rationale (Klimesch 1999, Bazanova & Vernon 2014):
    Low-alpha  (8-10 Hz):  General alertness, tonic arousal, attentional demands
    High-alpha (10-12 Hz): Task-specific cortical processing, semantic memory,
                           emotional engagement. MORE emotion-specific than full-band
                           alpha. High-alpha suppression indexes active emotional
                           processing. High-alpha asymmetry (HAA) is a more precise
                           valence indicator than standard FAA for emotional states.

Nonlinear complexity rationale:
    HFD (Higuchi 1988): Measures waveform fractal dimension [1.0-2.0]. Arousal
        increases neural complexity, producing higher HFD. Ahmadlou et al. (2012)
        showed HFD discriminates emotional states. HFD + band powers improves
        emotion accuracy by 3-8% vs band powers alone on DEAP.
    SampEn (Richman & Moorman 2000): Measures temporal regularity/predictability.
        Complements spectral entropy (frequency domain) by capturing time-domain
        regularity. Higher SampEn = more complex/unpredictable temporal dynamics.
        Emotional engagement increases temporal complexity.
    LZC (Lempel & Ziv 1976): Binary sequence complexity measure [0-1]. Higher
        LZC = more complex brain activity. Correlates with consciousness level
        and emotional engagement. Fast O(N) computation.

References:
    - CNN-KAN-F2CA (PLOS ONE 2025): DE/PSD/asymmetry features on 4-channel SEED
    - Zheng & Lu (2015): DASM/RASM features (SJTU BCMI Lab, SEED dataset)
    - Davidson (1992): Frontal alpha asymmetry for valence
    - Klimesch (1999): Alpha sub-bands and cognitive processing
    - Bazanova & Vernon (2014): High-alpha specificity for emotional states
    - Hjorth (1970): Time-domain EEG descriptors
    - Higuchi (1988): Fractal dimension of irregular time series
    - Richman & Moorman (2000): Sample entropy for physiological signals
    - Lempel & Ziv (1976): Complexity of finite sequences
    - Ahmadlou et al. (2012): HFD for EEG emotion classification
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.neural_complexity import (
    _higuchi_fractal_dimension,
    _lempel_ziv_complexity,
    _sample_entropy,
)
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

# 5-band layout matching DASM/RASM convention (original bands)
_BANDS_5 = ["delta", "theta", "alpha", "beta", "gamma"]

# Alpha sub-bands for finer-grained emotion features
_ALPHA_SUB_BANDS = ["low_alpha", "high_alpha"]

# Spectral band importance weights for emotion-relevant feature extraction.
#
# Neuroscience rationale (Muse 2 @ AF7/AF8/TP9/TP10):
#   alpha (1.5x): PRIMARY emotion band. FAA is the most validated valence marker
#                  (Davidson 1992). Inversely related to cortical arousal.
#   low_alpha (1.2x): General alertness / tonic arousal. Less emotion-specific
#                      than high_alpha (Klimesch 1999). Moderate weight.
#   high_alpha (1.8x): Task-specific processing, emotional engagement. MORE
#                       emotion-specific than full-band alpha (Bazanova & Vernon 2014).
#                       High-alpha suppression = active emotional processing.
#                       Highest weight of all bands.
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
    "low_alpha": 1.2,
    "high_alpha": 1.8,
    "beta": 1.0,
    "gamma": 0.3,
}

# Muse 2 channel order (BrainFlow board_id 22/38)
_CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
_N_CHANNELS = 4

# Expected feature count (53 original + 15 alpha sub-band + 12 nonlinear complexity)
ENHANCED_FEATURE_DIM = 80

# Feature name registry for interpretability
FEATURE_NAMES: List[str] = []

# 20 DE features: de_{band}_{channel} (5 main bands x 4 channels)
for _band in _BANDS_5:
    for _ch in _CHANNEL_NAMES:
        FEATURE_NAMES.append(f"de_{_band}_{_ch}")

# 8 DE sub-band features: de_{sub_band}_{channel} (2 alpha sub-bands x 4 channels)
for _band in _ALPHA_SUB_BANDS:
    for _ch in _CHANNEL_NAMES:
        FEATURE_NAMES.append(f"de_{_band}_{_ch}")

# 5 DASM: dasm_{band} (AF8 - AF7) for main bands
for _band in _BANDS_5:
    FEATURE_NAMES.append(f"dasm_{_band}")

# 2 DASM sub-band: dasm_{sub_band} (AF8 - AF7) for alpha sub-bands
for _band in _ALPHA_SUB_BANDS:
    FEATURE_NAMES.append(f"dasm_{_band}")

# 5 RASM: rasm_{band} (AF8 / AF7) for main bands
for _band in _BANDS_5:
    FEATURE_NAMES.append(f"rasm_{_band}")

# 2 RASM sub-band: rasm_{sub_band} (AF8 / AF7) for alpha sub-bands
for _band in _ALPHA_SUB_BANDS:
    FEATURE_NAMES.append(f"rasm_{_band}")

# 5 DCAU: dcau_{band} (TP10 - TP9) for main bands
for _band in _BANDS_5:
    FEATURE_NAMES.append(f"dcau_{_band}")

# 2 DCAU sub-band: dcau_{sub_band} (TP10 - TP9) for alpha sub-bands
for _band in _ALPHA_SUB_BANDS:
    FEATURE_NAMES.append(f"dcau_{_band}")

# 1 FAA + 1 HAA + 1 FMT
FEATURE_NAMES.append("faa")
FEATURE_NAMES.append("haa")
FEATURE_NAMES.append("fmt_relative")

# 12 Hjorth: hjorth_{param}_{channel}
for _ch in _CHANNEL_NAMES:
    for _param in ["activity", "mobility", "complexity"]:
        FEATURE_NAMES.append(f"hjorth_{_param}_{_ch}")

# 4 spectral entropy: se_{channel}
for _ch in _CHANNEL_NAMES:
    FEATURE_NAMES.append(f"se_{_ch}")

# 4 Higuchi fractal dimension: hfd_{channel}
for _ch in _CHANNEL_NAMES:
    FEATURE_NAMES.append(f"hfd_{_ch}")

# 4 sample entropy: sampen_{channel}
for _ch in _CHANNEL_NAMES:
    FEATURE_NAMES.append(f"sampen_{_ch}")

# 4 Lempel-Ziv complexity: lzc_{channel}
for _ch in _CHANNEL_NAMES:
    FEATURE_NAMES.append(f"lzc_{_ch}")

assert len(FEATURE_NAMES) == ENHANCED_FEATURE_DIM, (
    f"Expected {ENHANCED_FEATURE_DIM} feature names, got {len(FEATURE_NAMES)}"
)


def _compute_high_alpha_asymmetry(
    signals: np.ndarray, fs: int, left_ch: int = 1, right_ch: int = 2
) -> float:
    """Compute High-Alpha Asymmetry (HAA) in the 10-12 Hz sub-band.

    HAA = log(right_high_alpha_power) - log(left_high_alpha_power)

    High-alpha (10-12 Hz) is more emotion-specific than full-band alpha (8-12 Hz).
    It indexes task-specific cortical processing and emotional engagement rather
    than general alertness (Bazanova & Vernon 2014).

    Positive HAA = right-dominant high-alpha = approach motivation / positive affect.
    Negative HAA = left-dominant high-alpha = withdrawal / negative affect.

    Args:
        signals: (n_channels, n_samples) EEG array.
        fs: Sampling rate in Hz.
        left_ch: Left frontal channel (AF7 = ch1 for Muse 2).
        right_ch: Right frontal channel (AF8 = ch2 for Muse 2).

    Returns:
        HAA value (float). 0.0 if channels unavailable.
    """
    if signals.ndim < 2 or signals.shape[0] <= max(left_ch, right_ch):
        return 0.0

    left_sig = preprocess(signals[left_ch], fs)
    right_sig = preprocess(signals[right_ch], fs)

    # Bandpass to high-alpha (10-12 Hz)
    left_ha = bandpass_filter(left_sig, 10.0, 12.0, fs)
    right_ha = bandpass_filter(right_sig, 10.0, 12.0, fs)

    left_power = float(np.var(left_ha))
    right_power = float(np.var(right_ha))

    # log(right) - log(left), same convention as FAA
    return float(
        np.log(max(right_power, 1e-12)) - np.log(max(left_power, 1e-12))
    )


def extract_enhanced_emotion_features(
    signals: np.ndarray,
    fs: int = 256,
) -> np.ndarray:
    """Extract the 80-dim enhanced emotion feature vector from 4-channel EEG.

    Args:
        signals: (n_channels, n_samples) raw EEG array. Must have >= 4 channels.
                 Channel order: TP9, AF7, AF8, TP10 (BrainFlow Muse 2).
        fs: Sampling rate in Hz.

    Returns:
        1D numpy array of shape (80,) with the feature vector.
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

    # ---- 20 DE features: per main band per channel ----
    # Apply spectral band importance weights: DE_weighted = DE_raw * weight.
    # This amplifies emotion-carrying bands (alpha 1.5x, theta 1.3x) and
    # suppresses noise-dominated bands (gamma 0.3x at AF7/AF8 = EMG artifact).
    for b_idx, band in enumerate(_BANDS_5):
        w = BAND_IMPORTANCE_WEIGHTS.get(band, 1.0)
        for ch_idx in range(n_ch):
            de = differential_entropy(processed[ch_idx], fs)
            features[idx] = de.get(band, 0.0) * w
            idx += 1

    # ---- 8 DE alpha sub-band features: per sub-band per channel ----
    # High-alpha (10-12 Hz) is more emotion-specific than full-band alpha.
    # Low-alpha (8-10 Hz) tracks general alertness/arousal.
    # Splitting provides finer-grained discrimination for valence detection.
    for band in _ALPHA_SUB_BANDS:
        w = BAND_IMPORTANCE_WEIGHTS.get(band, 1.0)
        for ch_idx in range(n_ch):
            de = differential_entropy(processed[ch_idx], fs)
            features[idx] = de.get(band, 0.0) * w
            idx += 1

    # ---- 5 DASM features: AF8 (ch2) - AF7 (ch1) per main band ----
    dasm_rasm = compute_dasm_rasm(signals, fs, left_ch=1, right_ch=2)
    for band in _BANDS_5:
        features[idx] = dasm_rasm.get(f"dasm_{band}", 0.0)
        idx += 1

    # ---- 2 DASM alpha sub-band features ----
    for band in _ALPHA_SUB_BANDS:
        features[idx] = dasm_rasm.get(f"dasm_{band}", 0.0)
        idx += 1

    # ---- 5 RASM features: AF8 / AF7 per main band ----
    for band in _BANDS_5:
        features[idx] = dasm_rasm.get(f"rasm_{band}", 0.0)
        idx += 1

    # ---- 2 RASM alpha sub-band features ----
    for band in _ALPHA_SUB_BANDS:
        features[idx] = dasm_rasm.get(f"rasm_{band}", 0.0)
        idx += 1

    # ---- 5 DCAU features: temporal asymmetry TP10 (ch3) - TP9 (ch0) per main band ----
    dcau = compute_dasm_rasm(signals, fs, left_ch=0, right_ch=3)
    for band in _BANDS_5:
        features[idx] = dcau.get(f"dasm_{band}", 0.0)
        idx += 1

    # ---- 2 DCAU alpha sub-band features ----
    for band in _ALPHA_SUB_BANDS:
        features[idx] = dcau.get(f"dasm_{band}", 0.0)
        idx += 1

    # ---- 1 FAA feature ----
    faa = compute_frontal_asymmetry(signals, fs, left_ch=1, right_ch=2)
    features[idx] = faa.get("frontal_asymmetry", 0.0)
    idx += 1

    # ---- 1 HAA feature (High-Alpha Asymmetry) ----
    # More emotion-specific than full-band FAA (Bazanova & Vernon 2014).
    # High-alpha (10-12 Hz) indexes emotional engagement, not just alertness.
    features[idx] = _compute_high_alpha_asymmetry(signals, fs, left_ch=1, right_ch=2)
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

    # ---- 4 HFD features: Higuchi fractal dimension per channel ----
    # HFD measures waveform complexity [1.0-2.0]. Higher HFD = more complex
    # neural activity. Arousal increases neural complexity (Ahmadlou et al. 2012).
    # HFD + band powers improves emotion accuracy by 3-8% vs band powers alone.
    for ch_idx in range(n_ch):
        features[idx] = _higuchi_fractal_dimension(processed[ch_idx])
        idx += 1

    # ---- 4 SampEn features: sample entropy per channel ----
    # SampEn measures temporal regularity/predictability. Complements spectral
    # entropy (frequency domain) with time-domain regularity information.
    # Higher SampEn = more complex/unpredictable temporal dynamics.
    for ch_idx in range(n_ch):
        features[idx] = _sample_entropy(processed[ch_idx])
        idx += 1

    # ---- 4 LZC features: Lempel-Ziv complexity per channel ----
    # LZC is a binary sequence complexity measure [0-1]. Higher LZC = more
    # complex brain activity. Fast O(N) computation. Correlates with
    # consciousness level and emotional engagement.
    for ch_idx in range(n_ch):
        features[idx] = _lempel_ziv_complexity(processed[ch_idx])
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
        current_features: 1D array of shape (80,) from extract_enhanced_emotion_features.
        history: List of previous feature vectors. Uses the most recent entry.
                 If None or empty, returns zeros for delta features.
        time_interval: Seconds between epochs (e.g. 2.0 for 50% overlap at 4-sec windows).

    Returns:
        1D numpy array of shape (160,):
            features[0:80]    = instantaneous features (passed through)
            features[80:160]  = delta features (current - previous) / time_interval
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
