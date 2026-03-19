"""CNN-KAN pseudo-RGB feature engineering for 4-channel EEG.

Maps Differential Entropy (DE), Power Spectral Density (PSD), and
Engagement/Vigilance Index (EVI) features to a pseudo-RGB image
representation suitable for CNN input.

Image construction:
    R channel = DE features (differential entropy per band per channel)
    G channel = PSD features (power spectral density per band per channel)
    B channel = EVI features (engagement/vigilance index per band per channel)

Each "pixel" corresponds to a (frequency band x EEG channel) combination.
Image shape: (n_channels * n_bands, n_time_windows, 3) for CNN input.

Frequency bands (5 standard):
    delta  (1-4 Hz)
    theta  (4-8 Hz)
    alpha  (8-13 Hz)
    beta   (13-30 Hz)
    gamma  (30-45 Hz)

EEG channels (4 for Muse 2):
    TP9, AF7, AF8, TP10

Reference:
    Zhong et al., "EEG-Based Emotion Recognition Using Regularized Graph
    Neural Networks" (2020) -- DE features for emotion classification.
    Li et al., "Exploring EEG Features in Cross-Subject Emotion Recognition"
    (2018) -- DE + PSD feature engineering.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

BAND_NAMES: List[str] = list(FREQUENCY_BANDS.keys())
N_BANDS: int = len(BAND_NAMES)

CHANNEL_NAMES: List[str] = ["TP9", "AF7", "AF8", "TP10"]
N_CHANNELS: int = len(CHANNEL_NAMES)

# Image row count: one row per (channel, band) pair
N_ROWS: int = N_CHANNELS * N_BANDS  # 4 * 5 = 20


# ── Feature computation functions ────────────────────────────────────────────


def compute_de_features(
    signals: np.ndarray,
    fs: float = 256.0,
) -> np.ndarray:
    """Compute Differential Entropy (DE) features per band per channel.

    DE for a Gaussian signal X with variance sigma^2:
        DE(X) = 0.5 * ln(2 * pi * e * sigma^2)

    In practice we estimate variance from the bandpass-filtered signal
    within each frequency band using Welch PSD integration.

    Args:
        signals: (n_channels, n_samples) raw EEG array.
        fs:      Sampling rate in Hz.

    Returns:
        de: (n_channels, n_bands) DE values.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    n_ch, n_samples = signals.shape
    de = np.zeros((n_ch, N_BANDS), dtype=np.float64)

    for ch_idx in range(n_ch):
        psd_freqs, psd_vals = _welch_psd(signals[ch_idx], fs)
        for b_idx, band_name in enumerate(BAND_NAMES):
            flo, fhi = FREQUENCY_BANDS[band_name]
            mask = (psd_freqs >= flo) & (psd_freqs <= fhi)
            if mask.any():
                band_power = float(np.mean(psd_vals[mask]))
                # DE = 0.5 * ln(2 * pi * e * sigma^2)
                # band_power approximates sigma^2 for that band
                de[ch_idx, b_idx] = 0.5 * np.log(
                    2.0 * np.pi * np.e * max(band_power, 1e-12)
                )
            else:
                de[ch_idx, b_idx] = 0.0

    return de


def compute_psd_features(
    signals: np.ndarray,
    fs: float = 256.0,
) -> np.ndarray:
    """Compute Power Spectral Density (PSD) features per band per channel.

    Uses Welch's method to estimate PSD, then averages within each band.

    Args:
        signals: (n_channels, n_samples) raw EEG array.
        fs:      Sampling rate in Hz.

    Returns:
        psd: (n_channels, n_bands) mean PSD per band in log scale.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    n_ch, n_samples = signals.shape
    psd_out = np.zeros((n_ch, N_BANDS), dtype=np.float64)

    for ch_idx in range(n_ch):
        psd_freqs, psd_vals = _welch_psd(signals[ch_idx], fs)
        for b_idx, band_name in enumerate(BAND_NAMES):
            flo, fhi = FREQUENCY_BANDS[band_name]
            mask = (psd_freqs >= flo) & (psd_freqs <= fhi)
            if mask.any():
                mean_power = float(np.mean(psd_vals[mask]))
                psd_out[ch_idx, b_idx] = np.log(max(mean_power, 1e-12))
            else:
                psd_out[ch_idx, b_idx] = 0.0

    return psd_out


def compute_evi_features(
    signals: np.ndarray,
    fs: float = 256.0,
) -> np.ndarray:
    """Compute Engagement/Vigilance Index (EVI) features per band per channel.

    EVI = beta / (alpha + theta) is a standard engagement index.
    We compute a per-band variant: for each band b, the EVI score is
    the ratio of that band's power to the sum of alpha + theta power,
    providing a normalized engagement profile across the spectrum.

    For alpha and theta bands themselves, EVI is the inverse ratio,
    capturing how much they dominate relative to beta.

    Args:
        signals: (n_channels, n_samples) raw EEG array.
        fs:      Sampling rate in Hz.

    Returns:
        evi: (n_channels, n_bands) engagement index features.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    n_ch, _ = signals.shape
    evi = np.zeros((n_ch, N_BANDS), dtype=np.float64)

    for ch_idx in range(n_ch):
        psd_freqs, psd_vals = _welch_psd(signals[ch_idx], fs)

        # Compute raw band powers
        band_powers = {}
        for band_name in BAND_NAMES:
            flo, fhi = FREQUENCY_BANDS[band_name]
            mask = (psd_freqs >= flo) & (psd_freqs <= fhi)
            band_powers[band_name] = (
                float(np.mean(psd_vals[mask])) if mask.any() else 1e-12
            )

        # Denominator: alpha + theta (engagement baseline)
        denom = max(
            band_powers["alpha"] + band_powers["theta"], 1e-12
        )

        for b_idx, band_name in enumerate(BAND_NAMES):
            # EVI ratio: band_power / (alpha + theta)
            ratio = band_powers[band_name] / denom
            # Log-scale for better dynamic range
            evi[ch_idx, b_idx] = np.log(max(ratio, 1e-12))

    return evi


def create_pseudo_rgb(
    de_features: np.ndarray,
    psd_features: np.ndarray,
    evi_features: np.ndarray,
) -> np.ndarray:
    """Create a pseudo-RGB image from DE, PSD, and EVI feature matrices.

    Stacks three feature types as R, G, B channels. Each row of the image
    corresponds to one (channel, band) combination. For a single time window,
    the image is (n_channels * n_bands, 1, 3).

    For multiple time windows (from sliding window extraction), the image
    becomes (n_channels * n_bands, n_time_windows, 3).

    Args:
        de_features:  (n_channels, n_bands) or (n_time_windows, n_channels, n_bands)
        psd_features: Same shape as de_features.
        evi_features: Same shape as de_features.

    Returns:
        image: (n_rows, n_time_windows, 3) pseudo-RGB image.
               n_rows = n_channels * n_bands.
    """
    de = np.asarray(de_features, dtype=np.float64)
    psd = np.asarray(psd_features, dtype=np.float64)
    evi = np.asarray(evi_features, dtype=np.float64)

    if de.ndim == 2:
        # Single time window: (n_ch, n_bands) -> (n_ch * n_bands, 1)
        r = de.flatten()[:, np.newaxis]
        g = psd.flatten()[:, np.newaxis]
        b = evi.flatten()[:, np.newaxis]
    elif de.ndim == 3:
        # Multiple time windows: (n_windows, n_ch, n_bands)
        n_windows = de.shape[0]
        r = de.reshape(n_windows, -1).T  # (n_ch*n_bands, n_windows)
        g = psd.reshape(n_windows, -1).T
        b = evi.reshape(n_windows, -1).T
    else:
        raise ValueError(
            f"Feature arrays must be 2-D or 3-D, got {de.ndim}-D"
        )

    # Normalize each channel to [0, 1] for image-like representation
    r = _min_max_normalize(r)
    g = _min_max_normalize(g)
    b = _min_max_normalize(b)

    # Stack: (n_rows, n_time_windows, 3)
    image = np.stack([r, g, b], axis=-1)
    return image


def prepare_cnn_kan_input(
    signals: np.ndarray,
    fs: float = 256.0,
    window_seconds: float = 4.0,
    overlap: float = 0.5,
) -> Dict[str, Any]:
    """Full pipeline: raw EEG -> pseudo-RGB image ready for CNN-KAN input.

    Segments the EEG signal into overlapping windows, extracts DE, PSD,
    and EVI features per window, then constructs the pseudo-RGB image.

    Args:
        signals:        (n_channels, n_samples) raw EEG.
        fs:             Sampling rate in Hz.
        window_seconds: Window length in seconds for feature extraction.
        overlap:        Overlap fraction between windows (0.0 to 0.9).

    Returns:
        dict with:
            image:        (n_rows, n_time_windows, 3) pseudo-RGB array.
            n_rows:       Number of rows (channels * bands).
            n_windows:    Number of time windows extracted.
            n_channels:   EEG channel count.
            n_bands:      Frequency band count.
            shape:        Tuple of image dimensions.
            window_seconds: Window size used.
            overlap:      Overlap fraction used.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    n_ch, n_samples = signals.shape
    window_samples = int(window_seconds * fs)
    hop_samples = max(1, int(window_samples * (1.0 - overlap)))

    # Extract windowed features
    de_list = []
    psd_list = []
    evi_list = []

    start = 0
    while start + window_samples <= n_samples:
        window = signals[:, start : start + window_samples]
        de_list.append(compute_de_features(window, fs))
        psd_list.append(compute_psd_features(window, fs))
        evi_list.append(compute_evi_features(window, fs))
        start += hop_samples

    # If signal is shorter than one window, use entire signal
    if len(de_list) == 0:
        de_list.append(compute_de_features(signals, fs))
        psd_list.append(compute_psd_features(signals, fs))
        evi_list.append(compute_evi_features(signals, fs))

    # Stack into (n_windows, n_ch, n_bands)
    de_all = np.array(de_list)
    psd_all = np.array(psd_list)
    evi_all = np.array(evi_list)

    # Create pseudo-RGB image
    image = create_pseudo_rgb(de_all, psd_all, evi_all)

    return {
        "image": image,
        "n_rows": image.shape[0],
        "n_windows": image.shape[1],
        "n_channels": n_ch,
        "n_bands": N_BANDS,
        "shape": image.shape,
        "window_seconds": window_seconds,
        "overlap": overlap,
    }


def feature_stats_to_dict(
    de_features: np.ndarray,
    psd_features: np.ndarray,
    evi_features: np.ndarray,
) -> Dict[str, Any]:
    """Summarize feature statistics for API reporting.

    Args:
        de_features:  (n_channels, n_bands) DE feature array.
        psd_features: (n_channels, n_bands) PSD feature array.
        evi_features: (n_channels, n_bands) EVI feature array.

    Returns:
        dict with per-feature-type statistics (mean, std, min, max)
        and per-band breakdowns.
    """
    de = np.asarray(de_features, dtype=np.float64)
    psd = np.asarray(psd_features, dtype=np.float64)
    evi = np.asarray(evi_features, dtype=np.float64)

    def _stats(arr: np.ndarray, name: str) -> Dict[str, Any]:
        return {
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_std": float(np.std(arr)),
            f"{name}_min": float(np.min(arr)),
            f"{name}_max": float(np.max(arr)),
        }

    result: Dict[str, Any] = {}
    result.update(_stats(de, "de"))
    result.update(_stats(psd, "psd"))
    result.update(_stats(evi, "evi"))

    # Per-band breakdown (averaged across channels)
    n_ch = de.shape[0]
    bands_info = {}
    for b_idx, band_name in enumerate(BAND_NAMES):
        bands_info[band_name] = {
            "de_mean": float(np.mean(de[:, b_idx])),
            "psd_mean": float(np.mean(psd[:, b_idx])),
            "evi_mean": float(np.mean(evi[:, b_idx])),
        }
    result["bands"] = bands_info
    result["n_channels"] = n_ch
    result["n_bands"] = N_BANDS
    result["band_names"] = BAND_NAMES
    result["channel_names"] = CHANNEL_NAMES[:n_ch]

    return result


# ── Internal helpers ─────────────────────────────────────────────────────────


def _welch_psd(
    signal: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PSD using Welch's method (scipy) or FFT fallback.

    Returns:
        (frequencies, psd_values) arrays.
    """
    n = len(signal)
    try:
        from scipy.signal import welch

        nperseg = min(n, int(fs * 2))
        nperseg = max(nperseg, 4)  # minimum segment length
        freqs, psd_vals = welch(signal, fs=fs, nperseg=nperseg)
        return freqs, psd_vals
    except ImportError:
        # Fallback to FFT-based PSD
        fft_vals = np.abs(np.fft.rfft(signal)) ** 2 / max(n, 1)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        return freqs, fft_vals


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    denom = arr_max - arr_min
    if denom < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr_min) / denom
