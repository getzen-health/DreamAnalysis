"""PPG/HRV feature extraction for Muse 2 ANCILLARY sensor.

Muse 2 has a built-in PPG (photoplethysmography) sensor accessible via
BrainFlow ANCILLARY preset. This module extracts HRV features for stress
and emotion detection (F1=0.876 on WESAD, orthogonal to EEG artifacts).
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Dict, Optional

log = logging.getLogger(__name__)

FS_PPG = 64  # Muse 2 PPG sampling rate (Hz)


def _find_peaks(signal: np.ndarray, min_distance_samples: int = 20, threshold_factor: float = 0.5) -> np.ndarray:
    """Detect R-peaks (local maxima) in PPG signal."""
    threshold = threshold_factor * (signal.max() - signal.min()) + signal.min()
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
            if not peaks or (i - peaks[-1]) >= min_distance_samples:
                peaks.append(i)
    return np.array(peaks, dtype=np.int64)


def extract_hrv_features(ppg_signal: np.ndarray, fs: float = FS_PPG) -> Dict[str, float]:
    """Extract HRV features from raw PPG signal.

    Args:
        ppg_signal: 1D PPG signal (samples,), raw from BrainFlow
        fs: sampling rate in Hz

    Returns:
        dict with HRV metrics:
          - mean_hr: mean heart rate (BPM)
          - sdnn: std of RR intervals (ms) — global HRV
          - rmssd: root mean square of successive RR differences (ms) — parasympathetic
          - pnn50: % successive differences > 50ms — vagal tone
          - lf_power: low-frequency power (0.04-0.15 Hz) — sympathetic
          - hf_power: high-frequency power (0.15-0.40 Hz) — parasympathetic
          - lf_hf_ratio: sympathovagal balance (stress indicator)
          - stress_index: derived stress score 0-1
    """
    if len(ppg_signal) < int(fs * 10):
        log.debug("PPG too short (%d samples) for reliable HRV", len(ppg_signal))
        return _empty_hrv()

    # Normalize
    sig = ppg_signal.astype(np.float64)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)

    # Find peaks (R-peaks equivalent)
    min_rr_samples = int(fs * 0.4)  # 40 BPM max
    peaks = _find_peaks(sig, min_distance_samples=min_rr_samples)

    if len(peaks) < 4:
        log.debug("Too few PPG peaks (%d) for HRV", len(peaks))
        return _empty_hrv()

    # RR intervals in ms
    rr_ms = np.diff(peaks) / fs * 1000.0

    # Remove physiological outliers (250–2000 ms = 30–240 BPM)
    rr_ms = rr_ms[(rr_ms >= 250) & (rr_ms <= 2000)]

    if len(rr_ms) < 3:
        return _empty_hrv()

    mean_hr = 60000.0 / rr_ms.mean()
    sdnn = rr_ms.std()
    successive_diff = np.diff(rr_ms)
    rmssd = np.sqrt((successive_diff ** 2).mean())
    pnn50 = float((np.abs(successive_diff) > 50).sum()) / len(successive_diff) * 100.0

    # Frequency domain (Lomb-Scargle for unevenly spaced RR)
    lf_power, hf_power, lf_hf_ratio = _frequency_domain_hrv(rr_ms, fs)

    # Stress index: high LF/HF, low RMSSD, high HR → stress
    # Normalized to 0-1 using population ranges
    hr_stress = np.clip((mean_hr - 60) / 60, 0, 1)        # 60 BPM=0, 120 BPM=1
    rmssd_calm = np.clip(rmssd / 80.0, 0, 1)              # 80ms=calm, 0ms=stressed
    lf_hf_stress = np.clip((lf_hf_ratio - 1.0) / 4.0, 0, 1)  # 1.0=balanced, 5.0=stressed
    stress_index = float(np.clip(0.35 * hr_stress + 0.35 * (1 - rmssd_calm) + 0.30 * lf_hf_stress, 0, 1))

    return {
        "mean_hr": round(float(mean_hr), 2),
        "sdnn": round(float(sdnn), 2),
        "rmssd": round(float(rmssd), 2),
        "pnn50": round(float(pnn50), 2),
        "lf_power": round(float(lf_power), 4),
        "hf_power": round(float(hf_power), 4),
        "lf_hf_ratio": round(float(lf_hf_ratio), 3),
        "stress_index": round(stress_index, 3),
        "n_rr_intervals": len(rr_ms),
    }


def _frequency_domain_hrv(rr_ms: np.ndarray, fs: float) -> tuple:
    """Estimate LF/HF power via FFT on interpolated RR series."""
    try:
        # Resample to 4 Hz grid (standard for HRV frequency analysis)
        target_fs = 4.0
        n_samples = max(64, int(len(rr_ms) * target_fs * rr_ms.mean() / 1000))
        # Interpolate
        t = np.cumsum(rr_ms) / 1000.0  # seconds
        t_uniform = np.linspace(t[0], t[-1], n_samples)
        rr_interp = np.interp(t_uniform, t, rr_ms)
        rr_interp -= rr_interp.mean()  # detrend

        freqs = np.fft.rfftfreq(n_samples, d=1.0/target_fs)
        psd = np.abs(np.fft.rfft(rr_interp)) ** 2

        freq_res = freqs[1] - freqs[0]
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

        lf_power = float(psd[lf_mask].sum() * freq_res)
        hf_power = float(psd[hf_mask].sum() * freq_res)
        lf_hf = lf_power / (hf_power + 1e-10)

        return lf_power, hf_power, min(lf_hf, 20.0)
    except Exception as e:
        log.debug("HRV frequency analysis failed: %s", e)
        return 0.0, 0.0, 1.0


def _empty_hrv() -> Dict[str, float]:
    return {
        "mean_hr": 0.0, "sdnn": 0.0, "rmssd": 0.0, "pnn50": 0.0,
        "lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 0.0,
        "stress_index": 0.0, "n_rr_intervals": 0,
    }
