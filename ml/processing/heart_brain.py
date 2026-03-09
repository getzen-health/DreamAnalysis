"""Heartbeat-Evoked Potential (HEP) computation and heart-brain coupling.

HEP: EEG signal time-locked to heartbeat R-peaks.
- Epoch: -200 ms to +600 ms around each R-peak
- Baseline correction: subtract mean of -200 to -50 ms pre-peak window
- Average across >= 5 beats for stable HEP
- HEP amplitude at frontal sites (AF7/AF8) reflects interoceptive awareness

Scientific basis:
- Scientific Reports (2025): HEP amplitude increases during emotional awareness
  and meditation
- Delta-HRV coupling: delta power synchronization with heart rate reflects
  sleep staging (Communications Biology, 2025)
"""

import logging
import numpy as np
from typing import Dict, Optional

from processing.ppg_features import PPGFeatureExtractor

log = logging.getLogger(__name__)

_ppg_extractor = PPGFeatureExtractor()

# HEP window parameters
_PRE_MS_DEFAULT = 200.0
_POST_MS_DEFAULT = 600.0
_BASELINE_END_MS = -50.0  # baseline from -pre_ms to this offset before peak

# Downsampled waveform target rate (Hz) for compact JSON response
_WAVEFORM_FS = 100.0

# Minimum beats required for a meaningful HEP average
_MIN_BEATS_HEP = 5


def compute_hep(
    eeg_signals: np.ndarray,
    r_peak_times: np.ndarray,
    fs_eeg: float = 256.0,
    pre_ms: float = _PRE_MS_DEFAULT,
    post_ms: float = _POST_MS_DEFAULT,
) -> Dict:
    """Compute Heartbeat-Evoked Potential from EEG + R-peak times.

    Args:
        eeg_signals: (n_channels, n_samples) or (n_samples,) EEG array in uV
        r_peak_times: R-peak times in seconds
        fs_eeg: EEG sampling rate in Hz
        pre_ms: pre-R-peak window in ms (used for baseline)
        post_ms: post-R-peak window in ms

    Returns dict with:
    - hep_amplitude: mean absolute HEP amplitude across frontal channels (uV)
    - hep_peak_latency_ms: latency of peak HEP deflection (ms post-R-peak)
    - n_epochs_averaged: number of heartbeats averaged
    - hep_waveform: list of floats (averaged ERP waveform, at _WAVEFORM_FS Hz)
    - coupling_quality: 0-1 score (more beats = more reliable)
    """
    empty = _empty_hep()

    if len(r_peak_times) == 0:
        return empty

    # Ensure 2-D array: (n_channels, n_samples)
    if eeg_signals.ndim == 1:
        eeg = eeg_signals[np.newaxis, :]
    else:
        eeg = eeg_signals

    n_channels, n_samples = eeg.shape
    pre_samp = int(round(pre_ms / 1000.0 * fs_eeg))
    post_samp = int(round(post_ms / 1000.0 * fs_eeg))
    epoch_len = pre_samp + post_samp

    # Baseline: from start of epoch to 50 ms before peak
    baseline_end_samp = pre_samp - int(round(abs(_BASELINE_END_MS) / 1000.0 * fs_eeg))
    baseline_end_samp = max(1, baseline_end_samp)

    epochs = []  # list of (n_channels, epoch_len) arrays
    for t_peak in r_peak_times:
        center = int(round(t_peak * fs_eeg))
        start = center - pre_samp
        end = center + post_samp
        if start < 0 or end > n_samples:
            continue
        epoch = eeg[:, start:end].astype(np.float64)
        # Baseline correction: subtract mean of [-pre_ms, -50 ms]
        baseline_mean = epoch[:, :baseline_end_samp].mean(axis=1, keepdims=True)
        epoch = epoch - baseline_mean
        epochs.append(epoch)

    n_epochs = len(epochs)
    if n_epochs < 1:
        return empty

    # Average across epochs
    avg_epoch = np.mean(epochs, axis=0)  # (n_channels, epoch_len)

    # Use frontal channels (AF7=ch1, AF8=ch2) when available; fall back to all
    frontal_indices = [1, 2] if n_channels >= 3 else list(range(n_channels))
    frontal_avg = avg_epoch[frontal_indices, :].mean(axis=0)  # (epoch_len,)

    # HEP amplitude: mean absolute amplitude in post-peak window
    hep_amplitude = float(np.abs(frontal_avg[pre_samp:]).mean())

    # Peak latency in ms (relative to R-peak onset = pre_samp)
    peak_offset = int(np.argmax(np.abs(frontal_avg[pre_samp:])))
    hep_peak_latency_ms = float(peak_offset / fs_eeg * 1000.0)

    # Downsample waveform for compact response
    time_orig = np.linspace(-pre_ms, post_ms, epoch_len)
    n_out = max(2, int(round((pre_ms + post_ms) / 1000.0 * _WAVEFORM_FS)))
    time_out = np.linspace(-pre_ms, post_ms, n_out)
    waveform = np.interp(time_out, time_orig, frontal_avg).tolist()

    # coupling_quality: sigmoid-like score based on number of averaged beats
    # saturates at ~20 beats
    coupling_quality = float(min(1.0, n_epochs / 20.0))

    return {
        "hep_amplitude": round(hep_amplitude, 4),
        "hep_peak_latency_ms": round(hep_peak_latency_ms, 2),
        "n_epochs_averaged": n_epochs,
        "hep_waveform": [round(v, 4) for v in waveform],
        "coupling_quality": round(coupling_quality, 4),
    }


def compute_heart_brain_coupling(
    eeg_signals: np.ndarray,
    ppg_signal: np.ndarray,
    fs_eeg: float = 256.0,
    fs_ppg: float = 64.0,
) -> Dict:
    """Full heart-brain coupling analysis combining HEP and HRV.

    Returns merged dict from compute_hep + PPGFeatureExtractor.extract_hrv, plus:
    - delta_hrv_coupling: Pearson r between instantaneous delta power and
      inter-beat intervals (sleep staging biomarker)
    - interoceptive_index: 0-1 score combining HEP coupling_quality and HF HRV
    """
    extractor = PPGFeatureExtractor(fs=fs_ppg)

    # HRV features
    hrv = extractor.extract_hrv(ppg_signal)
    r_peak_times = extractor.get_r_peak_times(ppg_signal)

    # Ensure 2-D EEG
    eeg_arr = np.array(eeg_signals)
    if eeg_arr.ndim == 1:
        eeg_arr = eeg_arr[np.newaxis, :]

    # HEP features
    hep = compute_hep(eeg_arr, r_peak_times, fs_eeg=fs_eeg)

    # Delta-HRV coupling: correlate per-beat delta power with IBI
    delta_hrv_coupling = _compute_delta_hrv_coupling(
        eeg_arr, r_peak_times, hrv, fs_eeg
    )

    # Interoceptive index: blends HEP quality and vagal tone (HF HRV)
    hf_norm = float(np.clip(hrv.get("hf_power", 0.0) / (hrv.get("hf_power", 0.0) + 1e-6), 0, 1))
    interoceptive_index = float(
        np.clip(0.5 * hep["coupling_quality"] + 0.5 * hf_norm, 0.0, 1.0)
    )

    result = {**hep, **hrv}
    result["delta_hrv_coupling"] = round(float(delta_hrv_coupling), 4)
    result["interoceptive_index"] = round(interoceptive_index, 4)
    return result


# ── Private helpers ──────────────────────────────────────────────────────────

def _compute_delta_hrv_coupling(
    eeg: np.ndarray,
    r_peak_times: np.ndarray,
    hrv: Dict,
    fs_eeg: float,
) -> float:
    """Pearson correlation between per-beat delta power and IBI series.

    Uses AF7 (ch1) or first available channel.
    Returns 0.0 when there are fewer than 3 beats.
    """
    if len(r_peak_times) < 3:
        return 0.0

    try:
        from processing.eeg_processor import extract_band_powers, preprocess

        # Use frontal channel if available
        ch_idx = 1 if eeg.shape[0] > 1 else 0
        channel = eeg[ch_idx]

        ibi_ms = np.diff(r_peak_times) * 1000.0  # ms
        ibi_ms = ibi_ms[(ibi_ms >= 250) & (ibi_ms <= 2000)]
        if len(ibi_ms) < 3:
            return 0.0

        # Compute delta power for each inter-beat segment
        delta_powers = []
        for i in range(len(ibi_ms)):
            t_start = int(r_peak_times[i] * fs_eeg)
            t_end = int(r_peak_times[i + 1] * fs_eeg)
            if t_end > len(channel) or (t_end - t_start) < int(fs_eeg * 0.5):
                delta_powers.append(np.nan)
                continue
            seg = channel[t_start:t_end]
            try:
                seg_proc = preprocess(seg, fs_eeg)
                bp = extract_band_powers(seg_proc, fs_eeg)
                delta_powers.append(float(bp.get("delta", np.nan)))
            except Exception:
                delta_powers.append(np.nan)

        delta_arr = np.array(delta_powers)
        # Remove NaN pairs
        valid = ~np.isnan(delta_arr)
        if valid.sum() < 3:
            return 0.0
        corr = float(np.corrcoef(delta_arr[valid], ibi_ms[: valid.sum()])[0, 1])
        return corr if np.isfinite(corr) else 0.0
    except Exception as exc:
        log.debug("delta_hrv_coupling computation failed: %s", exc)
        return 0.0


def _empty_hep() -> Dict:
    return {
        "hep_amplitude": 0.0,
        "hep_peak_latency_ms": 0.0,
        "n_epochs_averaged": 0,
        "hep_waveform": [],
        "coupling_quality": 0.0,
    }
