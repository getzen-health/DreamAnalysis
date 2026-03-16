"""PPG signal processing for Muse 2 ANCILLARY sensor.

Muse 2 PPG: photoplethysmography from the forehead, enabled via
board.config_board("p50") in BrainFlow. Sampling rate: 64 Hz.

Pipeline:
  1. Detrend (remove DC offset and slow baseline wander)
  2. Bandpass filter 0.5-5 Hz (covers heart rate 30-300 BPM)
  3. Peak detection (systolic peaks)
  4. Heart rate from inter-beat intervals
  5. HRV metrics: SDNN, RMSSD, pNN50 (time-domain); LF/HF (frequency-domain)
  6. Respiratory rate from PPG amplitude modulation (0.1-0.5 Hz)
  7. Signal quality index (0-1)

Scientific basis:
- Task Force ESC/NASPE (1996): standard HRV frequency bands definition
- Clifton & Clifton (2012): PPG signal quality indices
- Schafer & Kratky (2008): respiratory rate from PPG amplitude envelope
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis

log = logging.getLogger(__name__)

# Muse 2 PPG sampling rate via BrainFlow ANCILLARY preset
FS_PPG: float = 64.0

# Physiological bounds
_MIN_IBI_MS: float = 250.0   # 240 BPM max
_MAX_IBI_MS: float = 2000.0  # 30 BPM min


# ── Preprocessing ─────────────────────────────────────────────────────────────

def detrend_ppg(signal: np.ndarray) -> np.ndarray:
    """Remove DC offset and linear drift from PPG signal.

    Uses scipy's detrend (least-squares linear fit removal).
    Equivalent to high-pass filtering at ~0 Hz.
    """
    if len(signal) < 2:
        return signal.astype(np.float64)
    return scipy_signal.detrend(signal.astype(np.float64), type="linear")


def bandpass_ppg(
    signal: np.ndarray,
    fs: float = FS_PPG,
    lowcut: float = 0.5,
    highcut: float = 5.0,
    order: int = 4,
) -> np.ndarray:
    """Bandpass filter PPG signal to isolate cardiac pulse (0.5-5 Hz).

    0.5 Hz low-cut: removes respiration and slow baseline wander.
    5.0 Hz high-cut: removes motion artifacts and high-frequency noise.
    Zero-phase (filtfilt) to avoid phase distortion.

    Args:
        signal: 1D PPG array (float).
        fs: Sampling frequency in Hz.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered signal (same length).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    if low >= high or low <= 0:
        return signal
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    padlen = 3 * max(len(a), len(b)) - 1
    if len(signal) <= padlen:
        return signal
    return scipy_signal.filtfilt(b, a, signal)


def preprocess_ppg(raw_ppg: np.ndarray, fs: float = FS_PPG) -> np.ndarray:
    """Full PPG preprocessing: detrend then bandpass 0.5-5 Hz.

    Args:
        raw_ppg: Raw PPG samples from BrainFlow (1D array).
        fs: Sampling rate in Hz.

    Returns:
        Preprocessed signal ready for peak detection.
    """
    sig = detrend_ppg(raw_ppg)
    sig = bandpass_ppg(sig, fs=fs, lowcut=0.5, highcut=5.0)
    return sig


# ── Peak detection ────────────────────────────────────────────────────────────

def detect_systolic_peaks(
    ppg_signal: np.ndarray,
    fs: float = FS_PPG,
) -> np.ndarray:
    """Detect systolic (upstroke) peaks in a preprocessed PPG signal.

    Uses scipy.signal.find_peaks with:
    - Minimum distance: 0.4 s (physiological limit, ~150 BPM)
    - Adaptive height threshold: 0.3 standard deviations above mean

    Args:
        ppg_signal: Preprocessed (bandpassed, detrended) PPG signal.
        fs: Sampling rate in Hz.

    Returns:
        Array of peak sample indices (int64). Empty if too few detected.
    """
    if len(ppg_signal) < 3:
        return np.array([], dtype=np.int64)

    std = float(np.std(ppg_signal))
    if std < 1e-10:
        return np.array([], dtype=np.int64)

    sig_norm = (ppg_signal - ppg_signal.mean()) / std

    # Minimum inter-peak distance = 0.4 s (caps at 150 BPM)
    min_distance = max(1, int(fs * 0.4))

    peaks, _ = scipy_signal.find_peaks(
        sig_norm,
        distance=min_distance,
        height=0.3,
    )
    return peaks.astype(np.int64)


def compute_ibi(peaks: np.ndarray, fs: float = FS_PPG) -> np.ndarray:
    """Compute inter-beat intervals (IBI) in milliseconds from peak indices.

    Removes physiologically implausible intervals outside 250-2000 ms.

    Args:
        peaks: Sample indices of detected peaks.
        fs: Sampling rate in Hz.

    Returns:
        Array of valid IBI values in milliseconds.
    """
    if len(peaks) < 2:
        return np.array([], dtype=np.float64)
    ibi_ms = np.diff(peaks).astype(np.float64) / fs * 1000.0
    return ibi_ms[(ibi_ms >= _MIN_IBI_MS) & (ibi_ms <= _MAX_IBI_MS)]


# ── Heart rate ────────────────────────────────────────────────────────────────

def compute_heart_rate(ibi_ms: np.ndarray) -> float:
    """Compute mean heart rate in BPM from IBI array.

    Returns 0.0 if fewer than 2 valid intervals.
    """
    if len(ibi_ms) < 2:
        return 0.0
    return float(60000.0 / ibi_ms.mean())


# ── HRV time-domain ───────────────────────────────────────────────────────────

def compute_sdnn(ibi_ms: np.ndarray) -> float:
    """SDNN: standard deviation of NN intervals (ms).

    Global HRV measure. Normal resting: 50-100 ms.
    Low (<20 ms) indicates reduced autonomic tone or stress.
    """
    if len(ibi_ms) < 2:
        return 0.0
    return float(np.std(ibi_ms, ddof=1))


def compute_rmssd(ibi_ms: np.ndarray) -> float:
    """RMSSD: root mean square of successive differences (ms).

    Primary parasympathetic (vagal) HRV index.
    Normal resting: 20-60 ms. Low = high sympathetic tone.
    """
    if len(ibi_ms) < 2:
        return 0.0
    successive = np.diff(ibi_ms)
    if len(successive) == 0:
        return 0.0
    return float(np.sqrt(np.mean(successive ** 2)))


def compute_pnn50(ibi_ms: np.ndarray) -> float:
    """pNN50: proportion of successive IBI differences > 50 ms.

    Returns value in [0, 1] (fraction, not percent).
    Strong parasympathetic indicator. Decreases with stress and age.
    """
    if len(ibi_ms) < 2:
        return 0.0
    successive = np.abs(np.diff(ibi_ms))
    if len(successive) == 0:
        return 0.0
    return float((successive > 50.0).sum()) / len(successive)


def compute_time_domain_hrv(ibi_ms: np.ndarray) -> Dict[str, float]:
    """Compute all time-domain HRV metrics from IBI array.

    Returns:
        dict with keys: mean_hr, sdnn, rmssd, pnn50, n_beats.
    """
    if len(ibi_ms) < 2:
        return {"mean_hr": 0.0, "sdnn": 0.0, "rmssd": 0.0, "pnn50": 0.0, "n_beats": 0}
    return {
        "mean_hr": round(compute_heart_rate(ibi_ms), 2),
        "sdnn": round(compute_sdnn(ibi_ms), 2),
        "rmssd": round(compute_rmssd(ibi_ms), 2),
        "pnn50": round(compute_pnn50(ibi_ms), 4),
        "n_beats": int(len(ibi_ms) + 1),  # n_beats = n_intervals + 1
    }


# ── HRV frequency-domain ──────────────────────────────────────────────────────

def compute_frequency_domain_hrv(
    ibi_ms: np.ndarray,
    resample_fs: float = 4.0,
) -> Tuple[float, float, float]:
    """Estimate LF and HF HRV power via FFT on uniformly resampled RR series.

    Standard frequency bands (Task Force ESC/NASPE, 1996):
      VLF: 0.003-0.04 Hz (vasomotor, thermoregulation — excluded, needs long recordings)
      LF:  0.04-0.15 Hz  (sympathetic + parasympathetic; baroreflex)
      HF:  0.15-0.40 Hz  (parasympathetic / vagal; respiratory sinus arrhythmia)

    LF/HF ratio:
      > 2.0 → sympathetic dominance (stress, exertion)
      0.5-2.0 → balanced autonomic tone
      < 0.5 → parasympathetic dominance (calm, recovery, sleep)

    Args:
        ibi_ms: Valid IBI array in milliseconds (≥8 intervals for frequency resolution).
        resample_fs: Uniform resampling rate (4 Hz is the standard).

    Returns:
        Tuple (lf_power, hf_power, lf_hf_ratio). All zero if computation fails.
    """
    if len(ibi_ms) < 8:
        return 0.0, 0.0, 1.0

    try:
        # Build time axis (seconds) from cumulative IBIs
        t = np.cumsum(ibi_ms) / 1000.0
        total_duration = t[-1] - t[0]
        if total_duration < 10.0:  # need at least ~10 s for LF resolution
            return 0.0, 0.0, 1.0

        n_samples = max(64, int(total_duration * resample_fs))
        t_uniform = np.linspace(t[0], t[-1], n_samples)
        ibi_interp = np.interp(t_uniform, t, ibi_ms)
        ibi_interp = ibi_interp - ibi_interp.mean()  # remove mean (detrend)

        freqs = np.fft.rfftfreq(n_samples, d=1.0 / resample_fs)
        psd = np.abs(np.fft.rfft(ibi_interp)) ** 2
        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

        lf_power = float(psd[lf_mask].sum() * freq_res)
        hf_power = float(psd[hf_mask].sum() * freq_res)
        lf_hf_ratio = min(lf_power / (hf_power + 1e-10), 20.0)

        return lf_power, hf_power, float(lf_hf_ratio)
    except Exception as exc:
        log.debug("HRV frequency analysis failed: %s", exc)
        return 0.0, 0.0, 1.0


# ── Respiratory rate ──────────────────────────────────────────────────────────

def compute_respiratory_rate(
    ppg_signal: np.ndarray,
    fs: float = FS_PPG,
) -> float:
    """Estimate respiratory rate from PPG amplitude modulation.

    Respiratory activity modulates the peak-to-peak amplitude of the PPG
    waveform at the breathing rate (0.1-0.5 Hz = 6-30 breaths/min).

    Method:
      1. Extract amplitude envelope via peak detection on the raw PPG
      2. Compute PSD of the envelope (Welch)
      3. Find dominant frequency in the respiratory band (0.1-0.5 Hz)

    Args:
        ppg_signal: Preprocessed PPG signal (bandpassed 0.5-5 Hz).
        fs: Sampling rate in Hz.

    Returns:
        Respiratory rate in breaths per minute. 0.0 if estimation fails.
    """
    if len(ppg_signal) < int(fs * 15):  # need at least 15 s for 0.1 Hz resolution
        return 0.0

    try:
        # Extract peak amplitudes for envelope
        min_dist = max(1, int(fs * 0.4))  # ~150 BPM max
        peaks, props = scipy_signal.find_peaks(
            ppg_signal,
            distance=min_dist,
            height=0.0,
        )
        if len(peaks) < 6:
            return 0.0

        # Build amplitude envelope as time series at 4 Hz (matches HRV resampling)
        target_fs = 4.0
        t_peaks = peaks / fs
        amplitudes = ppg_signal[peaks]

        total_duration = t_peaks[-1] - t_peaks[0]
        if total_duration < 10.0:
            return 0.0

        n_samples = max(64, int(total_duration * target_fs))
        t_uniform = np.linspace(t_peaks[0], t_peaks[-1], n_samples)
        env = np.interp(t_uniform, t_peaks, amplitudes)
        env = env - env.mean()  # detrend

        # Welch PSD to find respiratory frequency
        nperseg = min(len(env), max(32, int(target_fs * 20)))  # 20-s segments
        freqs, psd = scipy_signal.welch(env, fs=target_fs, nperseg=nperseg)

        # Respiratory band: 0.1-0.5 Hz (6-30 breaths/min)
        resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
        if not resp_mask.any():
            return 0.0

        dominant_idx = np.argmax(psd[resp_mask])
        resp_freq = float(freqs[resp_mask][dominant_idx])

        return round(resp_freq * 60.0, 1)  # Hz → breaths/min

    except Exception as exc:
        log.debug("Respiratory rate estimation failed: %s", exc)
        return 0.0


# ── Signal quality index ──────────────────────────────────────────────────────

def compute_ppg_sqi(
    ppg_signal: np.ndarray,
    fs: float = FS_PPG,
) -> Dict[str, float]:
    """Compute PPG signal quality index (SQI).

    Combines four quality criteria (each 0-1):
      1. Perfusion index: ratio of AC to DC amplitude (higher = better contact)
      2. Kurtosis: PPG should be slightly peaked (kurtosis ~3-5 for clean signal)
      3. SNR proxy: ratio of cardiac-band energy to total signal energy
      4. Regularity: coefficient of variation of IBI (low = regular, high = noisy)

    Overall SQI is the geometric mean of the four components.

    Args:
        ppg_signal: Raw (or preprocessed) PPG signal.
        fs: Sampling rate in Hz.

    Returns:
        dict with 'sqi' (overall, 0-1) and component scores.
        sqi < 0.3 → poor quality (noisy/disconnected)
        sqi 0.3-0.6 → moderate quality
        sqi > 0.6 → good quality
    """
    if len(ppg_signal) < int(fs * 5):
        return {"sqi": 0.0, "perfusion": 0.0, "kurtosis_score": 0.0, "snr": 0.0, "regularity": 0.0}

    try:
        sig = ppg_signal.astype(np.float64)

        # 1. Perfusion index: AC amplitude / DC level
        ac = float(sig.max() - sig.min())
        dc = float(np.abs(sig.mean())) + 1e-10
        perfusion_raw = ac / dc
        perfusion = float(np.clip(perfusion_raw / 0.5, 0, 1))  # 0.5 = good contact

        # 2. Kurtosis score: clean PPG has kurtosis 3-6; artifacts push it high
        kurt = float(kurtosis(sig, fisher=False))  # Fisher=False: Gaussian=3
        if 2.0 <= kurt <= 6.0:
            kurtosis_score = 1.0
        elif kurt < 2.0:
            kurtosis_score = max(0.0, kurt / 2.0)
        else:
            kurtosis_score = max(0.0, 1.0 - (kurt - 6.0) / 10.0)

        # 3. SNR proxy: cardiac-band power / total power
        nyq = 0.5 * fs
        low = 0.5 / nyq
        high = min(5.0 / nyq, 0.99)
        b, a = scipy_signal.butter(4, [low, high], btype="band")
        padlen = 3 * max(len(a), len(b)) - 1
        if len(sig) > padlen:
            cardiac = scipy_signal.filtfilt(b, a, sig)
            cardiac_power = float(np.mean(cardiac ** 2))
            total_power = float(np.mean(sig ** 2))
            snr = float(np.clip(cardiac_power / (total_power + 1e-10), 0, 1))
        else:
            snr = 0.5  # insufficient data — neutral score

        # 4. Regularity from IBI coefficient of variation
        preprocessed = preprocess_ppg(sig, fs=fs)
        peaks = detect_systolic_peaks(preprocessed, fs=fs)
        ibi = compute_ibi(peaks, fs=fs)
        if len(ibi) >= 3:
            cv = float(np.std(ibi, ddof=1)) / (float(np.mean(ibi)) + 1e-10)
            regularity = float(np.clip(1.0 - cv / 0.3, 0, 1))  # CV=0.3 → score=0
        else:
            regularity = 0.0

        # Overall SQI: geometric mean of four components
        components = [perfusion, kurtosis_score, snr, regularity]
        # Avoid log(0): add small floor
        floored = [max(c, 0.01) for c in components]
        sqi = float(np.exp(np.mean(np.log(floored))))

        return {
            "sqi": round(sqi, 3),
            "perfusion": round(perfusion, 3),
            "kurtosis_score": round(kurtosis_score, 3),
            "snr": round(snr, 3),
            "regularity": round(regularity, 3),
        }

    except Exception as exc:
        log.debug("PPG SQI computation failed: %s", exc)
        return {"sqi": 0.0, "perfusion": 0.0, "kurtosis_score": 0.0, "snr": 0.0, "regularity": 0.0}


# ── Full pipeline ─────────────────────────────────────────────────────────────

def extract_hrv_features(ppg_signal: np.ndarray, fs: float = FS_PPG) -> Dict[str, float]:
    """Extract all HRV features from a raw PPG signal.

    Full pipeline:
      1. Preprocess (detrend + bandpass)
      2. Detect systolic peaks
      3. Compute IBI
      4. Time-domain HRV: mean_hr, SDNN, RMSSD, pNN50
      5. Frequency-domain HRV: LF, HF, LF/HF
      6. Signal quality index
      7. Stress index derived from HRV

    Requires at least 10 seconds of signal for reliable results.
    Frequency domain requires at least 30 seconds for LF resolution.

    Args:
        ppg_signal: Raw PPG samples (1D array). Muse 2: 64 Hz.
        fs: Sampling rate in Hz.

    Returns:
        Dict with all HRV metrics. All-zero dict returned for insufficient data.
    """
    if len(ppg_signal) < int(fs * 10):
        log.debug("PPG too short (%d samples) for reliable HRV; need >= %d", len(ppg_signal), int(fs * 10))
        return _empty_hrv()

    processed = preprocess_ppg(ppg_signal, fs=fs)
    peaks = detect_systolic_peaks(processed, fs=fs)
    ibi_ms = compute_ibi(peaks, fs=fs)

    if len(ibi_ms) < 3:
        log.debug("Too few valid IBI intervals (%d); need >= 3", len(ibi_ms))
        return _empty_hrv()

    td = compute_time_domain_hrv(ibi_ms)
    lf_power, hf_power, lf_hf_ratio = compute_frequency_domain_hrv(ibi_ms)
    sqi_info = compute_ppg_sqi(ppg_signal, fs=fs)
    resp_rate = compute_respiratory_rate(processed, fs=fs)

    # Stress index: high HR + low RMSSD + high LF/HF → stress
    hr_stress = float(np.clip((td["mean_hr"] - 60.0) / 60.0, 0.0, 1.0))
    rmssd_calm = float(np.clip(td["rmssd"] / 80.0, 0.0, 1.0))
    lf_hf_stress = float(np.clip((lf_hf_ratio - 1.0) / 4.0, 0.0, 1.0))
    stress_index = float(np.clip(
        0.35 * hr_stress + 0.35 * (1.0 - rmssd_calm) + 0.30 * lf_hf_stress,
        0.0, 1.0,
    ))

    return {
        "mean_hr": td["mean_hr"],
        "sdnn": td["sdnn"],
        "rmssd": td["rmssd"],
        "pnn50": td["pnn50"],
        "lf_power": round(lf_power, 6),
        "hf_power": round(hf_power, 6),
        "lf_hf_ratio": round(lf_hf_ratio, 3),
        "respiratory_rate": resp_rate,
        "stress_index": round(stress_index, 3),
        "sqi": sqi_info["sqi"],
        "n_beats": td["n_beats"],
        "n_rr_intervals": len(ibi_ms),
    }


def _empty_hrv() -> Dict[str, float]:
    return {
        "mean_hr": 0.0,
        "sdnn": 0.0,
        "rmssd": 0.0,
        "pnn50": 0.0,
        "lf_power": 0.0,
        "hf_power": 0.0,
        "lf_hf_ratio": 0.0,
        "respiratory_rate": 0.0,
        "stress_index": 0.0,
        "sqi": 0.0,
        "n_beats": 0,
        "n_rr_intervals": 0,
    }
