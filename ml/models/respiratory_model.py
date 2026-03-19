"""Passive respiratory biometrics from audio amplitude envelope.

Extracts breathing patterns from microphone audio by analysing the
low-frequency amplitude envelope (0.1-0.5 Hz band). No external audio
libraries required -- all processing uses numpy and scipy.

Respiratory features:
  - breaths_per_minute (respiratory rate)
  - inhalation_exhalation_ratio
  - breath_regularity (coefficient of variation of cycle durations)
  - sigh_detection (cycles with amplitude > 2x median)

Breathing-pattern classification:
  calm / normal / stressed / anxious / exercise

Emotion correlation:
  Builds per-user baselines and correlates respiratory pattern shifts
  with reported emotion data over time.

References:
  - Massaroni et al. (2019): Contact-Based Methods for Measuring
    Respiratory Rate. Sensors 19(4), 908.
  - Schulz et al. (2013): Respiratory modulation of startle. Psychophysiology.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    from scipy.signal import butter, filtfilt, find_peaks

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False
    log.warning(
        "scipy is not installed. respiratory_model will not function. "
        "Install with: pip install scipy"
    )


def _require_scipy() -> None:
    """Raise a clear error when scipy is not available."""
    if not _SCIPY_AVAILABLE:
        raise RuntimeError(
            "scipy is required for respiratory signal processing but is not "
            "installed. Run: pip install scipy"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Physiological respiratory frequency band (Hz)
RESP_LO_HZ = 0.1   # ~6 breaths/min
RESP_HI_HZ = 0.5   # ~30 breaths/min

# Sigh detection: amplitude must exceed this multiple of the median peak
SIGH_AMPLITUDE_FACTOR = 2.0

# Minimum number of detected breath cycles to produce features
MIN_BREATH_CYCLES = 2

# Breathing-state classification thresholds (breaths per minute)
_BPM_CALM_MAX = 10.0
_BPM_NORMAL_MAX = 18.0
_BPM_STRESSED_MAX = 24.0
_BPM_ANXIOUS_MAX = 30.0
# Above _BPM_ANXIOUS_MAX -> "exercise"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _amplitude_envelope(signal: np.ndarray, fs: float) -> np.ndarray:
    """Compute the amplitude envelope of a signal.

    Uses the absolute value followed by a low-pass smoothing filter
    to produce a clean envelope suitable for respiratory analysis.
    """
    _require_scipy()

    # Absolute value to rectify
    envelope = np.abs(signal).astype(np.float64)

    # Low-pass at 1 Hz to get a smooth envelope (well above resp band)
    nyq = fs / 2.0
    cutoff = min(1.0, nyq * 0.95)  # safety clamp
    if cutoff / nyq < 0.01:
        # Sample rate too low for meaningful filtering
        return envelope
    b, a = butter(2, cutoff / nyq, btype="low")
    return filtfilt(b, a, envelope)


def _bandpass_respiratory(signal: np.ndarray, fs: float,
                          lo: float = RESP_LO_HZ,
                          hi: float = RESP_HI_HZ) -> np.ndarray:
    """Bandpass-filter a signal to the respiratory frequency band.

    Args:
        signal: 1-D amplitude envelope.
        fs:     Sampling rate in Hz.
        lo:     Low cutoff in Hz (default 0.1).
        hi:     High cutoff in Hz (default 0.5).

    Returns:
        Filtered signal of the same length.
    """
    _require_scipy()

    nyq = fs / 2.0
    lo_norm = lo / nyq
    hi_norm = hi / nyq

    # Clamp to valid Butterworth range
    lo_norm = max(1e-4, min(lo_norm, 0.999))
    hi_norm = max(lo_norm + 1e-4, min(hi_norm, 0.999))

    b, a = butter(2, [lo_norm, hi_norm], btype="band")
    return filtfilt(b, a, signal)


def extract_respiratory_rate(
    amplitude: np.ndarray,
    fs: float,
) -> Dict[str, Any]:
    """Extract respiratory rate from an audio amplitude envelope.

    Args:
        amplitude: 1-D numpy array -- raw amplitude values from the
                   microphone (mono). Can be a raw waveform or a
                   pre-computed amplitude envelope.
        fs:        Sampling rate in Hz.

    Returns:
        Dict with keys:
            - respiratory_rate_bpm: estimated breaths per minute
            - peak_indices: sample indices of detected breath peaks
            - cycle_durations_s: duration of each breath cycle in seconds
            - filtered_envelope: the bandpass-filtered respiratory signal
            - error: set if computation fails
    """
    _require_scipy()

    amplitude = np.asarray(amplitude, dtype=np.float64).ravel()

    if len(amplitude) < int(fs * 4):
        return {"error": "signal_too_short", "min_duration_s": 4.0}

    # Step 1: compute amplitude envelope then bandpass to respiratory band
    envelope = _amplitude_envelope(amplitude, fs)
    resp_signal = _bandpass_respiratory(envelope, fs)

    # Step 2: find peaks (each peak = one inhalation maximum)
    # Minimum distance between peaks: corresponds to max 30 bpm
    min_dist_samples = int(fs * (60.0 / 35.0))  # slightly above max bpm
    min_dist_samples = max(1, min_dist_samples)

    peaks, properties = find_peaks(
        resp_signal,
        distance=min_dist_samples,
        prominence=np.std(resp_signal) * 0.2,
    )

    if len(peaks) < MIN_BREATH_CYCLES:
        return {
            "error": "insufficient_peaks",
            "peaks_found": int(len(peaks)),
            "min_required": MIN_BREATH_CYCLES,
        }

    # Step 3: compute cycle durations
    cycle_durations = np.diff(peaks) / fs  # seconds per cycle

    # Step 4: respiratory rate
    mean_cycle = float(np.mean(cycle_durations))
    bpm = 60.0 / mean_cycle if mean_cycle > 0 else 0.0

    return {
        "respiratory_rate_bpm": round(bpm, 2),
        "peak_indices": peaks.tolist(),
        "cycle_durations_s": [round(d, 3) for d in cycle_durations.tolist()],
        "filtered_envelope": resp_signal,
    }


def compute_respiratory_features(
    amplitude: np.ndarray,
    fs: float,
) -> Dict[str, Any]:
    """Compute detailed respiratory features from audio amplitude.

    Args:
        amplitude: 1-D numpy array of audio amplitude values.
        fs:        Sampling rate in Hz.

    Returns:
        Dict with keys:
            - breaths_per_minute
            - inhalation_exhalation_ratio
            - breath_regularity  (coefficient of variation -- lower is more regular)
            - sigh_count
            - sigh_indices   (which breath cycles were sighs)
            - mean_breath_amplitude
            - error: set on failure
    """
    _require_scipy()

    # Get base rate info first
    rate_info = extract_respiratory_rate(amplitude, fs)
    if "error" in rate_info:
        return rate_info

    peaks = np.array(rate_info["peak_indices"])
    cycle_durations = np.array(rate_info["cycle_durations_s"])
    resp_signal = rate_info["filtered_envelope"]

    # ---- Inhalation / exhalation ratio -----------------------------------
    # For each cycle, split at the peak to measure rising (inhale) vs
    # falling (exhale) durations.
    troughs, _ = find_peaks(-resp_signal, distance=max(1, int(fs * 1.5)))

    inhale_durations: List[float] = []
    exhale_durations: List[float] = []

    for i in range(len(peaks)):
        # Find the trough just before this peak (start of inhale)
        prior_troughs = troughs[troughs < peaks[i]]
        if len(prior_troughs) == 0:
            continue
        inhale_start = prior_troughs[-1]

        # Find the trough just after this peak (end of exhale)
        post_troughs = troughs[troughs > peaks[i]]
        if len(post_troughs) == 0:
            continue
        exhale_end = post_troughs[0]

        inhale_dur = (peaks[i] - inhale_start) / fs
        exhale_dur = (exhale_end - peaks[i]) / fs

        if inhale_dur > 0 and exhale_dur > 0:
            inhale_durations.append(inhale_dur)
            exhale_durations.append(exhale_dur)

    if len(inhale_durations) > 0 and len(exhale_durations) > 0:
        ie_ratio = float(np.mean(inhale_durations) / np.mean(exhale_durations))
    else:
        ie_ratio = 1.0  # fallback

    # ---- Breath regularity (CV of cycle durations) -----------------------
    if len(cycle_durations) >= 2:
        cv = float(np.std(cycle_durations) / np.mean(cycle_durations))
    else:
        cv = 0.0

    # ---- Sigh detection --------------------------------------------------
    # A sigh is a breath whose peak amplitude exceeds SIGH_AMPLITUDE_FACTOR
    # times the median peak amplitude.
    peak_amplitudes = resp_signal[peaks]
    median_amp = float(np.median(np.abs(peak_amplitudes)))
    sigh_mask = np.abs(peak_amplitudes) > (SIGH_AMPLITUDE_FACTOR * median_amp)
    sigh_indices = np.where(sigh_mask)[0].tolist()
    sigh_count = int(np.sum(sigh_mask))

    # ---- Mean breath amplitude -------------------------------------------
    mean_amp = float(np.mean(np.abs(peak_amplitudes)))

    return {
        "breaths_per_minute": rate_info["respiratory_rate_bpm"],
        "inhalation_exhalation_ratio": round(ie_ratio, 3),
        "breath_regularity": round(cv, 4),
        "sigh_count": sigh_count,
        "sigh_indices": sigh_indices,
        "mean_breath_amplitude": round(mean_amp, 6),
        "cycle_durations_s": rate_info["cycle_durations_s"],
        "peak_indices": rate_info["peak_indices"],
    }


def classify_breathing_pattern(features: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a breathing pattern into a physiological state.

    Args:
        features: Output of ``compute_respiratory_features()``.

    Returns:
        Dict with keys:
            - state: one of "calm", "normal", "stressed", "anxious", "exercise"
            - confidence: 0.0-1.0
            - contributing_factors: list of explanatory strings
    """
    if "error" in features:
        return {
            "state": "unknown",
            "confidence": 0.0,
            "contributing_factors": [features["error"]],
        }

    bpm = features["breaths_per_minute"]
    cv = features["breath_regularity"]
    ie_ratio = features["inhalation_exhalation_ratio"]
    sigh_count = features.get("sigh_count", 0)

    factors: List[str] = []
    confidence = 0.7  # base confidence

    # ---- Primary classification by respiratory rate ----------------------
    if bpm <= _BPM_CALM_MAX:
        state = "calm"
        factors.append(f"low respiratory rate ({bpm:.1f} bpm)")
    elif bpm <= _BPM_NORMAL_MAX:
        state = "normal"
        factors.append(f"normal respiratory rate ({bpm:.1f} bpm)")
    elif bpm <= _BPM_STRESSED_MAX:
        state = "stressed"
        factors.append(f"elevated respiratory rate ({bpm:.1f} bpm)")
    elif bpm <= _BPM_ANXIOUS_MAX:
        state = "anxious"
        factors.append(f"high respiratory rate ({bpm:.1f} bpm)")
    else:
        state = "exercise"
        factors.append(f"very high respiratory rate ({bpm:.1f} bpm)")

    # ---- Regularity adjustments ------------------------------------------
    if cv > 0.35:
        # Highly irregular breathing -> likely anxious or stressed
        if state in ("calm", "normal"):
            state = "stressed"
            factors.append(f"irregular breathing (CV={cv:.2f})")
        confidence = max(0.3, confidence - 0.15)
    elif cv < 0.10:
        # Very regular -> higher confidence in calm/normal
        if state in ("calm", "normal"):
            confidence = min(1.0, confidence + 0.1)
            factors.append(f"very regular breathing (CV={cv:.2f})")

    # ---- Sigh adjustments ------------------------------------------------
    if sigh_count >= 3:
        if state == "normal":
            state = "stressed"
        factors.append(f"frequent sighs ({sigh_count})")
        confidence = max(0.3, confidence - 0.1)

    # ---- I/E ratio adjustments -------------------------------------------
    if ie_ratio < 0.6:
        # Prolonged exhale -> likely deliberate calming
        factors.append(f"prolonged exhale (I:E={ie_ratio:.2f})")
        if state in ("normal", "stressed"):
            confidence = max(0.4, confidence - 0.05)
    elif ie_ratio > 1.5:
        # Short exhale, long inhale -> possible gasping / anxious
        factors.append(f"short exhale (I:E={ie_ratio:.2f})")
        if state in ("calm", "normal"):
            state = "stressed"

    return {
        "state": state,
        "confidence": round(min(1.0, max(0.0, confidence)), 2),
        "contributing_factors": factors,
    }


def compute_respiratory_emotion_correlation(
    respiratory_history: List[Dict[str, Any]],
    emotion_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Correlate respiratory patterns with emotion data over time.

    Builds a personal baseline by comparing respiratory features against
    concurrently reported or detected emotions.

    Args:
        respiratory_history: List of dicts from ``compute_respiratory_features()``.
            Each entry should also contain a ``"timestamp"`` key (epoch seconds).
        emotion_history: List of dicts with at least ``"valence"`` (-1..1),
            ``"arousal"`` (0..1), and ``"timestamp"`` (epoch seconds).

    Returns:
        Dict with:
            - bpm_valence_corr:  Pearson r between BPM and valence
            - bpm_arousal_corr:  Pearson r between BPM and arousal
            - regularity_valence_corr: Pearson r between CV and valence
            - n_matched_pairs: number of time-aligned data points used
            - baseline: personal baseline stats
            - error: set if insufficient data
    """
    if len(respiratory_history) < 3 or len(emotion_history) < 3:
        return {
            "error": "insufficient_history",
            "min_required": 3,
            "respiratory_count": len(respiratory_history),
            "emotion_count": len(emotion_history),
        }

    # ---- Time-align respiratory and emotion observations -----------------
    # Match each respiratory observation to the nearest emotion within 60s.
    bpm_vals: List[float] = []
    cv_vals: List[float] = []
    valence_vals: List[float] = []
    arousal_vals: List[float] = []

    emotion_ts = np.array([e.get("timestamp", 0.0) for e in emotion_history])

    for resp in respiratory_history:
        resp_ts = resp.get("timestamp", 0.0)
        if resp_ts == 0.0:
            continue

        bpm = resp.get("breaths_per_minute")
        cv = resp.get("breath_regularity")
        if bpm is None or cv is None:
            continue

        # Find closest emotion entry within 60 seconds
        diffs = np.abs(emotion_ts - resp_ts)
        closest_idx = int(np.argmin(diffs))
        if diffs[closest_idx] > 60.0:
            continue

        emo = emotion_history[closest_idx]
        v = emo.get("valence")
        a = emo.get("arousal")
        if v is None or a is None:
            continue

        bpm_vals.append(float(bpm))
        cv_vals.append(float(cv))
        valence_vals.append(float(v))
        arousal_vals.append(float(a))

    n_pairs = len(bpm_vals)
    if n_pairs < 3:
        return {
            "error": "insufficient_matched_pairs",
            "n_matched_pairs": n_pairs,
            "min_required": 3,
        }

    bpm_arr = np.array(bpm_vals)
    cv_arr = np.array(cv_vals)
    valence_arr = np.array(valence_vals)
    arousal_arr = np.array(arousal_vals)

    # ---- Pearson correlations --------------------------------------------
    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        mat = np.corrcoef(x, y)
        return float(mat[0, 1])

    bpm_valence_r = _safe_corr(bpm_arr, valence_arr)
    bpm_arousal_r = _safe_corr(bpm_arr, arousal_arr)
    cv_valence_r = _safe_corr(cv_arr, valence_arr)

    # ---- Personal baseline -----------------------------------------------
    baseline = {
        "mean_bpm": round(float(np.mean(bpm_arr)), 2),
        "std_bpm": round(float(np.std(bpm_arr)), 2),
        "mean_regularity": round(float(np.mean(cv_arr)), 4),
        "mean_valence": round(float(np.mean(valence_arr)), 3),
        "mean_arousal": round(float(np.mean(arousal_arr)), 3),
    }

    return {
        "bpm_valence_corr": round(bpm_valence_r, 4),
        "bpm_arousal_corr": round(bpm_arousal_r, 4),
        "regularity_valence_corr": round(cv_valence_r, 4),
        "n_matched_pairs": n_pairs,
        "baseline": baseline,
    }
