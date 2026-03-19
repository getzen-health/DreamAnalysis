"""Voice biomarker depression/anxiety screening.

Extracts vocal biomarkers correlated with depression and anxiety, producing
PHQ-9 compatible depression risk scores (0-27) and GAD-7 compatible anxiety
risk scores (0-21) from voice features alone.

Based on:
    Sonde Health (2024): vocal biomarkers detect depression with >80% sensitivity
    Kintsugi (2025): 25s speech -> >70% depression screening accuracy
    JMIR Mental Health (2025): pause patterns strongest single predictor of depression
    Brain Sciences (2025): jitter has NEGATIVE relationship with anxiety
        (lower jitter = tighter vocal control = higher anxiety)

Key biomarkers:
    - Pitch variability (F0 std, range) -- reduced in depression
    - Speaking rate -- slowed in depression, variable in anxiety
    - Pause frequency/duration -- increased in depression (strongest predictor)
    - Jitter -- lower in anxiety (tighter vocal control)
    - Shimmer -- elevated in depression (vocal fold tension changes)
    - Formant stability -- reduced in anxiety (articulatory tension)
    - Energy contour -- flattened in depression, irregular in anxiety
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# PHQ-9 severity thresholds (standard clinical cutoffs)
_PHQ9_MINIMAL = 4
_PHQ9_MILD = 9
_PHQ9_MODERATE = 14
_PHQ9_MOD_SEVERE = 19
# 20-27 = severe

# GAD-7 severity thresholds
_GAD7_MINIMAL = 4
_GAD7_MILD = 9
_GAD7_MODERATE = 14
# 15-21 = severe


def _severity_label_phq9(score: float) -> str:
    """Map PHQ-9 score to severity label."""
    if score <= _PHQ9_MINIMAL:
        return "minimal"
    elif score <= _PHQ9_MILD:
        return "mild"
    elif score <= _PHQ9_MODERATE:
        return "moderate"
    elif score <= _PHQ9_MOD_SEVERE:
        return "moderately_severe"
    return "severe"


def _severity_label_gad7(score: float) -> str:
    """Map GAD-7 score to severity label."""
    if score <= _GAD7_MINIMAL:
        return "minimal"
    elif score <= _GAD7_MILD:
        return "mild"
    elif score <= _GAD7_MODERATE:
        return "moderate"
    return "severe"


def extract_vocal_biomarkers(audio: np.ndarray, sr: int = 16000) -> Dict:
    """Extract vocal biomarkers correlated with depression/anxiety.

    Extracts pitch variability, speaking rate, pause frequency, jitter,
    shimmer, formant stability, and energy contour features.

    Args:
        audio: 1-D float32 mono waveform.
        sr: Sample rate in Hz.

    Returns:
        Dict of biomarker values. Returns {"error": "..."} on failure.
    """
    if audio is None or len(audio) < sr * 2:
        return {"error": "insufficient_audio", "detail": "need at least 2 seconds"}

    result: Dict = {}

    # -- F0 (pitch) features via autocorrelation --------------------------
    f0_values = _estimate_f0_autocorr(audio, sr)
    f0_valid = f0_values[f0_values > 0]

    result["f0_mean"] = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
    result["f0_std"] = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0
    result["f0_range"] = float(np.ptp(f0_valid)) if len(f0_valid) > 0 else 0.0
    result["pitch_variability"] = result["f0_std"] / max(result["f0_mean"], 1e-9)

    # -- Jitter (cycle-to-cycle pitch perturbation) -----------------------
    result.update(_compute_jitter(f0_valid))

    # -- Shimmer (amplitude perturbation) ---------------------------------
    frame_len = int(sr * 0.025)  # 25ms frames
    hop = int(sr * 0.010)  # 10ms hop
    rms = _compute_rms(audio, frame_len, hop)
    result.update(_compute_shimmer(rms))

    # -- Pause metrics (strongest depression predictor) -------------------
    result.update(_extract_pause_metrics(audio, sr, rms))

    # -- Speaking rate ----------------------------------------------------
    result.update(_extract_speaking_rate(audio, sr, f0_values))

    # -- Formant stability ------------------------------------------------
    result.update(_compute_formant_stability(audio, sr))

    # -- Energy contour ---------------------------------------------------
    result["energy_mean"] = float(np.mean(rms)) if len(rms) > 0 else 0.0
    result["energy_std"] = float(np.std(rms)) if len(rms) > 0 else 0.0
    result["energy_range"] = float(np.ptp(rms)) if len(rms) > 0 else 0.0
    result["energy_slope"] = _compute_energy_slope(rms)

    return result


def score_depression_risk(biomarkers: Dict) -> Dict:
    """Compute PHQ-9 compatible depression risk score from vocal biomarkers.

    Maps voice features to a 0-27 scale compatible with PHQ-9 clinical scoring.
    Higher scores indicate higher depression risk.

    Args:
        biomarkers: Output of extract_vocal_biomarkers().

    Returns:
        Dict with phq9_score, severity, contributing_indicators, disclaimer.
    """
    if "error" in biomarkers:
        return {
            "phq9_score": 0.0,
            "severity": "unknown",
            "indicators": [],
            "disclaimer": _DISCLAIMER,
        }

    indicators: List[str] = []
    score_components: List[float] = []

    # 1. Reduced pitch variability (flat prosody) -- depression hallmark
    pv = biomarkers.get("pitch_variability", 0.0)
    if pv < 0.10:
        score_components.append(0.25)
        indicators.append("very_low_pitch_variability")
    elif pv < 0.15:
        score_components.append(0.15)
        indicators.append("low_pitch_variability")
    else:
        score_components.append(0.0)

    # 2. Increased pause frequency/duration -- strongest predictor
    pause_ratio = biomarkers.get("silence_ratio", 0.0)
    if pause_ratio > 0.45:
        score_components.append(0.30)
        indicators.append("excessive_pausing")
    elif pause_ratio > 0.30:
        score_components.append(0.18)
        indicators.append("elevated_pausing")
    else:
        score_components.append(0.0)

    # 3. Slowed speaking rate
    speaking_rate = biomarkers.get("speaking_rate", 0.0)
    if 0.0 < speaking_rate < 2.0:
        score_components.append(0.20)
        indicators.append("very_slow_speech")
    elif 0.0 < speaking_rate < 3.0:
        score_components.append(0.10)
        indicators.append("slow_speech")
    else:
        score_components.append(0.0)

    # 4. Elevated shimmer (vocal fold tension)
    shimmer = biomarkers.get("shimmer_local", 0.0)
    if shimmer > 0.15:
        score_components.append(0.15)
        indicators.append("elevated_shimmer")
    elif shimmer > 0.10:
        score_components.append(0.08)
        indicators.append("mild_shimmer_elevation")
    else:
        score_components.append(0.0)

    # 5. Flattened energy contour
    energy_range = biomarkers.get("energy_range", 0.0)
    if energy_range < 0.02:
        score_components.append(0.10)
        indicators.append("flat_energy_contour")
    else:
        score_components.append(0.0)

    # Combine: 0-1 risk then scale to PHQ-9 range (0-27)
    raw_risk = float(np.clip(sum(score_components), 0.0, 1.0))
    phq9_score = round(raw_risk * 27.0, 1)

    return {
        "phq9_score": phq9_score,
        "severity": _severity_label_phq9(phq9_score),
        "indicators": indicators,
        "disclaimer": _DISCLAIMER,
    }


def score_anxiety_risk(biomarkers: Dict) -> Dict:
    """Compute GAD-7 compatible anxiety risk score from vocal biomarkers.

    Maps voice features to a 0-21 scale compatible with GAD-7 clinical scoring.
    Higher scores indicate higher anxiety risk.

    Args:
        biomarkers: Output of extract_vocal_biomarkers().

    Returns:
        Dict with gad7_score, severity, contributing_indicators, disclaimer.
    """
    if "error" in biomarkers:
        return {
            "gad7_score": 0.0,
            "severity": "unknown",
            "indicators": [],
            "disclaimer": _DISCLAIMER,
        }

    indicators: List[str] = []
    score_components: List[float] = []

    # 1. Low jitter (tighter vocal control = anxiety)
    jitter = biomarkers.get("jitter_local", 0.0)
    if 0.0 < jitter < 0.005:
        score_components.append(0.25)
        indicators.append("very_low_jitter_tight_control")
    elif 0.0 < jitter < 0.01:
        score_components.append(0.12)
        indicators.append("low_jitter")
    else:
        score_components.append(0.0)

    # 2. Variable speaking rate (rushed/uneven pacing)
    rate_var = biomarkers.get("speaking_rate_variability", 0.0)
    if rate_var > 0.30:
        score_components.append(0.20)
        indicators.append("variable_speaking_rate")
    else:
        score_components.append(0.0)

    # 3. Reduced formant stability (articulatory tension)
    formant_stab = biomarkers.get("formant_stability", 1.0)
    if formant_stab < 0.5:
        score_components.append(0.20)
        indicators.append("low_formant_stability")
    elif formant_stab < 0.7:
        score_components.append(0.10)
        indicators.append("mild_formant_instability")
    else:
        score_components.append(0.0)

    # 4. Elevated pitch mean (voice tension)
    f0_mean = biomarkers.get("f0_mean", 0.0)
    if f0_mean > 220:
        score_components.append(0.15)
        indicators.append("elevated_pitch")
    else:
        score_components.append(0.0)

    # 5. Irregular energy contour
    energy_std = biomarkers.get("energy_std", 0.0)
    energy_mean = biomarkers.get("energy_mean", 1e-9)
    energy_cv = energy_std / max(energy_mean, 1e-9)
    if energy_cv > 0.8:
        score_components.append(0.20)
        indicators.append("irregular_energy_contour")
    elif energy_cv > 0.5:
        score_components.append(0.10)
        indicators.append("mildly_irregular_energy")
    else:
        score_components.append(0.0)

    raw_risk = float(np.clip(sum(score_components), 0.0, 1.0))
    gad7_score = round(raw_risk * 21.0, 1)

    return {
        "gad7_score": gad7_score,
        "severity": _severity_label_gad7(gad7_score),
        "indicators": indicators,
        "disclaimer": _DISCLAIMER,
    }


def track_vocal_trend(history: List[Dict], window: int = 7) -> Dict:
    """Track longitudinal vocal biomarker trends.

    Determines whether vocal biomarkers are trending toward or away from
    clinical thresholds over time.

    Args:
        history: List of past screening results (each containing phq9_score
                 and/or gad7_score). Most recent last.
        window: Number of recent entries to consider.

    Returns:
        Dict with depression_trend, anxiety_trend, and direction labels.
    """
    if not history or len(history) < 2:
        return {
            "depression_trend": "insufficient_data",
            "anxiety_trend": "insufficient_data",
            "depression_slope": 0.0,
            "anxiety_slope": 0.0,
            "entries_analyzed": len(history) if history else 0,
        }

    recent = history[-window:]
    n = len(recent)

    phq9_scores = [
        entry.get("phq9_score", entry.get("depression", {}).get("phq9_score", 0.0))
        for entry in recent
    ]
    gad7_scores = [
        entry.get("gad7_score", entry.get("anxiety", {}).get("gad7_score", 0.0))
        for entry in recent
    ]

    x = np.arange(n, dtype=float)

    dep_slope = _linear_slope(x, np.array(phq9_scores, dtype=float))
    anx_slope = _linear_slope(x, np.array(gad7_scores, dtype=float))

    def _trend_label(slope: float) -> str:
        if slope > 0.5:
            return "worsening"
        elif slope < -0.5:
            return "improving"
        return "stable"

    return {
        "depression_trend": _trend_label(dep_slope),
        "anxiety_trend": _trend_label(anx_slope),
        "depression_slope": round(dep_slope, 4),
        "anxiety_slope": round(anx_slope, 4),
        "entries_analyzed": n,
    }


def compute_screening_profile(audio: np.ndarray, sr: int = 16000) -> Dict:
    """Full screening: extract biomarkers, score depression + anxiety.

    Single-call convenience that runs the full pipeline.

    Args:
        audio: 1-D float32 mono waveform.
        sr: Sample rate in Hz.

    Returns:
        Dict with biomarkers, depression, anxiety, and disclaimer.
    """
    biomarkers = extract_vocal_biomarkers(audio, sr)
    depression = score_depression_risk(biomarkers)
    anxiety = score_anxiety_risk(biomarkers)

    return {
        "biomarkers": biomarkers,
        "depression": depression,
        "anxiety": anxiety,
        "disclaimer": _DISCLAIMER,
    }


def profile_to_dict(profile: Dict) -> Dict:
    """Normalize a screening profile to a JSON-safe dictionary.

    Ensures all numpy types are converted to native Python types.

    Args:
        profile: Output of compute_screening_profile().

    Returns:
        JSON-serializable dict.
    """
    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return _convert(profile)


# -- internal helpers ---------------------------------------------------------

_DISCLAIMER = (
    "These scores are research-grade estimates, not clinical diagnoses. "
    "Consult a qualified mental health professional for any concerns."
)


def _estimate_f0_autocorr(audio: np.ndarray, sr: int) -> np.ndarray:
    """Estimate F0 using autocorrelation method (no librosa dependency)."""
    frame_len = int(sr * 0.030)  # 30ms frames
    hop = int(sr * 0.010)  # 10ms hop
    min_lag = int(sr / 400)  # 400 Hz max pitch
    max_lag = int(sr / 60)  # 60 Hz min pitch

    n_frames = max(1, (len(audio) - frame_len) // hop)
    f0_values = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_len]
        if len(frame) < frame_len:
            break

        # Autocorrelation
        frame_norm = frame - np.mean(frame)
        energy = np.sum(frame_norm ** 2)
        if energy < 1e-10:
            continue

        corr = np.correlate(frame_norm, frame_norm, mode="full")
        corr = corr[len(corr) // 2:]

        # Search for peak in valid pitch range
        search_start = min(min_lag, len(corr) - 1)
        search_end = min(max_lag, len(corr))
        if search_start >= search_end:
            continue

        segment = corr[search_start:search_end]
        if len(segment) == 0:
            continue

        peak_idx = np.argmax(segment) + search_start
        if corr[peak_idx] > 0.3 * corr[0]:  # voiced threshold
            f0_values[i] = sr / peak_idx

    return f0_values


def _compute_rms(audio: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Compute frame-level RMS energy."""
    n_frames = max(1, (len(audio) - frame_len) // hop)
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_len]
        if len(frame) > 0:
            rms[i] = float(np.sqrt(np.mean(frame ** 2)))
    return rms


def _compute_jitter(f0_valid: np.ndarray) -> Dict:
    """Compute jitter metrics from F0 values."""
    if len(f0_valid) < 3:
        return {"jitter_local": 0.0, "jitter_rap": 0.0}

    periods = 1.0 / np.maximum(f0_valid, 1e-9)
    diffs = np.abs(np.diff(periods))
    mean_period = float(np.mean(periods))

    jitter_local = float(np.mean(diffs) / max(mean_period, 1e-9))

    # RAP: relative average perturbation (3-point)
    if len(periods) >= 3:
        smoothed = np.convolve(periods, np.ones(3) / 3, mode="valid")
        rap_diffs = np.abs(periods[1:len(smoothed) + 1] - smoothed)
        jitter_rap = float(np.mean(rap_diffs) / max(mean_period, 1e-9))
    else:
        jitter_rap = jitter_local

    return {"jitter_local": round(jitter_local, 6), "jitter_rap": round(jitter_rap, 6)}


def _compute_shimmer(rms: np.ndarray) -> Dict:
    """Compute shimmer metrics from RMS amplitude."""
    if len(rms) < 3:
        return {"shimmer_local": 0.0, "shimmer_apq3": 0.0}

    voiced = rms[rms > np.max(rms) * 0.1]
    if len(voiced) < 3:
        return {"shimmer_local": 0.0, "shimmer_apq3": 0.0}

    diffs = np.abs(np.diff(voiced))
    mean_amp = float(np.mean(voiced))
    shimmer_local = float(np.mean(diffs) / max(mean_amp, 1e-9))

    # APQ3: amplitude perturbation quotient (3-point)
    smoothed = np.convolve(voiced, np.ones(3) / 3, mode="valid")
    apq3_diffs = np.abs(voiced[1:len(smoothed) + 1] - smoothed)
    shimmer_apq3 = float(np.mean(apq3_diffs) / max(mean_amp, 1e-9))

    return {
        "shimmer_local": round(shimmer_local, 6),
        "shimmer_apq3": round(shimmer_apq3, 6),
    }


def _extract_pause_metrics(
    audio: np.ndarray, sr: int, rms: np.ndarray
) -> Dict:
    """Extract pause/silence metrics -- strongest depression predictor."""
    if len(rms) == 0:
        return {
            "silence_ratio": 0.0,
            "pause_count": 0,
            "mean_pause_duration": 0.0,
            "max_pause_duration": 0.0,
        }

    threshold = np.max(rms) * 0.08  # silence threshold
    is_silent = rms < threshold

    silence_ratio = float(np.mean(is_silent))

    # Count pause segments
    pauses: List[int] = []
    current_pause = 0
    for s in is_silent:
        if s:
            current_pause += 1
        else:
            if current_pause > 2:  # minimum 2 frames for a pause
                pauses.append(current_pause)
            current_pause = 0
    if current_pause > 2:
        pauses.append(current_pause)

    hop_sec = 0.010  # 10ms hop used in RMS
    pause_durations = [p * hop_sec for p in pauses]

    return {
        "silence_ratio": round(silence_ratio, 4),
        "pause_count": len(pauses),
        "mean_pause_duration": round(float(np.mean(pause_durations)), 4) if pause_durations else 0.0,
        "max_pause_duration": round(float(np.max(pause_durations)), 4) if pause_durations else 0.0,
    }


def _extract_speaking_rate(
    audio: np.ndarray, sr: int, f0_values: np.ndarray
) -> Dict:
    """Estimate speaking rate and its variability."""
    voiced_mask = f0_values > 0
    total_frames = len(f0_values)
    if total_frames == 0:
        return {"speaking_rate": 0.0, "speaking_rate_variability": 0.0}

    # Count voiced-to-unvoiced transitions (proxy for syllables)
    transitions = 0
    for i in range(1, len(voiced_mask)):
        if voiced_mask[i] and not voiced_mask[i - 1]:
            transitions += 1

    duration_sec = len(audio) / sr
    speaking_rate = transitions / max(duration_sec, 0.01)

    # Variability: compute local rates in 1-second windows
    hop_sec = 0.010
    frames_per_sec = int(1.0 / hop_sec)
    local_rates: List[float] = []
    for start in range(0, len(voiced_mask) - frames_per_sec, frames_per_sec):
        chunk = voiced_mask[start:start + frames_per_sec]
        local_trans = sum(
            1 for j in range(1, len(chunk)) if chunk[j] and not chunk[j - 1]
        )
        local_rates.append(float(local_trans))

    rate_var = float(np.std(local_rates)) if len(local_rates) > 1 else 0.0

    return {
        "speaking_rate": round(speaking_rate, 3),
        "speaking_rate_variability": round(rate_var, 4),
    }


def _compute_formant_stability(audio: np.ndarray, sr: int) -> Dict:
    """Estimate formant stability using spectral centroid variability.

    Uses spectral centroid as a proxy for first formant tracking.
    Lower stability = more articulatory tension (anxiety marker).
    """
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    n_frames = max(1, (len(audio) - frame_len) // hop)

    centroids: List[float] = []
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_len]
        if len(frame) < frame_len:
            break

        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)

        total_energy = np.sum(spectrum)
        if total_energy > 1e-10:
            centroid = float(np.sum(freqs * spectrum) / total_energy)
            centroids.append(centroid)

    if len(centroids) < 2:
        return {"formant_stability": 1.0}

    centroid_arr = np.array(centroids)
    mean_c = float(np.mean(centroid_arr))
    std_c = float(np.std(centroid_arr))
    cv = std_c / max(mean_c, 1e-9)

    # Stability: 1.0 = perfectly stable, 0.0 = very unstable
    stability = float(np.clip(1.0 - cv, 0.0, 1.0))

    return {"formant_stability": round(stability, 4)}


def _compute_energy_slope(rms: np.ndarray) -> float:
    """Compute linear slope of energy contour."""
    if len(rms) < 2:
        return 0.0
    x = np.arange(len(rms), dtype=float)
    return float(_linear_slope(x, rms))


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Compute slope via least-squares linear regression."""
    n = len(x)
    if n < 2:
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


# -- module-level singleton ---------------------------------------------------

_screener_instance: Optional[Dict] = None


def get_screener():
    """Return a dict of screening functions (stateless module)."""
    return {
        "extract_vocal_biomarkers": extract_vocal_biomarkers,
        "score_depression_risk": score_depression_risk,
        "score_anxiety_risk": score_anxiety_risk,
        "track_vocal_trend": track_vocal_trend,
        "compute_screening_profile": compute_screening_profile,
        "profile_to_dict": profile_to_dict,
    }
