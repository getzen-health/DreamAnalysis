"""Voice fatigue scanner — acoustic biomarker analysis (issue #377).

Extracts four acoustic biomarkers from raw audio and computes a composite
fatigue index (0-100).  Higher scores indicate greater vocal/cognitive fatigue.

Biomarkers:
    HNR  (harmonic-to-noise ratio, dB)  — higher = cleaner, less fatigued voice
    jitter (local, %)                   — pitch perturbation; rises with fatigue
    shimmer (local, dB)                 — amplitude perturbation; rises with fatigue
    speaking_rate (syllables/sec proxy) — slows down with fatigue

Scoring:
    Compare each biomarker to a personal baseline (from VoiceBaselineCalibrator
    supplemented with fatigue-specific calibration data) or population norms.
    Deviation in the fatigue-indicating direction contributes to the index.

Population norms (literature-derived):
    HNR  : ~20 dB (range 10-35 dB)
    jitter: ~0.5 % (range 0.1-3 %)
    shimmer: ~0.3 dB (range 0.1-1.5 dB)
    speaking_rate: ~4.5 syllables/sec (ZCR-based proxy, normalised to 0-1)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Population norms ──────────────────────────────────────────────────────────

_NORM_HNR_DB: float = 20.0          # dB — healthy resting voice
_NORM_JITTER_PCT: float = 0.5       # % — local pitch perturbation
_NORM_SHIMMER_DB: float = 0.3       # dB — local amplitude perturbation
_NORM_SPEAKING_RATE: float = 0.48   # normalised ZCR proxy (~4.5 syll/sec)

# Max expected deviations (used for normalising contribution to 0-1)
_MAX_HNR_DELTA: float = 15.0        # dB below norm before full contribution
_MAX_JITTER_DELTA: float = 2.0      # % above norm
_MAX_SHIMMER_DELTA: float = 1.0     # dB above norm
_MAX_RATE_DELTA: float = 0.30       # normalised ZCR drop

# Weights for the composite fatigue index (must sum to 1.0)
_W_HNR: float = 0.35
_W_JITTER: float = 0.25
_W_SHIMMER: float = 0.20
_W_RATE: float = 0.20


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class FatigueResult:
    """Output of VoiceFatigueScanner.scan().

    Attributes:
        fatigue_index:     Composite score 0-100. Higher = more fatigued.
        hnr_db:            Measured harmonic-to-noise ratio in dB.
        hnr_delta:         Deviation from baseline/norm (negative = lower HNR = worse).
        jitter_pct:        Measured local jitter in %.
        jitter_ratio:      jitter / baseline_jitter (>1 = more perturbation).
        shimmer_db:        Measured local shimmer in dB.
        shimmer_ratio:     shimmer / baseline_shimmer (>1 = more perturbation).
        speaking_rate_proxy: Normalised ZCR-based speaking-rate proxy.
        speaking_rate_ratio: rate / baseline_rate (<1 = slower = more fatigued).
        confidence:        0-1 confidence in the estimate (lower for very short audio).
        recommendations:   List of human-readable action suggestions.
        baseline_used:     True if a personal baseline was available and used.
    """
    fatigue_index: float
    hnr_db: float
    hnr_delta: float
    jitter_pct: float
    jitter_ratio: float
    shimmer_db: float
    shimmer_ratio: float
    speaking_rate_proxy: float
    speaking_rate_ratio: float
    confidence: float
    recommendations: List[str] = field(default_factory=list)
    baseline_used: bool = False


# ── Core biomarker extraction ─────────────────────────────────────────────────

def _extract_hnr(y: np.ndarray, sr: int) -> float:
    """Estimate harmonics-to-noise ratio (HNR) in dB via autocorrelation.

    Method:
        1. Compute normalised autocorrelation at the dominant pitch lag.
        2. HNR = 10 * log10(r / (1 - r)) where r is the normalised peak.

    Returns 0.0 if no voiced frame is detected.
    """
    frame_len = int(sr * 0.025)  # 25 ms
    hop_len = int(sr * 0.010)    # 10 ms
    f0_min, f0_max = 60, 500
    lag_max = int(sr / f0_min)
    lag_min = int(sr / f0_max)

    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    hnr_values: List[float] = []

    for i in range(n_frames):
        start = i * hop_len
        frame = y[start: start + frame_len]
        if len(frame) < frame_len:
            break

        win = frame * np.hanning(len(frame))
        ac = np.correlate(win, win, mode="full")
        ac = ac[len(ac) // 2:]
        if ac[0] < 1e-10:
            continue
        ac_norm = ac / ac[0]

        search = ac_norm[lag_min: lag_max + 1]
        if len(search) == 0:
            continue
        peak_idx = int(np.argmax(search))
        r = float(search[peak_idx])
        if r <= 0.0 or r >= 1.0:
            continue
        # Only count clearly voiced frames
        if r < 0.3:
            continue
        hnr_values.append(10.0 * np.log10(r / (1.0 - r)))

    return float(np.mean(hnr_values)) if hnr_values else 0.0


def _extract_jitter(y: np.ndarray, sr: int) -> float:
    """Estimate local jitter (%) — mean absolute F0 perturbation / mean F0.

    Jitter measures cycle-to-cycle pitch irregularity.  Higher values indicate
    increased vocal-fold aperiodicity, which rises under fatigue/stress.

    Returns 0.0 if fewer than 3 voiced periods are found.
    """
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    f0_min, f0_max = 60, 500
    lag_max = int(sr / f0_min)
    lag_min = int(sr / f0_max)

    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    period_lengths: List[float] = []  # in samples

    for i in range(n_frames):
        start = i * hop_len
        frame = y[start: start + frame_len]
        if len(frame) < frame_len:
            break
        win = frame * np.hanning(len(frame))
        ac = np.correlate(win, win, mode="full")
        ac = ac[len(ac) // 2:]
        if ac[0] < 1e-10:
            continue
        ac_norm = ac / ac[0]
        search = ac_norm[lag_min: lag_max + 1]
        if len(search) == 0:
            continue
        peak_idx = int(np.argmax(search))
        r = float(search[peak_idx])
        if r < 0.3:
            continue
        period_lengths.append(float(lag_min + peak_idx))

    if len(period_lengths) < 3:
        return 0.0

    periods = np.array(period_lengths)
    mean_period = periods.mean()
    if mean_period < 1e-6:
        return 0.0

    abs_diffs = np.abs(np.diff(periods))
    jitter_pct = float(abs_diffs.mean() / mean_period * 100.0)
    return min(jitter_pct, 10.0)  # cap at 10% for numerical stability


def _extract_shimmer(y: np.ndarray, sr: int) -> float:
    """Estimate local shimmer (dB) — mean amplitude perturbation per period.

    Shimmer measures cycle-to-cycle amplitude irregularity.  Rises with fatigue.

    Returns 0.0 if fewer than 3 voiced periods are found.
    """
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    f0_min, f0_max = 60, 500
    lag_max = int(sr / f0_min)
    lag_min = int(sr / f0_max)

    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    amplitudes: List[float] = []  # RMS per period

    for i in range(n_frames):
        start = i * hop_len
        frame = y[start: start + frame_len]
        if len(frame) < frame_len:
            break
        win = frame * np.hanning(len(frame))
        ac = np.correlate(win, win, mode="full")
        ac = ac[len(ac) // 2:]
        if ac[0] < 1e-10:
            continue
        ac_norm = ac / ac[0]
        search = ac_norm[lag_min: lag_max + 1]
        if len(search) == 0:
            continue
        peak_idx = int(np.argmax(search))
        r = float(search[peak_idx])
        if r < 0.3:
            continue
        # Use the RMS of the original (unwindowed) frame as amplitude proxy
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms > 1e-9:
            amplitudes.append(rms)

    if len(amplitudes) < 3:
        return 0.0

    amps = np.array(amplitudes)
    abs_diffs = np.abs(np.diff(amps))
    mean_amp = (amps[:-1] + amps[1:]) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        shimmer_linear = np.where(mean_amp > 1e-9, abs_diffs / mean_amp, 0.0)

    shimmer_mean = float(shimmer_linear.mean())
    if shimmer_mean < 1e-9:
        return 0.0
    # Convert to dB
    shimmer_db = abs(20.0 * np.log10(1.0 + shimmer_mean))
    return min(float(shimmer_db), 5.0)  # cap at 5 dB


def _extract_speaking_rate_proxy(y: np.ndarray, sr: int) -> float:
    """Normalised zero-crossing-rate as speaking-rate proxy.

    Same formula used in extract_acoustic_features() — result is ~0-1.
    """
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    zcr_frames: List[float] = []

    for i in range(n_frames):
        start = i * hop_len
        frame = y[start: start + frame_len]
        if len(frame) < frame_len:
            break
        crossings = float(np.sum(np.diff(np.sign(frame)) != 0)) / frame_len
        zcr_frames.append(crossings)

    if not zcr_frames:
        return 0.0
    return float(min(1.0, np.mean(zcr_frames) * 20))


# ── Main scanner class ────────────────────────────────────────────────────────

class VoiceFatigueScanner:
    """Morning / real-time voice fatigue scan using acoustic biomarkers.

    Usage::

        scanner = VoiceFatigueScanner()
        result = scanner.scan(audio_array, sr=22050)
        # result.fatigue_index: 0-100

    With personal baseline (from previous rested-voice recording)::

        baseline = {
            "hnr_db": 22.0,
            "jitter_pct": 0.4,
            "shimmer_db": 0.25,
            "speaking_rate_proxy": 0.50,
        }
        result = scanner.scan(audio_array, sr=22050, baseline=baseline)
    """

    # Minimum audio duration for a reliable scan (seconds)
    _MIN_DURATION_SEC: float = 1.0

    def scan(
        self,
        audio: np.ndarray,
        sr: int,
        baseline: Optional[Dict[str, float]] = None,
    ) -> FatigueResult:
        """Extract acoustic biomarkers and compute fatigue index.

        Args:
            audio:    1-D float32 audio array.
            sr:       Sample rate in Hz.
            baseline: Optional dict with keys hnr_db, jitter_pct, shimmer_db,
                      speaking_rate_proxy representing the user's rested voice.
                      When None, population norms are used.

        Returns:
            FatigueResult with fatigue_index in [0, 100].
        """
        if audio is None or len(audio) < int(sr * self._MIN_DURATION_SEC):
            log.warning("VoiceFatigueScanner: audio too short for reliable scan")
            return FatigueResult(
                fatigue_index=0.0,
                hnr_db=0.0,
                hnr_delta=0.0,
                jitter_pct=0.0,
                jitter_ratio=1.0,
                shimmer_db=0.0,
                shimmer_ratio=1.0,
                speaking_rate_proxy=0.0,
                speaking_rate_ratio=1.0,
                confidence=0.0,
                recommendations=["Audio too short — record at least 1 second."],
                baseline_used=False,
            )

        y = audio.astype(np.float32)
        peak = np.abs(y).max()
        if peak > 1e-9:
            y = y / peak

        # ── Extract biomarkers ─────────────────────────────────────────────
        hnr_db = _extract_hnr(y, sr)
        jitter_pct = _extract_jitter(y, sr)
        shimmer_db = _extract_shimmer(y, sr)
        rate_proxy = _extract_speaking_rate_proxy(y, sr)

        # ── Determine reference values ─────────────────────────────────────
        baseline_used = baseline is not None
        ref_hnr = float(baseline.get("hnr_db", _NORM_HNR_DB)) if baseline else _NORM_HNR_DB
        ref_jitter = float(baseline.get("jitter_pct", _NORM_JITTER_PCT)) if baseline else _NORM_JITTER_PCT
        ref_shimmer = float(baseline.get("shimmer_db", _NORM_SHIMMER_DB)) if baseline else _NORM_SHIMMER_DB
        ref_rate = float(baseline.get("speaking_rate_proxy", _NORM_SPEAKING_RATE)) if baseline else _NORM_SPEAKING_RATE

        # ── Compute per-biomarker fatigue contributions (0-1 each) ─────────
        # HNR: lower than reference → fatigued
        hnr_delta = hnr_db - ref_hnr  # negative = more fatigued
        hnr_contrib = float(np.clip(-hnr_delta / _MAX_HNR_DELTA, 0.0, 1.0))

        # Jitter: higher than reference → fatigued
        jitter_delta = jitter_pct - ref_jitter
        jitter_contrib = float(np.clip(jitter_delta / _MAX_JITTER_DELTA, 0.0, 1.0))
        jitter_ratio = jitter_pct / max(ref_jitter, 1e-6)

        # Shimmer: higher than reference → fatigued
        shimmer_delta = shimmer_db - ref_shimmer
        shimmer_contrib = float(np.clip(shimmer_delta / _MAX_SHIMMER_DELTA, 0.0, 1.0))
        shimmer_ratio = shimmer_db / max(ref_shimmer, 1e-6)

        # Speaking rate: lower than reference → fatigued
        rate_delta = ref_rate - rate_proxy  # positive = slower = more fatigued
        rate_contrib = float(np.clip(rate_delta / _MAX_RATE_DELTA, 0.0, 1.0))
        speaking_rate_ratio = rate_proxy / max(ref_rate, 1e-6)

        # ── Composite fatigue index (0-100) ────────────────────────────────
        raw_score = (
            _W_HNR * hnr_contrib
            + _W_JITTER * jitter_contrib
            + _W_SHIMMER * shimmer_contrib
            + _W_RATE * rate_contrib
        )
        fatigue_index = float(np.clip(raw_score * 100.0, 0.0, 100.0))

        # ── Confidence — based on audio duration and HNR detectability ─────
        duration_sec = len(y) / sr
        # Full confidence from 5 seconds upward; scales linearly below that
        duration_conf = float(np.clip(duration_sec / 5.0, 0.0, 1.0))
        # HNR > 0 means at least some voiced frames were found
        voice_conf = 1.0 if hnr_db > 0 else 0.3
        confidence = float(np.clip(duration_conf * voice_conf, 0.0, 1.0))

        # ── Recommendations ────────────────────────────────────────────────
        recs: List[str] = []
        if fatigue_index >= 70:
            recs.append("High vocal fatigue detected — consider resting your voice.")
            recs.append("Hydrate well: dehydration significantly worsens vocal HNR.")
        elif fatigue_index >= 40:
            recs.append("Moderate vocal fatigue — warm up your voice before calls.")
        else:
            recs.append("Vocal fatigue is low — voice is in good shape.")

        if jitter_ratio > 2.0:
            recs.append("Elevated pitch jitter — try vocal warm-up exercises.")
        if shimmer_ratio > 2.0:
            recs.append("Elevated shimmer — possible vocal-fold tension or hoarseness.")
        if speaking_rate_ratio < 0.7:
            recs.append("Speaking rate notably slower than baseline — may indicate tiredness.")
        if hnr_db > 0 and hnr_delta < -10:
            recs.append("HNR significantly below baseline — noisy or breathy voice quality.")

        return FatigueResult(
            fatigue_index=round(fatigue_index, 2),
            hnr_db=round(hnr_db, 2),
            hnr_delta=round(hnr_delta, 2),
            jitter_pct=round(jitter_pct, 4),
            jitter_ratio=round(jitter_ratio, 4),
            shimmer_db=round(shimmer_db, 4),
            shimmer_ratio=round(shimmer_ratio, 4),
            speaking_rate_proxy=round(rate_proxy, 4),
            speaking_rate_ratio=round(speaking_rate_ratio, 4),
            confidence=round(confidence, 4),
            recommendations=recs,
            baseline_used=baseline_used,
        )


# ── Module-level singleton ─────────────────────────────────────────────────────

_scanner_instance: Optional[VoiceFatigueScanner] = None


def get_voice_fatigue_scanner() -> VoiceFatigueScanner:
    """Return (or create) the module-level VoiceFatigueScanner singleton."""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = VoiceFatigueScanner()
    return _scanner_instance
