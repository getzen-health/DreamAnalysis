"""Cognitive reserve estimator from EEG spectral biomarkers.

Estimates brain resilience and cognitive reserve using spectral biomarkers:
- Alpha peak frequency (APF): dominant frequency in 8-13 Hz band
- Aperiodic (1/f) slope: flatter slope = more cognitive reserve
- Theta/alpha ratio: higher = less reserve
- Brain age index: spectral proxy for brain age (higher delta/theta, lower alpha = older)

References:
    Stern (2009) — Cognitive reserve theory
    Scally et al. (2018) — EEG spectral measures of cognitive reserve
    Cesnaite et al. (2023) — Individual alpha frequency and cognitive reserve
    Voytek et al. (2015) — Age-related changes in 1/f aperiodic slope
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


class CognitiveReserveEstimator:
    """Estimate cognitive reserve from EEG spectral biomarkers.

    Uses alpha peak frequency, aperiodic 1/f slope, theta/alpha ratio,
    and a brain age index derived from band power ratios.

    Usage:
        estimator = CognitiveReserveEstimator()
        result = estimator.predict(eeg_array, fs=256)
        # result['reserve_score'] in [0, 100]
    """

    VALID_CATEGORIES = ("low", "moderate", "high")

    def __init__(self, fs: float = 256.0) -> None:
        self._default_fs = fs
        self._score_history: List[float] = []
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Estimate cognitive reserve from a single EEG epoch.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) float array, µV.
            fs: Sampling rate in Hz.

        Returns:
            dict with keys:
                reserve_score      float  0-100  overall cognitive reserve
                brain_age_index    float  0-1    0=young brain, 1=aged brain
                alpha_peak_freq    float  Hz     dominant alpha frequency
                aperiodic_slope    float         1/f slope (negative; flatter = more reserve)
                theta_alpha_ratio  float         higher = less reserve
                reserve_category   str           "low" | "moderate" | "high"
                biomarkers         dict          sub-dict with the same numeric keys
        """
        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        apf = self._estimate_alpha_peak_freq(eeg, fs)
        slope = self._estimate_aperiodic_slope(eeg, fs)
        tar = self._estimate_theta_alpha_ratio(eeg, fs)
        bai = self._estimate_brain_age_index(eeg, fs)

        reserve_score = self._compute_reserve_score(apf, slope, tar, bai)
        reserve_category = self._categorize(reserve_score)

        biomarkers = {
            "reserve_score": round(reserve_score, 2),
            "brain_age_index": round(bai, 4),
            "alpha_peak_freq": round(apf, 3),
            "aperiodic_slope": round(slope, 4),
            "theta_alpha_ratio": round(tar, 4),
        }

        return {
            "reserve_score": round(reserve_score, 2),
            "brain_age_index": round(bai, 4),
            "alpha_peak_freq": round(apf, 3),
            "aperiodic_slope": round(slope, 4),
            "theta_alpha_ratio": round(tar, 4),
            "reserve_category": reserve_category,
            "biomarkers": biomarkers,
        }

    def update_history(self, score: float) -> None:
        """Append a reserve score to the longitudinal history."""
        with self._lock:
            self._score_history.append(float(score))

    def get_longitudinal_trend(self, n_sessions: Optional[int] = None) -> Dict:
        """Compute slope and trend from stored score history.

        Args:
            n_sessions: If given, only use the last n_sessions scores.

        Returns:
            dict with:
                slope_per_session  float  points per session
                trend              str    "improving" | "stable" | "declining"
                n_sessions         int    number of data points used
        """
        with self._lock:
            history = list(self._score_history)

        if n_sessions is not None:
            history = history[-n_sessions:]

        if len(history) < 2:
            return {
                "slope_per_session": 0.0,
                "trend": "stable",
                "n_sessions": len(history),
            }

        x = np.arange(len(history), dtype=float)
        slope = float(np.polyfit(x, history, 1)[0])

        if slope > 0.5:
            trend = "improving"
        elif slope < -0.5:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "slope_per_session": round(slope, 4),
            "trend": trend,
            "n_sessions": len(history),
        }

    def reset(self) -> None:
        """Clear longitudinal history."""
        with self._lock:
            self._score_history.clear()

    # ── Private signal-processing helpers ────────────────────────────────────

    def _welch_per_channel(self, eeg: np.ndarray, fs: float):
        """Run Welch PSD on each channel; return mean freqs and mean PSD."""
        psds = []
        freqs_ref = None
        for ch in range(eeg.shape[0]):
            sig = eeg[ch]
            nperseg = min(len(sig), int(fs * 2))
            if nperseg < 8:
                continue
            try:
                freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=nperseg)
                psds.append(psd)
                if freqs_ref is None:
                    freqs_ref = freqs
            except Exception:
                continue

        if not psds or freqs_ref is None:
            # Fallback: return flat spectrum with realistic frequencies
            freqs_ref = np.linspace(0, fs / 2, 128)
            mean_psd = np.ones_like(freqs_ref) * 1e-6
        else:
            mean_psd = np.mean(np.array(psds), axis=0)

        return freqs_ref, mean_psd

    def _band_power(self, freqs: np.ndarray, psd: np.ndarray,
                    fmin: float, fmax: float) -> float:
        """Integrate PSD in [fmin, fmax] Hz."""
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return 1e-10
        if hasattr(np, "trapezoid"):
            power = float(np.trapezoid(psd[mask], freqs[mask]))
        else:
            power = float(np.trapz(psd[mask], freqs[mask]))
        return max(power, 1e-10)

    def _estimate_alpha_peak_freq(self, eeg: np.ndarray, fs: float) -> float:
        """Alpha peak frequency: peak of PSD in 8-13 Hz range.

        Returns a value clamped to [8.0, 13.0].
        """
        freqs, psd = self._welch_per_channel(eeg, fs)
        mask = (freqs >= 8.0) & (freqs <= 13.0)
        if not np.any(mask):
            return 10.0

        alpha_freqs = freqs[mask]
        alpha_psd = psd[mask]
        peak_idx = int(np.argmax(alpha_psd))
        apf = float(alpha_freqs[peak_idx])
        return float(np.clip(apf, 8.0, 13.0))

    def _estimate_aperiodic_slope(self, eeg: np.ndarray, fs: float) -> float:
        """Fit log-log PSD to estimate 1/f aperiodic slope.

        More negative = steeper 1/f = less reserve (older brain signature).
        Typical range: -3.5 (steep/aged) to -1.0 (flat/younger).
        """
        freqs, psd = self._welch_per_channel(eeg, fs)
        # Fit in 2-40 Hz range (avoids DC and high-freq noise)
        valid = (freqs >= 2.0) & (freqs <= 40.0) & (psd > 0)
        if np.sum(valid) < 4:
            return -2.0

        log_f = np.log10(freqs[valid] + 1e-12)
        log_p = np.log10(psd[valid] + 1e-12)
        try:
            slope, _ = np.polyfit(log_f, log_p, 1)
        except Exception:
            return -2.0
        return float(slope)

    def _estimate_theta_alpha_ratio(self, eeg: np.ndarray, fs: float) -> float:
        """Theta (4-8 Hz) / alpha (8-13 Hz) power ratio.

        Higher ratio = more theta relative to alpha = less cognitive reserve.
        """
        freqs, psd = self._welch_per_channel(eeg, fs)
        theta = self._band_power(freqs, psd, 4.0, 8.0)
        alpha = self._band_power(freqs, psd, 8.0, 13.0)
        return theta / (alpha + 1e-10)

    def _estimate_brain_age_index(self, eeg: np.ndarray, fs: float) -> float:
        """Spectral brain age proxy.

        Higher delta/theta relative to alpha → older brain signature → higher index.
        Returns float in [0, 1].
        """
        freqs, psd = self._welch_per_channel(eeg, fs)
        delta = self._band_power(freqs, psd, 0.5, 4.0)
        theta = self._band_power(freqs, psd, 4.0, 8.0)
        alpha = self._band_power(freqs, psd, 8.0, 13.0)
        beta = self._band_power(freqs, psd, 13.0, 30.0)

        total = delta + theta + alpha + beta + 1e-10
        # Weighted "old brain" fraction: delta and theta contribute positively;
        # alpha and beta suppress the index.
        age_raw = (0.4 * delta + 0.4 * theta - 0.1 * alpha - 0.1 * beta) / total
        # Shift and scale to [0, 1]
        bai = float(np.clip((age_raw + 0.4) / 0.8, 0.0, 1.0))
        return bai

    def _compute_reserve_score(self, apf: float, slope: float,
                                tar: float, bai: float) -> float:
        """Combine biomarkers into a single 0-100 reserve score.

        Component weights:
            APF       30% — higher peak freq = more reserve
            slope     25% — flatter (less negative) = more reserve
            TAR       25% — lower theta/alpha ratio = more reserve
            BAI       20% — lower brain age index = more reserve
        """
        # APF score: 8 Hz → 0, 12 Hz → 100
        apf_score = float(np.clip((apf - 8.0) / 4.0 * 100.0, 0.0, 100.0))

        # Slope score: typical range −3.5 (steep/old) to −1.0 (flat/young)
        # More negative slope = steeper = less reserve → invert
        slope_score = float(np.clip((slope - (-3.5)) / ((-1.0) - (-3.5)) * 100.0, 0.0, 100.0))

        # TAR score: 0 → 100 (ratio 0), 100 → 0 (ratio ≥ 5)
        tar_score = float(np.clip((1.0 - tar / 5.0) * 100.0, 0.0, 100.0))

        # BAI score: 0 → 100 (young brain), 1 → 0 (aged brain)
        bai_score = float(np.clip((1.0 - bai) * 100.0, 0.0, 100.0))

        reserve = (
            0.30 * apf_score
            + 0.25 * slope_score
            + 0.25 * tar_score
            + 0.20 * bai_score
        )
        return float(np.clip(reserve, 0.0, 100.0))

    def _categorize(self, score: float) -> str:
        if score >= 65.0:
            return "high"
        if score >= 35.0:
            return "moderate"
        return "low"


# ── Singleton factory ─────────────────────────────────────────────────────────

_instance: Optional[CognitiveReserveEstimator] = None
_instance_lock = threading.Lock()


def get_cognitive_reserve_estimator() -> CognitiveReserveEstimator:
    """Return the process-wide CognitiveReserveEstimator singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CognitiveReserveEstimator()
    return _instance
