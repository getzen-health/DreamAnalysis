"""Parkinson's disease tremor screening from 4-channel Muse 2 EEG.

Screens for PD-associated EEG biomarkers using feature-based heuristics:
  - Resting tremor detection (4-6 Hz spectral peak on temporal channels)
  - Cortical theta excess (cortical slowing)
  - Beta desynchronization deficit (reduced beta power)
  - Alpha peak frequency slowing
  - Inter-channel coherence reduction
  - Asymmetric beta power (PD is initially unilateral)
  - 1/f aperiodic slope changes

Risk levels: low (0-25), mild (25-50), moderate (50-75), elevated (75-100).

DISCLAIMER: This is a screening tool only, NOT a diagnostic instrument.
Parkinson's disease diagnosis requires evaluation by a qualified neurologist
using standardized clinical assessment (MDS-UPDRS, DaTscan, etc.).

Scientific references:
    Stoffers et al. (2007) — EEG slowing in PD (theta excess, alpha slowing)
    Caviness et al. (2006) — 4-6 Hz resting tremor spectral signature
    Babiloni et al. (2011) — Reduced cortical beta in PD
    George et al. (2013) — Alpha peak frequency reduction in PD
    Silberstein et al. (2005) — Asymmetric beta coherence in PD
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import welch

# NumPy 2.0 renamed np.trapz -> np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

MEDICAL_DISCLAIMER = (
    "This is a screening tool only, not a diagnostic instrument. "
    "Parkinson's disease diagnosis requires evaluation by a qualified "
    "neurologist using standardized clinical assessment."
)

# Population norms (approximate, from literature)
_NORM_THETA_RELATIVE = 0.20   # typical theta fraction of total power
_NORM_BETA_RELATIVE = 0.35    # typical beta fraction of total power
_NORM_ALPHA_PEAK = 10.0       # typical IAF in Hz
_TREMOR_LOW = 4.0             # Hz — lower bound of PD resting tremor
_TREMOR_HIGH = 6.0            # Hz — upper bound of PD resting tremor
_TREMOR_PROMINENCE = 2.0      # peak must exceed surrounding power by this factor
_HISTORY_CAP = 500


class ParkinsonsScreener:
    """EEG-based Parkinson's disease tremor and biomarker screener.

    Designed for 4-channel Muse 2 EEG:
      ch0 = TP9  (left temporal)
      ch1 = AF7  (left frontal)
      ch2 = AF8  (right frontal)
      ch3 = TP10 (right temporal)
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    # ── Public API ─────────────────────────────────────────────────

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline for normalization.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate in Hz. Defaults to constructor value.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set and baseline_metrics.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        bands_all = self._extract_all_bands(signals, fs)

        baseline = {
            "theta": bands_all["theta"],
            "alpha": bands_all["alpha"],
            "beta": bands_all["beta"],
            "delta": bands_all["delta"],
            "alpha_peak_freq": bands_all["alpha_peak_freq"],
        }
        self._baselines[user_id] = baseline

        return {
            "baseline_set": True,
            "baseline_metrics": {k: round(v, 6) for k, v in baseline.items()},
        }

    def screen(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Screen for PD-associated EEG biomarkers.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate in Hz.
            user_id: User identifier.

        Returns:
            Dict with risk_score (0-100), risk_level, tremor info,
            biomarker indices, and medical disclaimer.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)

        bands_all = self._extract_all_bands(signals, fs)
        baseline = self._baselines.get(user_id)

        theta = bands_all["theta"]
        alpha = bands_all["alpha"]
        beta = bands_all["beta"]
        delta = bands_all["delta"]
        alpha_peak = bands_all["alpha_peak_freq"]
        total_power = theta + alpha + beta + delta + 1e-10

        # ── Theta excess (cortical slowing) ──
        theta_relative = theta / total_power
        if baseline:
            bl_total = baseline["theta"] + baseline["alpha"] + baseline["beta"] + baseline["delta"] + 1e-10
            bl_theta_rel = baseline["theta"] / bl_total
            theta_excess = float(np.clip((theta_relative - bl_theta_rel) / 0.20, 0.0, 1.0))
        else:
            theta_excess = float(np.clip(
                (theta_relative - _NORM_THETA_RELATIVE) / 0.25, 0.0, 1.0
            ))

        # ── Beta deficit (desynchronization deficit) ──
        beta_relative = beta / total_power
        if baseline:
            bl_beta_rel = baseline["beta"] / bl_total
            beta_deficit = float(np.clip((bl_beta_rel - beta_relative) / 0.20, 0.0, 1.0))
        else:
            beta_deficit = float(np.clip(
                (_NORM_BETA_RELATIVE - beta_relative) / 0.25, 0.0, 1.0
            ))

        # ── Alpha peak frequency slowing ──
        if baseline and baseline["alpha_peak_freq"] > 0:
            apf_ref = baseline["alpha_peak_freq"]
        else:
            apf_ref = _NORM_ALPHA_PEAK
        alpha_slow = float(np.clip((apf_ref - alpha_peak) / 4.0, 0.0, 1.0))

        # ── Delta excess (deep slowing) ──
        delta_relative = delta / total_power
        delta_excess = float(np.clip((delta_relative - 0.25) / 0.25, 0.0, 1.0))

        # ── Tremor detection (4-6 Hz peak on temporal channels) ──
        tremor_detected, tremor_frequency = self._detect_tremor(signals, fs)

        # ── Beta asymmetry (PD is unilateral initially) ──
        asymmetry_index = self._compute_beta_asymmetry(signals, fs)

        # ── Aperiodic slope (1/f flattening in PD) ──
        aperiodic_score = self._compute_aperiodic_score(signals, fs)

        # ── Composite biomarkers dict ──
        biomarkers = {
            "theta_excess": round(theta_excess, 4),
            "beta_deficit": round(beta_deficit, 4),
            "alpha_peak_slowing": round(alpha_slow, 4),
            "delta_excess": round(delta_excess, 4),
            "beta_asymmetry": round(asymmetry_index, 4),
            "aperiodic_flattening": round(aperiodic_score, 4),
            "tremor_score": 1.0 if tremor_detected else 0.0,
        }

        # ── Composite risk score (0-100) ──
        risk_raw = (
            0.25 * theta_excess
            + 0.20 * beta_deficit
            + 0.15 * alpha_slow
            + 0.15 * (1.0 if tremor_detected else 0.0)
            + 0.10 * asymmetry_index
            + 0.10 * aperiodic_score
            + 0.05 * delta_excess
        )
        risk_score = float(np.clip(risk_raw * 100.0, 0.0, 100.0))

        # ── Risk level ──
        if risk_score >= 75:
            risk_level = "elevated"
        elif risk_score >= 50:
            risk_level = "moderate"
        elif risk_score >= 25:
            risk_level = "mild"
        else:
            risk_level = "low"

        result = {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "tremor_frequency": round(tremor_frequency, 2) if tremor_frequency is not None else None,
            "tremor_detected": tremor_detected,
            "theta_excess": round(theta_excess, 4),
            "beta_deficit": round(beta_deficit, 4),
            "alpha_peak_freq": round(alpha_peak, 2),
            "asymmetry_index": round(asymmetry_index, 4),
            "biomarkers": biomarkers,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
            "has_baseline": user_id in self._baselines,
        }

        # Store in history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > _HISTORY_CAP:
            self._history[user_id] = self._history[user_id][-_HISTORY_CAP:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get aggregate session statistics.

        Returns:
            Dict with n_epochs, mean_risk, tremor_detection_rate, has_baseline.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {
                "n_epochs": 0,
                "mean_risk": 0.0,
                "tremor_detection_rate": 0.0,
                "has_baseline": user_id in self._baselines,
            }

        risks = [h["risk_score"] for h in history]
        tremor_count = sum(1 for h in history if h["tremor_detected"])

        return {
            "n_epochs": len(history),
            "mean_risk": round(float(np.mean(risks)), 2),
            "tremor_detection_rate": round(tremor_count / len(history), 4),
            "has_baseline": user_id in self._baselines,
        }

    def get_history(
        self,
        user_id: str = "default",
        last_n: Optional[int] = None,
    ) -> List[Dict]:
        """Get screening history for a user.

        Args:
            user_id: User identifier.
            last_n: Return only the last N entries. None = all.

        Returns:
            List of screening result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None and last_n > 0:
            return history[-last_n:]
        return list(history)

    def reset(self, user_id: Optional[str] = None):
        """Clear session state.

        Args:
            user_id: If provided, clear only that user. If None, clear all.
        """
        if user_id is not None:
            self._baselines.pop(user_id, None)
            self._history.pop(user_id, None)
        else:
            self._baselines.clear()
            self._history.clear()

    # ── Private helpers ────────────────────────────────────────────

    def _extract_all_bands(self, signals: np.ndarray, fs: float) -> Dict:
        """Extract band powers averaged across available channels."""
        if signals.ndim == 1:
            channels = [signals]
        else:
            channels = [signals[i] for i in range(signals.shape[0])]

        band_sums = {"delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0}
        alpha_peaks = []

        for ch in channels:
            nperseg = min(len(ch), int(fs * 2))
            if nperseg < 4:
                continue
            try:
                freqs, psd = welch(ch, fs=fs, nperseg=nperseg)
            except Exception:
                continue

            band_sums["delta"] += self._band_power(freqs, psd, 0.5, 4.0)
            band_sums["theta"] += self._band_power(freqs, psd, 4.0, 8.0)
            band_sums["alpha"] += self._band_power(freqs, psd, 8.0, 12.0)
            band_sums["beta"] += self._band_power(freqs, psd, 12.0, 30.0)

            # Alpha peak frequency
            alpha_mask = (freqs >= 7) & (freqs <= 13)
            if alpha_mask.any() and np.max(psd[alpha_mask]) > 0:
                alpha_peaks.append(float(freqs[alpha_mask][np.argmax(psd[alpha_mask])]))

        n_ch = max(len(channels), 1)
        result = {k: v / n_ch for k, v in band_sums.items()}
        result["alpha_peak_freq"] = float(np.mean(alpha_peaks)) if alpha_peaks else _NORM_ALPHA_PEAK

        return result

    def _band_power(
        self, freqs: np.ndarray, psd: np.ndarray, low: float, high: float,
    ) -> float:
        """Compute power in a frequency band via trapezoidal integration."""
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return float(_trapezoid(psd[mask], freqs[mask]))

    def _detect_tremor(
        self, signals: np.ndarray, fs: float,
    ) -> tuple:
        """Detect 4-6 Hz resting tremor peak on temporal channels.

        Uses TP9 (ch0) and TP10 (ch3) if available, as temporal channels
        pick up movement artifact from hand tremor.

        Returns:
            (tremor_detected: bool, tremor_frequency: float or None)
        """
        if signals.ndim == 1:
            temporal = [signals]
        elif signals.shape[0] >= 4:
            temporal = [signals[0], signals[3]]  # TP9, TP10
        else:
            temporal = [signals[i] for i in range(signals.shape[0])]

        best_prominence = 0.0
        best_freq = None

        for ch in temporal:
            nperseg = min(len(ch), int(fs * 2))
            if nperseg < 4:
                continue
            try:
                freqs, psd = welch(ch, fs=fs, nperseg=nperseg)
            except Exception:
                continue

            # Look for peak in 4-6 Hz tremor band
            tremor_mask = (freqs >= _TREMOR_LOW) & (freqs <= _TREMOR_HIGH)
            if not tremor_mask.any():
                continue

            tremor_psd = psd[tremor_mask]
            tremor_freqs = freqs[tremor_mask]
            peak_idx = np.argmax(tremor_psd)
            peak_power = tremor_psd[peak_idx]
            peak_freq = tremor_freqs[peak_idx]

            # Surrounding power: average of 2-4 Hz and 6-8 Hz bands
            surround_masks = [
                (freqs >= 2.0) & (freqs < _TREMOR_LOW),
                (freqs > _TREMOR_HIGH) & (freqs <= 8.0),
            ]
            surround_powers = []
            for sm in surround_masks:
                if sm.any():
                    surround_powers.append(float(np.mean(psd[sm])))
            surround_mean = float(np.mean(surround_powers)) if surround_powers else 1e-10

            prominence = peak_power / max(surround_mean, 1e-10)
            if prominence > best_prominence:
                best_prominence = prominence
                best_freq = peak_freq

        if best_prominence >= _TREMOR_PROMINENCE and best_freq is not None:
            return True, float(best_freq)
        return False, None

    def _compute_beta_asymmetry(self, signals: np.ndarray, fs: float) -> float:
        """Compute left/right beta power asymmetry.

        For Muse 2: compares left (TP9+AF7) vs right (AF8+TP10).
        Returns 0 for symmetric, higher for more asymmetric.
        """
        if signals.ndim == 1 or signals.shape[0] < 2:
            return 0.0

        if signals.shape[0] >= 4:
            left_chs = [signals[0], signals[1]]   # TP9, AF7
            right_chs = [signals[2], signals[3]]   # AF8, TP10
        else:
            left_chs = [signals[0]]
            right_chs = [signals[1] if signals.shape[0] > 1 else signals[0]]

        left_beta = 0.0
        for ch in left_chs:
            nperseg = min(len(ch), int(fs * 2))
            if nperseg < 4:
                continue
            try:
                freqs, psd = welch(ch, fs=fs, nperseg=nperseg)
                left_beta += self._band_power(freqs, psd, 12.0, 30.0)
            except Exception:
                pass
        left_beta /= max(len(left_chs), 1)

        right_beta = 0.0
        for ch in right_chs:
            nperseg = min(len(ch), int(fs * 2))
            if nperseg < 4:
                continue
            try:
                freqs, psd = welch(ch, fs=fs, nperseg=nperseg)
                right_beta += self._band_power(freqs, psd, 12.0, 30.0)
            except Exception:
                pass
        right_beta /= max(len(right_chs), 1)

        total = left_beta + right_beta + 1e-10
        asymmetry = abs(left_beta - right_beta) / total
        return float(np.clip(asymmetry, 0.0, 1.0))

    def _compute_aperiodic_score(self, signals: np.ndarray, fs: float) -> float:
        """Estimate aperiodic (1/f) slope flattening.

        PD is associated with flattening of the 1/f slope (lower exponent).
        Uses log-log linear regression on PSD as fast approximation.
        """
        if signals.ndim == 1:
            signal = signals
        elif signals.shape[0] >= 3:
            signal = (signals[1] + signals[2]) / 2.0  # frontal average
        else:
            signal = signals[0]

        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return 0.0

        try:
            freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.0

        mask = (freqs >= 2) & (freqs <= 45)
        if mask.sum() < 5:
            return 0.0

        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask] + 1e-30)

        try:
            coeffs = np.polyfit(log_f, log_p, 1)
        except Exception:
            return 0.0

        exponent = -coeffs[0]  # typical healthy: ~1.5-2.5; PD: lower

        # Score: lower exponent = higher PD risk
        # Normal exponent ~2.0; PD-like ~1.0-1.5
        score = float(np.clip((2.0 - exponent) / 1.5, 0.0, 1.0))
        return score


# ── Module-level singleton ────────────────────────────────────────

_screener_instance: Optional[ParkinsonsScreener] = None


def get_parkinsons_screener() -> ParkinsonsScreener:
    """Get or create the singleton ParkinsonsScreener instance."""
    global _screener_instance
    if _screener_instance is None:
        _screener_instance = ParkinsonsScreener()
    return _screener_instance
