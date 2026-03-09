"""Brain maturation tracker via aperiodic + periodic EEG features.

Estimates brain age from EEG and computes the brain age gap (BAG),
a quantitative biomarker of neural development. BAG deviations
flag potential neurodevelopmental concerns.

Key developmental features:
- Aperiodic exponent (1/f slope) decreases with maturation
- Alpha peak frequency increases (6 Hz infant → 10 Hz adult)
- Theta relative power decreases with age
- Beta relative power increases with age
- Spectral entropy increases with cortical complexity

References:
    NeuroImage 2025 — 938 EEG recordings, MAE 91.7 days brain age
    Frontiers in Neurology 2025 — Auto-EEG brain age, r>0.8
    Frontiers in Aging Neuroscience 2025 — Lifespan brain age via aperiodic features
"""
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


class BrainMaturationTracker:
    """Track brain maturation via EEG developmental features.

    Estimates brain age, computes brain age gap (BAG), and tracks
    longitudinal maturation trajectory.
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        self._history: Dict[str, List[Dict]] = {}
        # Normative ranges by age (approximate, from literature)
        self._norms = {
            "alpha_peak_hz": {"child": (7, 9), "adolescent": (9, 10.5), "adult": (9.5, 11)},
            "theta_relative": {"child": (0.25, 0.40), "adolescent": (0.15, 0.25), "adult": (0.10, 0.20)},
            "aperiodic_exp": {"child": (1.5, 2.5), "adolescent": (1.2, 1.8), "adult": (0.8, 1.5)},
        }

    def assess(
        self,
        signals: np.ndarray,
        chronological_age: Optional[float] = None,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess brain maturation from EEG signals.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            chronological_age: Known age in years (for BAG computation).
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with estimated_brain_age, brain_age_gap,
            maturation_features, developmental_stage, and normative_comparison.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_channels = signals.shape[0]
        features = self._extract_maturation_features(signals, fs)

        # Estimate brain age from features
        estimated_age = self._estimate_brain_age(features)

        # Brain age gap
        bag = None
        if chronological_age is not None:
            bag = round(estimated_age - chronological_age, 2)

        # Developmental stage
        stage = self._classify_stage(features)

        # Normative comparison
        comparison = self._normative_comparison(features, chronological_age)

        result = {
            "estimated_brain_age": round(estimated_age, 1),
            "brain_age_gap": bag,
            "developmental_stage": stage,
            "maturation_features": {k: round(v, 4) for k, v in features.items()},
            "normative_comparison": comparison,
            "n_channels": n_channels,
        }

        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > 200:
            self._history[user_id] = self._history[user_id][-200:]

        return result

    def get_trajectory(self, user_id: str = "default") -> List[Dict]:
        """Get longitudinal maturation trajectory."""
        return self._history.get(user_id, [])

    def get_summary(self, user_id: str = "default") -> Dict:
        """Summary statistics across assessments."""
        history = self._history.get(user_id, [])
        if not history:
            return {"n_assessments": 0}

        ages = [h["estimated_brain_age"] for h in history]
        bags = [h["brain_age_gap"] for h in history if h["brain_age_gap"] is not None]

        return {
            "n_assessments": len(history),
            "mean_estimated_age": round(float(np.mean(ages)), 1),
            "std_estimated_age": round(float(np.std(ages)), 1),
            "mean_bag": round(float(np.mean(bags)), 1) if bags else None,
            "latest_stage": history[-1]["developmental_stage"],
        }

    def reset(self, user_id: str = "default"):
        """Clear history."""
        self._history.pop(user_id, None)

    # ── Feature extraction ──────────────────────────────────────

    def _extract_maturation_features(self, signals: np.ndarray, fs: float) -> Dict:
        """Extract maturation-sensitive features from multichannel EEG."""
        all_powers = []
        all_alpha_peaks = []
        all_exponents = []

        for ch in range(signals.shape[0]):
            sig = signals[ch]
            powers = self._relative_band_powers(sig, fs)
            all_powers.append(powers)

            alpha_peak = self._find_alpha_peak(sig, fs)
            all_alpha_peaks.append(alpha_peak)

            exponent = self._aperiodic_exponent(sig, fs)
            all_exponents.append(exponent)

        # Average across channels
        avg_powers = {}
        for band in ["delta", "theta", "alpha", "beta"]:
            avg_powers[band] = float(np.mean([p[band] for p in all_powers]))

        return {
            "alpha_peak_hz": float(np.mean(all_alpha_peaks)),
            "aperiodic_exponent": float(np.mean(all_exponents)),
            "theta_relative": avg_powers["theta"],
            "alpha_relative": avg_powers["alpha"],
            "beta_relative": avg_powers["beta"],
            "delta_relative": avg_powers["delta"],
            "theta_beta_ratio": avg_powers["theta"] / max(avg_powers["beta"], 1e-10),
            "spectral_entropy": self._spectral_entropy(signals[0], fs),
        }

    def _relative_band_powers(self, signal: np.ndarray, fs: float) -> Dict:
        """Compute relative band powers."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return {"delta": 0.25, "theta": 0.25, "alpha": 0.25, "beta": 0.25}

        try:
            freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return {"delta": 0.25, "theta": 0.25, "alpha": 0.25, "beta": 0.25}

        def bp(lo, hi):
            mask = (freqs >= lo) & (freqs <= hi)
            if not np.any(mask):
                return 0.0
            return float(np.trapezoid(psd[mask], freqs[mask]) if hasattr(np, 'trapezoid')
                         else np.trapz(psd[mask], freqs[mask]))

        total = bp(0.5, 45) + 1e-10
        return {
            "delta": bp(0.5, 4) / total,
            "theta": bp(4, 8) / total,
            "alpha": bp(8, 12) / total,
            "beta": bp(12, 30) / total,
        }

    def _find_alpha_peak(self, signal: np.ndarray, fs: float) -> float:
        """Find alpha peak frequency (Individual Alpha Frequency)."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return 10.0

        try:
            freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return 10.0

        # Search in alpha range (6-13 Hz)
        mask = (freqs >= 6) & (freqs <= 13)
        if not np.any(mask):
            return 10.0

        alpha_freqs = freqs[mask]
        alpha_psd = psd[mask]
        peak_idx = np.argmax(alpha_psd)
        return float(alpha_freqs[peak_idx])

    def _aperiodic_exponent(self, signal: np.ndarray, fs: float) -> float:
        """Estimate 1/f aperiodic exponent from PSD."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return 1.5

        try:
            freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return 1.5

        # Fit log-log line in aperiodic range (1-40 Hz)
        mask = (freqs >= 1) & (freqs <= 40)
        if np.sum(mask) < 5:
            return 1.5

        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask] + 1e-20)

        try:
            slope, _ = np.polyfit(log_f, log_p, 1)
        except Exception:
            return 1.5

        return float(-slope)  # Positive exponent

    def _spectral_entropy(self, signal: np.ndarray, fs: float) -> float:
        """Normalized spectral entropy."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return 0.5

        try:
            freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.5

        # Normalize PSD to probability distribution
        psd_norm = psd / (np.sum(psd) + 1e-10)
        psd_norm = psd_norm[psd_norm > 0]
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        max_entropy = np.log2(len(psd_norm)) if len(psd_norm) > 0 else 1
        return float(entropy / max(max_entropy, 1e-10))

    # ── Brain age estimation ─────────────────────────────────────

    def _estimate_brain_age(self, features: Dict) -> float:
        """Estimate brain age from maturation features.

        Simple feature-based model (not a trained ML model):
        - Alpha peak frequency: 6 Hz → infant, 10 Hz → adult
        - Aperiodic exponent: 2.5 → infant, 1.0 → adult
        - Theta/beta ratio: high → young, low → mature
        """
        alpha_peak = features["alpha_peak_hz"]
        exponent = features["aperiodic_exponent"]
        tbr = features["theta_beta_ratio"]
        entropy = features["spectral_entropy"]

        # Map alpha peak to age (rough linear: 6 Hz → 2 yr, 10.5 Hz → 25 yr)
        age_from_alpha = (alpha_peak - 6) / 4.5 * 23 + 2
        age_from_alpha = float(np.clip(age_from_alpha, 0, 80))

        # Map exponent to age (2.5 → 0 yr, 0.8 → 50 yr)
        age_from_exp = (2.5 - exponent) / 1.7 * 50
        age_from_exp = float(np.clip(age_from_exp, 0, 80))

        # Map TBR to age (high TBR → young)
        age_from_tbr = float(np.clip(30 - tbr * 10, 5, 70))

        # Weighted average
        estimated = 0.40 * age_from_alpha + 0.35 * age_from_exp + 0.25 * age_from_tbr
        return float(np.clip(estimated, 0, 100))

    def _classify_stage(self, features: Dict) -> str:
        """Classify developmental stage from features."""
        alpha_peak = features["alpha_peak_hz"]
        theta_rel = features["theta_relative"]
        exponent = features["aperiodic_exponent"]

        if alpha_peak < 8 or theta_rel > 0.35 or exponent > 2.0:
            return "early_development"
        elif alpha_peak < 9.5 or theta_rel > 0.25 or exponent > 1.5:
            return "developing"
        elif alpha_peak < 10.5 and theta_rel < 0.25:
            return "adolescent"
        else:
            return "mature"

    def _normative_comparison(
        self, features: Dict, age: Optional[float] = None
    ) -> Dict:
        """Compare features to normative ranges."""
        if age is None:
            return {"status": "no_age_provided"}

        if age < 12:
            group = "child"
        elif age < 18:
            group = "adolescent"
        else:
            group = "adult"

        comparisons = {}
        for key in ["alpha_peak_hz", "theta_relative", "aperiodic_exp"]:
            feat_key = key if key != "aperiodic_exp" else "aperiodic_exponent"
            val = features.get(feat_key, 0)
            norm = self._norms.get(key, {}).get(group)
            if norm:
                if val < norm[0]:
                    comparisons[key] = "below_normal"
                elif val > norm[1]:
                    comparisons[key] = "above_normal"
                else:
                    comparisons[key] = "normal"

        return {"age_group": group, "comparisons": comparisons}
