"""Cognitive reserve estimator from EEG complexity metrics.

Estimates brain resilience / cognitive reserve using spectral entropy,
aperiodic (1/f) slope, alpha peak frequency, and cross-channel coherence.
Higher reserve = more complex, efficient neural processing = better
resistance to age-related cognitive decline.

Features used:
- Spectral entropy (higher = more complex = more reserve)
- Aperiodic exponent (flatter 1/f = younger/healthier brain)
- Alpha peak frequency (higher in cognitively healthy individuals)
- Inter-channel coherence (higher = more efficient network)
- Theta/beta ratio (lower = better executive function)

References:
    Stern (2009) — Cognitive reserve theory
    Scally et al. (2018) — EEG spectral measures of cognitive reserve
    Cesnaite et al. (2023) — Individual alpha frequency and cognitive reserve
"""
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


class CognitiveReserveEstimator:
    """Estimate cognitive reserve from EEG complexity metrics.

    Higher reserve scores indicate more complex, efficient neural
    processing and better resilience to cognitive decline.
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        age: Optional[int] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record baseline EEG for reserve estimation.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate.
            age: User's age (for normative comparison).
            user_id: User identifier.

        Returns:
            Dict with baseline metrics.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        metrics = self._compute_metrics(signals, fs)
        metrics["age"] = age
        self._baselines[user_id] = metrics

        return {
            "baseline_set": True,
            "metrics": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in metrics.items()},
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess cognitive reserve from current EEG.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with reserve_score (0-100), reserve_level, component scores,
            normative comparison, and recommendations.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        metrics = self._compute_metrics(signals, fs)
        baseline = self._baselines.get(user_id, {})

        # Component scores (0-100 each)
        entropy_score = self._score_entropy(metrics["spectral_entropy"])
        slope_score = self._score_aperiodic(metrics["aperiodic_exponent"])
        apf_score = self._score_alpha_peak(metrics["alpha_peak_freq"])
        coherence_score = self._score_coherence(metrics["mean_coherence"])
        tbr_score = self._score_tbr(metrics["theta_beta_ratio"])

        # Weighted reserve score
        reserve = (
            0.25 * entropy_score
            + 0.20 * slope_score
            + 0.20 * apf_score
            + 0.20 * coherence_score
            + 0.15 * tbr_score
        )
        reserve = float(np.clip(reserve, 0, 100))

        # Level
        if reserve >= 80:
            level = "high"
        elif reserve >= 60:
            level = "moderate_high"
        elif reserve >= 40:
            level = "moderate"
        elif reserve >= 20:
            level = "low_moderate"
        else:
            level = "low"

        # Normative comparison
        age = baseline.get("age")
        normative = self._normative_comparison(reserve, age) if age else None

        # Recommendations
        recommendations = self._get_recommendations(
            entropy_score, slope_score, apf_score, coherence_score, tbr_score
        )

        result = {
            "reserve_score": round(reserve, 1),
            "reserve_level": level,
            "components": {
                "spectral_entropy": round(entropy_score, 1),
                "aperiodic_slope": round(slope_score, 1),
                "alpha_peak_freq": round(apf_score, 1),
                "coherence": round(coherence_score, 1),
                "theta_beta_ratio": round(tbr_score, 1),
            },
            "raw_metrics": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in metrics.items()},
            "normative": normative,
            "recommendations": recommendations,
        }

        # Store history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > 365:
            self._history[user_id] = self._history[user_id][-365:]

        return result

    def get_trajectory(self, user_id: str = "default") -> Dict:
        """Get reserve score trajectory over time."""
        history = self._history.get(user_id, [])
        if len(history) < 2:
            return {"trend": "insufficient_data", "n_assessments": len(history)}

        scores = [h["reserve_score"] for h in history]
        slope = float(np.polyfit(range(len(scores)), scores, 1)[0])

        if slope > 0.5:
            trend = "improving"
        elif slope < -0.5:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(slope, 3),
            "mean_score": round(float(np.mean(scores)), 1),
            "n_assessments": len(history),
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get assessment history."""
        history = self._history.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return history

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session statistics."""
        history = self._history.get(user_id, [])
        if not history:
            return {"n_assessments": 0, "has_baseline": user_id in self._baselines}

        scores = [h["reserve_score"] for h in history]
        return {
            "n_assessments": len(history),
            "has_baseline": user_id in self._baselines,
            "mean_reserve": round(float(np.mean(scores)), 1),
            "std_reserve": round(float(np.std(scores)), 1),
            "latest_level": history[-1]["reserve_level"],
        }

    def reset(self, user_id: str = "default"):
        """Clear all data for user."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # ── Private helpers ─────────────────────────────────────────

    def _compute_metrics(self, signals: np.ndarray, fs: float) -> Dict:
        """Compute all reserve-relevant EEG metrics."""
        n_ch = signals.shape[0]

        # Per-channel spectral entropy
        entropies = []
        alpha_peaks = []
        aperiodic_exps = []
        theta_powers = []
        beta_powers = []

        for ch in range(n_ch):
            sig = signals[ch]
            nperseg = min(len(sig), int(fs * 2))
            if nperseg < 4:
                continue

            try:
                freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=nperseg)
            except Exception:
                continue

            # Spectral entropy
            psd_norm = psd / (np.sum(psd) + 1e-10)
            psd_norm = psd_norm[psd_norm > 0]
            se = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            se_max = np.log2(len(psd_norm)) if len(psd_norm) > 0 else 1
            entropies.append(se / (se_max + 1e-10))

            # Alpha peak frequency
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            if np.any(alpha_mask):
                alpha_psd = psd[alpha_mask]
                alpha_freqs = freqs[alpha_mask]
                peak_idx = np.argmax(alpha_psd)
                alpha_peaks.append(float(alpha_freqs[peak_idx]))

            # Aperiodic exponent (1/f slope in log-log)
            valid = (freqs >= 2) & (freqs <= 40)
            if np.sum(valid) > 2:
                log_f = np.log10(freqs[valid] + 1e-10)
                log_p = np.log10(psd[valid] + 1e-10)
                try:
                    slope, _ = np.polyfit(log_f, log_p, 1)
                    aperiodic_exps.append(-slope)
                except Exception:
                    pass

            # Theta and beta for TBR
            theta_mask = (freqs >= 4) & (freqs <= 8)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            if np.any(theta_mask):
                if hasattr(np, "trapezoid"):
                    theta_powers.append(float(np.trapezoid(psd[theta_mask], freqs[theta_mask])))
                else:
                    theta_powers.append(float(np.trapz(psd[theta_mask], freqs[theta_mask])))
            if np.any(beta_mask):
                if hasattr(np, "trapezoid"):
                    beta_powers.append(float(np.trapezoid(psd[beta_mask], freqs[beta_mask])))
                else:
                    beta_powers.append(float(np.trapz(psd[beta_mask], freqs[beta_mask])))

        # Inter-channel coherence (mean across pairs)
        coherences = []
        if n_ch >= 2:
            for i in range(n_ch):
                for j in range(i + 1, n_ch):
                    nperseg = min(signals.shape[1], int(fs * 2))
                    if nperseg < 4:
                        continue
                    try:
                        f_coh, coh = scipy_signal.coherence(
                            signals[i], signals[j], fs=fs, nperseg=nperseg
                        )
                        alpha_mask = (f_coh >= 8) & (f_coh <= 12)
                        if np.any(alpha_mask):
                            coherences.append(float(np.mean(coh[alpha_mask])))
                    except Exception:
                        pass

        mean_theta = float(np.mean(theta_powers)) if theta_powers else 0.0
        mean_beta = float(np.mean(beta_powers)) if beta_powers else 1e-10

        return {
            "spectral_entropy": float(np.mean(entropies)) if entropies else 0.5,
            "alpha_peak_freq": float(np.mean(alpha_peaks)) if alpha_peaks else 10.0,
            "aperiodic_exponent": float(np.mean(aperiodic_exps)) if aperiodic_exps else 1.5,
            "mean_coherence": float(np.mean(coherences)) if coherences else 0.5,
            "theta_beta_ratio": mean_theta / (mean_beta + 1e-10),
        }

    def _score_entropy(self, entropy: float) -> float:
        """Score spectral entropy. Higher = more reserve."""
        # Normalized entropy 0-1, optimal ~0.7-0.9
        return float(np.clip(entropy * 120, 0, 100))

    def _score_aperiodic(self, exponent: float) -> float:
        """Score aperiodic exponent. Flatter (lower) = younger brain."""
        # Typical range: 1.0 (young) to 2.5 (elderly)
        # Lower = better
        return float(np.clip((2.5 - exponent) / 1.5 * 100, 0, 100))

    def _score_alpha_peak(self, apf: float) -> float:
        """Score alpha peak frequency. Higher = better reserve."""
        # Typical range: 8-12 Hz, optimal ~10-11 Hz
        return float(np.clip((apf - 7) / 5 * 100, 0, 100))

    def _score_coherence(self, coherence: float) -> float:
        """Score mean coherence. Higher = better network efficiency."""
        return float(np.clip(coherence * 100, 0, 100))

    def _score_tbr(self, tbr: float) -> float:
        """Score theta/beta ratio. Lower = better executive function."""
        # Typical healthy adult TBR: 1.5-3.0
        # Higher = worse (ADHD-like pattern)
        return float(np.clip((5 - tbr) / 4 * 100, 0, 100))

    def _normative_comparison(self, score: float, age: int) -> Dict:
        """Compare reserve score against age-normative data."""
        # Simplified normative ranges
        if age < 30:
            expected_range = (55, 85)
        elif age < 50:
            expected_range = (45, 75)
        elif age < 70:
            expected_range = (35, 65)
        else:
            expected_range = (25, 55)

        mid = (expected_range[0] + expected_range[1]) / 2
        if score >= expected_range[1]:
            comparison = "above_average"
        elif score >= mid:
            comparison = "average"
        elif score >= expected_range[0]:
            comparison = "below_average"
        else:
            comparison = "significantly_below"

        return {
            "age_group_range": expected_range,
            "comparison": comparison,
            "percentile_estimate": int(np.clip(
                50 + (score - mid) / (expected_range[1] - expected_range[0]) * 80,
                5, 95
            )),
        }

    def _get_recommendations(self, entropy, slope, apf, coherence, tbr) -> List[str]:
        """Generate recommendations based on component scores."""
        recs = []
        if entropy < 50:
            recs.append("cognitive_training: engage in novel learning and problem-solving")
        if slope < 40:
            recs.append("sleep_quality: prioritize deep sleep for neural restoration")
        if apf < 50:
            recs.append("aerobic_exercise: regular cardio increases alpha peak frequency")
        if coherence < 50:
            recs.append("meditation: increases inter-hemispheric coherence")
        if tbr < 40:
            recs.append("attention_training: neurofeedback to reduce theta/beta ratio")
        return recs
