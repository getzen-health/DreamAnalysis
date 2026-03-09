"""Circadian rhythm monitor — track alertness and optimal timing.

Uses EEG markers to estimate position in the circadian cycle and
predict optimal windows for cognitive tasks, sleep, and exercise.

Key EEG markers of circadian position:
- Alpha power: peaks in mid-afternoon, lowest at night
- Theta/alpha ratio: rises with sleep pressure (Process S)
- Alpha peak frequency: decreases with fatigue across the day
- Beta power: reflects homeostatic sleep pressure

References:
    Borbely (1982) — Two-process model of sleep regulation
    Cajochen et al. (1999) — EEG markers of circadian phase
    Akerstedt & Gillberg (1990) — Subjective and EEG sleepiness
"""
from typing import Dict, List, Optional

import numpy as np
_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
from scipy import signal as scipy_signal


class CircadianMonitor:
    """Track circadian alertness and optimal cognitive windows.

    Estimates alertness level and position in the circadian cycle
    from EEG markers, providing timing recommendations.
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    def set_morning_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        hour_of_day: int = 9,
        user_id: str = "default",
    ) -> Dict:
        """Record morning baseline for circadian tracking.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate.
            hour_of_day: Current hour (0-23).
            user_id: User identifier.

        Returns:
            Dict with baseline metrics.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        metrics = self._extract_metrics(signals, fs)
        metrics["hour"] = hour_of_day
        self._baselines[user_id] = metrics

        return {
            "baseline_set": True,
            "hour_of_day": hour_of_day,
            "metrics": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in metrics.items()},
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        hour_of_day: Optional[int] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess current alertness and circadian position.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate.
            hour_of_day: Current hour (0-23), for context.
            user_id: User identifier.

        Returns:
            Dict with alertness_score, alertness_level, sleep_pressure,
            optimal_window, and recommendations.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        metrics = self._extract_metrics(signals, fs)
        baseline = self._baselines.get(user_id, {})

        # Alertness from alpha/theta balance
        alpha = metrics["alpha_power"]
        theta = metrics["theta_power"]
        beta = metrics["beta_power"]
        total = alpha + theta + beta + 1e-10

        # Alertness: high alpha + beta, low theta
        alpha_frac = alpha / total
        theta_frac = theta / total
        beta_frac = beta / total

        alertness = float(np.clip(
            0.40 * alpha_frac * 3 + 0.35 * beta_frac * 2 - 0.25 * theta_frac * 3 + 0.3,
            0, 1
        ))
        alertness_score = round(alertness * 100, 1)

        # Sleep pressure (Process S proxy): theta/alpha ratio increase
        theta_alpha = theta / (alpha + 1e-10)
        if baseline:
            bl_ta = baseline.get("theta_power", 1) / (baseline.get("alpha_power", 1) + 1e-10)
            sleep_pressure = float(np.clip((theta_alpha / (bl_ta + 1e-10) - 1) * 2, 0, 1))
        else:
            sleep_pressure = float(np.clip(theta_alpha / 3, 0, 1))

        # Alpha peak frequency shift (lower = more fatigued)
        apf = metrics["alpha_peak_freq"]
        apf_fatigue = float(np.clip((11 - apf) / 3, 0, 1))

        # Alertness level
        if alertness_score >= 80:
            level = "peak"
        elif alertness_score >= 60:
            level = "alert"
        elif alertness_score >= 40:
            level = "moderate"
        elif alertness_score >= 20:
            level = "drowsy"
        else:
            level = "very_drowsy"

        # Optimal window recommendation
        if hour_of_day is not None:
            optimal = self._get_optimal_window(alertness, sleep_pressure, hour_of_day)
        else:
            optimal = None

        # Recommendations
        recommendations = self._get_recommendations(level, sleep_pressure, apf_fatigue)

        result = {
            "alertness_score": alertness_score,
            "alertness_level": level,
            "sleep_pressure": round(sleep_pressure, 4),
            "apf_fatigue": round(apf_fatigue, 4),
            "alpha_peak_freq": round(apf, 2),
            "optimal_window": optimal,
            "recommendations": recommendations,
            "hour_of_day": hour_of_day,
            "has_baseline": bool(baseline),
        }

        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > 500:
            self._history[user_id] = self._history[user_id][-500:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session statistics."""
        history = self._history.get(user_id, [])
        if not history:
            return {"n_epochs": 0, "has_baseline": user_id in self._baselines}

        scores = [h["alertness_score"] for h in history]
        levels = [h["alertness_level"] for h in history]
        level_counts = {}
        for l in levels:
            level_counts[l] = level_counts.get(l, 0) + 1

        return {
            "n_epochs": len(history),
            "has_baseline": user_id in self._baselines,
            "mean_alertness": round(float(np.mean(scores)), 1),
            "peak_alertness": round(float(np.max(scores)), 1),
            "dominant_level": max(level_counts, key=level_counts.get),
            "level_distribution": level_counts,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get assessment history."""
        history = self._history.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return history

    def reset(self, user_id: str = "default"):
        """Clear all data."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # ── Private helpers ─────────────────────────────────────────

    def _extract_metrics(self, signals: np.ndarray, fs: float) -> Dict:
        """Extract circadian-relevant EEG metrics."""
        bands = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
        result = {}

        for band, (low, high) in bands.items():
            powers = []
            for ch in range(signals.shape[0]):
                nperseg = min(len(signals[ch]), int(fs * 2))
                if nperseg < 4:
                    continue
                try:
                    freqs, psd = scipy_signal.welch(signals[ch], fs=fs, nperseg=nperseg)
                    mask = (freqs >= low) & (freqs <= high)
                    if np.any(mask):
                        if hasattr(np, "trapezoid"):
                            powers.append(float(_trapezoid(psd[mask], freqs[mask])))
                        else:
                            powers.append(float(np.trapz(psd[mask], freqs[mask])))
                except Exception:
                    pass
            result[f"{band}_power"] = float(np.mean(powers)) if powers else 0.0

        # Alpha peak frequency
        alpha_peaks = []
        for ch in range(signals.shape[0]):
            nperseg = min(len(signals[ch]), int(fs * 2))
            if nperseg < 4:
                continue
            try:
                freqs, psd = scipy_signal.welch(signals[ch], fs=fs, nperseg=nperseg)
                mask = (freqs >= 8) & (freqs <= 13)
                if np.any(mask):
                    peak_idx = np.argmax(psd[mask])
                    alpha_peaks.append(float(freqs[mask][peak_idx]))
            except Exception:
                pass

        result["alpha_peak_freq"] = float(np.mean(alpha_peaks)) if alpha_peaks else 10.0
        return result

    def _get_optimal_window(self, alertness: float, sleep_pressure: float,
                            hour: int) -> Dict:
        """Predict optimal windows based on current state and time."""
        # Best cognitive work: morning (9-11) or after lunch dip (15-17)
        if alertness > 0.6 and sleep_pressure < 0.4:
            task = "deep_work"
        elif alertness > 0.4:
            task = "routine_tasks"
        else:
            task = "rest_or_nap"

        # Estimate hours until optimal sleep
        if hour < 14:
            hours_until_sleep = max(0, 22 - hour)
        elif sleep_pressure > 0.6:
            hours_until_sleep = max(0, 2)  # High pressure → sleep soon
        else:
            hours_until_sleep = max(0, 22 - hour)

        return {
            "recommended_task": task,
            "estimated_hours_until_sleep": hours_until_sleep,
            "nap_recommended": sleep_pressure > 0.5 and 12 <= hour <= 16,
        }

    def _get_recommendations(self, level: str, sleep_pressure: float,
                             apf_fatigue: float) -> List[str]:
        """Generate circadian-aware recommendations."""
        recs = []
        if level in ("drowsy", "very_drowsy"):
            recs.append("alertness: take a 20-min nap or go for a walk")
        if sleep_pressure > 0.6:
            recs.append("sleep_pressure: high — plan for earlier bedtime")
        if apf_fatigue > 0.5:
            recs.append("mental_fatigue: alpha peak shifted — take a cognitive break")
        if level == "peak":
            recs.append("optimal: tackle your most demanding task now")
        return recs
