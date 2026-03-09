"""Adaptive workload controller for VR/AR/gaming applications.

Monitors cognitive load in real time and outputs adaptation commands
to adjust task difficulty, information density, or pacing. Designed
for integration with VR/AR headsets that include EEG sensors.

Workload zones:
- Underload: low theta, low beta → increase difficulty
- Optimal: moderate theta, moderate beta → maintain
- Overload: high theta, high beta, low alpha → reduce difficulty
- Fatigue: rising theta, declining beta over time → suggest break

References:
    Kothe & Makeig (2011) — Real-time cognitive workload estimation
    Mühl et al. (2014) — Survey of affective BCI for games
    Wobrock et al. (2018) — Continuous workload estimation for adaptive VR
"""
from typing import Dict, List, Optional

import numpy as np
_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
from scipy import signal as scipy_signal


class WorkloadAdapter:
    """Real-time cognitive workload monitor with adaptation commands.

    Outputs difficulty adjustment signals for VR/AR/game applications
    based on EEG workload estimation.
    """

    def __init__(self, fs: float = 256.0, adaptation_rate: float = 0.1):
        """
        Args:
            fs: EEG sampling rate.
            adaptation_rate: How fast difficulty adjusts (0-1).
        """
        self._fs = fs
        self._adaptation_rate = adaptation_rate
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}
        self._difficulty: Dict[str, float] = {}

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting baseline for workload normalization.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with baseline band powers.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        powers = self._extract_powers(signals, fs)
        self._baselines[user_id] = powers
        self._difficulty[user_id] = 0.5  # Start at medium difficulty

        return {
            "baseline_set": True,
            "band_powers": {k: round(v, 6) for k, v in powers.items()},
            "initial_difficulty": 0.5,
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess cognitive workload and output adaptation command.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG epoch.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with workload_score, workload_zone, difficulty_adjustment,
            current_difficulty, adaptation_command, and fatigue_index.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        powers = self._extract_powers(signals, fs)
        baseline = self._baselines.get(user_id, {})

        # Compute workload index
        theta = powers.get("theta", 0)
        alpha = powers.get("alpha", 0)
        beta = powers.get("beta", 0)
        total = theta + alpha + beta + 1e-10

        # Workload: frontal theta increase + alpha suppression
        theta_frac = theta / total
        alpha_frac = alpha / total
        beta_frac = beta / total

        # Workload score (0-1): high theta + low alpha = high workload
        workload = float(np.clip(
            0.45 * theta_frac * 3 + 0.35 * (1 - alpha_frac) + 0.20 * beta_frac * 2,
            0, 1
        ))

        # Fatigue detection: rising theta ratio over time
        fatigue = self._compute_fatigue(user_id, theta_frac)

        # Zone classification
        if fatigue > 0.7:
            zone = "fatigue"
        elif workload >= 0.7:
            zone = "overload"
        elif workload >= 0.35:
            zone = "optimal"
        else:
            zone = "underload"

        # Difficulty adaptation
        current_diff = self._difficulty.get(user_id, 0.5)
        if zone == "overload":
            adjustment = -self._adaptation_rate
        elif zone == "underload":
            adjustment = self._adaptation_rate
        elif zone == "fatigue":
            adjustment = -self._adaptation_rate * 1.5
        else:
            adjustment = 0.0

        new_diff = float(np.clip(current_diff + adjustment, 0, 1))
        self._difficulty[user_id] = new_diff

        # Adaptation command
        if zone == "fatigue":
            command = "reduce_and_break"
            message = "User fatigued — reduce load and suggest break"
        elif zone == "overload":
            command = "reduce_difficulty"
            message = "Cognitive overload — simplify task"
        elif zone == "underload":
            command = "increase_difficulty"
            message = "User underloaded — increase challenge"
        else:
            command = "maintain"
            message = "Workload in optimal zone"

        result = {
            "workload_score": round(workload, 4),
            "workload_zone": zone,
            "fatigue_index": round(fatigue, 4),
            "difficulty_adjustment": round(adjustment, 4),
            "current_difficulty": round(new_diff, 4),
            "adaptation_command": command,
            "adaptation_message": message,
            "band_powers": {k: round(v, 6) for k, v in powers.items()},
            "has_baseline": user_id in self._baselines,
        }

        # Store history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > 1000:
            self._history[user_id] = self._history[user_id][-1000:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session statistics."""
        history = self._history.get(user_id, [])
        if not history:
            return {"n_epochs": 0, "has_baseline": user_id in self._baselines}

        workloads = [h["workload_score"] for h in history]
        zones = [h["workload_zone"] for h in history]
        zone_counts = {}
        for z in zones:
            zone_counts[z] = zone_counts.get(z, 0) + 1

        optimal_pct = zone_counts.get("optimal", 0) / len(zones)

        return {
            "n_epochs": len(history),
            "has_baseline": user_id in self._baselines,
            "mean_workload": round(float(np.mean(workloads)), 4),
            "current_difficulty": round(self._difficulty.get(user_id, 0.5), 4),
            "zone_distribution": zone_counts,
            "optimal_percentage": round(optimal_pct, 4),
            "fatigue_index": round(history[-1]["fatigue_index"], 4),
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
        self._difficulty.pop(user_id, None)

    # ── Private helpers ─────────────────────────────────────────

    def _extract_powers(self, signals: np.ndarray, fs: float) -> Dict:
        """Extract band powers averaged across channels."""
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
            result[band] = float(np.mean(powers)) if powers else 0.0
        return result

    def _compute_fatigue(self, user_id: str, theta_frac: float) -> float:
        """Compute fatigue from theta fraction trend."""
        history = self._history.get(user_id, [])
        if len(history) < 5:
            return 0.0

        # Look at recent workload scores — rising trend = fatigue
        recent = [h["workload_score"] for h in history[-20:]]
        if len(recent) < 5:
            return 0.0

        # Simple: compare first half vs second half
        first = np.mean(recent[:len(recent)//2])
        second = np.mean(recent[len(recent)//2:])
        drift = second - first

        # Also consider sustained high workload
        sustained_high = sum(1 for w in recent[-10:] if w > 0.6) / min(10, len(recent))

        fatigue = float(np.clip(drift * 3 + sustained_high * 0.5, 0, 1))
        return fatigue
