"""Hemispheric balance monitor — track left/right brain asymmetry.

Monitors the balance of neural activity between left and right
hemispheres across all frequency bands. Extends FAA (frontal alpha
asymmetry) to full-spectrum hemispheric analysis.

Channels:
- Left hemisphere: TP9 (ch0), AF7 (ch1)
- Right hemisphere: AF8 (ch2), TP10 (ch3)

Asymmetry is computed per band: log(right) - log(left).
Positive = greater right alpha (= greater left activation).

References:
    Davidson (1992) — Anterior asymmetry and emotion
    Coan & Allen (2004) — Frontal EEG asymmetry
    Reznik & Allen (2018) — Frontal asymmetry as predictor
"""
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


class HemisphericBalanceMonitor:
    """Monitor left/right brain hemispheric balance across bands.

    Computes per-band asymmetry indices and tracks balance over time.
    """

    BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
    }

    # Muse 2 channel mapping
    LEFT_CHANNELS = (0, 1)   # TP9, AF7
    RIGHT_CHANNELS = (2, 3)  # AF8, TP10

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting baseline for asymmetry normalization.

        Args:
            signals: (n_channels, n_samples) EEG (4 channels expected).
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with baseline asymmetries per band.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        asymmetries = self._compute_asymmetries(signals, fs)
        self._baselines[user_id] = asymmetries

        return {
            "baseline_set": True,
            "baseline_asymmetries": {
                k: round(v, 4) for k, v in asymmetries.items()
            },
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess current hemispheric balance.

        Args:
            signals: (n_channels, n_samples) EEG.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with per-band asymmetry, overall balance score,
            dominance label, and deviation from baseline.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        asymmetries = self._compute_asymmetries(signals, fs)
        baseline = self._baselines.get(user_id, {})

        # Deviation from baseline
        deviations = {}
        if baseline:
            for band in asymmetries:
                bl = baseline.get(band, 0)
                deviations[band] = round(asymmetries[band] - bl, 4)

        # Overall balance score: mean absolute asymmetry (0 = perfect balance)
        abs_asym = [abs(v) for v in asymmetries.values()]
        balance_score = float(np.clip(100 - np.mean(abs_asym) * 200, 0, 100))

        # Dominant hemisphere (from alpha asymmetry — most validated)
        alpha_asym = asymmetries.get("alpha", 0)
        if alpha_asym > 0.1:
            dominance = "left_dominant"  # Right alpha higher = left more active
        elif alpha_asym < -0.1:
            dominance = "right_dominant"
        else:
            dominance = "balanced"

        # Emotional valence proxy from alpha asymmetry
        valence = float(np.clip(np.tanh(alpha_asym * 2), -1, 1))

        result = {
            "asymmetries": {k: round(v, 4) for k, v in asymmetries.items()},
            "balance_score": round(balance_score, 1),
            "dominance": dominance,
            "valence_proxy": round(valence, 4),
            "deviations_from_baseline": deviations if deviations else None,
            "has_baseline": bool(baseline),
        }

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

        balance_scores = [h["balance_score"] for h in history]
        dominances = [h["dominance"] for h in history]
        dom_counts = {}
        for d in dominances:
            dom_counts[d] = dom_counts.get(d, 0) + 1

        return {
            "n_epochs": len(history),
            "has_baseline": user_id in self._baselines,
            "mean_balance": round(float(np.mean(balance_scores)), 1),
            "dominant_pattern": max(dom_counts, key=dom_counts.get),
            "dominance_distribution": dom_counts,
            "mean_valence": round(float(np.mean(
                [h["valence_proxy"] for h in history]
            )), 4),
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

    def _compute_asymmetries(self, signals: np.ndarray, fs: float) -> Dict:
        """Compute per-band hemispheric asymmetry."""
        n_ch = signals.shape[0]
        result = {}

        for band, (low, high) in self.BANDS.items():
            # Left hemisphere power
            left_powers = []
            for ch in self.LEFT_CHANNELS:
                if ch < n_ch:
                    left_powers.append(self._band_power(signals[ch], fs, low, high))

            # Right hemisphere power
            right_powers = []
            for ch in self.RIGHT_CHANNELS:
                if ch < n_ch:
                    right_powers.append(self._band_power(signals[ch], fs, low, high))

            left_mean = float(np.mean(left_powers)) if left_powers else 1e-10
            right_mean = float(np.mean(right_powers)) if right_powers else 1e-10

            # Asymmetry: log(right) - log(left)
            if left_mean > 1e-10 and right_mean > 1e-10:
                result[band] = float(np.log(right_mean) - np.log(left_mean))
            else:
                result[band] = 0.0

        return result

    def _band_power(self, sig: np.ndarray, fs: float, low: float, high: float) -> float:
        """Compute band power for a single channel."""
        nperseg = min(len(sig), int(fs * 2))
        if nperseg < 4:
            return 0.0
        try:
            freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=nperseg)
            mask = (freqs >= low) & (freqs <= high)
            if not np.any(mask):
                return 0.0
            if hasattr(np, "trapezoid"):
                return float(np.trapezoid(psd[mask], freqs[mask]))
            return float(np.trapz(psd[mask], freqs[mask]))
        except Exception:
            return 0.0
