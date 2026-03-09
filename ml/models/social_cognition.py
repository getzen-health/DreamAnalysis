"""Social cognition detector — EEG markers of social processing.

Detects neural signatures of social cognitive processes: empathy,
theory of mind, social engagement, and interpersonal synchrony.
Uses mu suppression (mirror neuron system proxy) and frontal theta
as primary markers.

Key EEG markers:
- Mu suppression (8-13 Hz at temporal sites): mirror neuron activation
  during social observation/empathy (Perry et al., 2010)
- Frontal theta increase: cognitive empathy / mentalizing (Mu et al., 2008)
- Alpha asymmetry shift during social vs non-social: approach motivation

References:
    Perry et al. (2010) — Mu suppression and empathy
    Mu et al. (2008) — EEG correlates of empathic concern
    Hari & Kujala (2009) — Brain basis for social interaction
"""
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


class SocialCognitionDetector:
    """Detect social cognitive processes from EEG.

    Measures empathy, mentalizing, and social engagement via
    mu suppression at temporal sites and frontal theta.
    """

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
        """Record non-social resting baseline.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with baseline mu and theta powers.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        mu_power = self._temporal_mu_power(signals, fs)
        theta_power = self._frontal_theta_power(signals, fs)

        baseline = {"mu": mu_power, "theta": theta_power}
        self._baselines[user_id] = baseline

        return {
            "baseline_set": True,
            "baseline_mu": round(mu_power, 6),
            "baseline_theta": round(theta_power, 6),
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        context: str = "observation",
        user_id: str = "default",
    ) -> Dict:
        """Assess social cognition state from EEG.

        Args:
            signals: (n_channels, n_samples) EEG epoch.
            fs: Sampling rate.
            context: Social context — "observation", "conversation",
                     "empathy_task", "mentalizing".
            user_id: User identifier.

        Returns:
            Dict with empathy_index, mentalizing_index, social_engagement,
            mu_suppression, social_state, and recommendations.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        baseline = self._baselines.get(user_id, {})
        bl_mu = baseline.get("mu", 0)
        bl_theta = baseline.get("theta", 0)

        # Current powers
        mu_power = self._temporal_mu_power(signals, fs)
        theta_power = self._frontal_theta_power(signals, fs)
        alpha_power = self._frontal_alpha_power(signals, fs)
        beta_power = self._frontal_beta_power(signals, fs)

        # Mu suppression ratio (< 1 means suppression = empathy activation)
        has_baseline = bl_mu > 1e-10
        if has_baseline:
            mu_ratio = mu_power / bl_mu
        else:
            mu_ratio = 1.0

        # Mu suppression index (0-1, higher = more suppression)
        mu_suppression = float(np.clip(1.0 - mu_ratio, 0, 1))

        # Theta increase ratio
        if bl_theta > 1e-10:
            theta_ratio = theta_power / bl_theta
        else:
            theta_ratio = 1.0

        # Mentalizing index from frontal theta increase
        mentalizing = float(np.clip((theta_ratio - 1.0) * 2, 0, 1))

        # Empathy index: mu suppression + theta increase
        empathy = float(np.clip(
            0.6 * mu_suppression + 0.4 * mentalizing, 0, 1
        ))

        # Social engagement: combination of empathy + beta (active processing)
        beta_engagement = beta_power / (alpha_power + beta_power + 1e-10)
        social_engagement = float(np.clip(
            0.5 * empathy + 0.3 * beta_engagement + 0.2 * mentalizing, 0, 1
        ))

        # Social state classification
        if empathy >= 0.6 and mentalizing >= 0.4:
            social_state = "deeply_engaged"
        elif empathy >= 0.4 or mentalizing >= 0.4:
            social_state = "socially_attentive"
        elif social_engagement >= 0.3:
            social_state = "passively_observing"
        else:
            social_state = "socially_disengaged"

        result = {
            "empathy_index": round(empathy, 4),
            "mentalizing_index": round(mentalizing, 4),
            "social_engagement": round(social_engagement, 4),
            "mu_suppression": round(mu_suppression, 4),
            "mu_ratio": round(mu_ratio, 4),
            "theta_ratio": round(theta_ratio, 4),
            "social_state": social_state,
            "context": context,
            "has_baseline": has_baseline,
            "recommendations": self._get_recommendations(social_state, empathy),
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

        empathy_scores = [h["empathy_index"] for h in history]
        states = [h["social_state"] for h in history]
        state_counts = {}
        for s in states:
            state_counts[s] = state_counts.get(s, 0) + 1

        return {
            "n_epochs": len(history),
            "has_baseline": user_id in self._baselines,
            "mean_empathy": round(float(np.mean(empathy_scores)), 4),
            "max_empathy": round(float(np.max(empathy_scores)), 4),
            "dominant_state": max(state_counts, key=state_counts.get),
            "state_distribution": state_counts,
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

    def _temporal_mu_power(self, signals: np.ndarray, fs: float) -> float:
        """Mu rhythm (8-13 Hz) at temporal sites TP9=ch0, TP10=ch3."""
        channels = [0, 3] if signals.shape[0] >= 4 else [0]
        powers = [self._band_power(signals[ch], fs, 8, 13)
                  for ch in channels if ch < signals.shape[0]]
        return float(np.mean(powers)) if powers else 0.0

    def _frontal_theta_power(self, signals: np.ndarray, fs: float) -> float:
        """Frontal theta (4-8 Hz) at AF7=ch1, AF8=ch2."""
        channels = [1, 2] if signals.shape[0] >= 3 else [0]
        powers = [self._band_power(signals[ch], fs, 4, 8)
                  for ch in channels if ch < signals.shape[0]]
        return float(np.mean(powers)) if powers else 0.0

    def _frontal_alpha_power(self, signals: np.ndarray, fs: float) -> float:
        """Frontal alpha (8-12 Hz)."""
        channels = [1, 2] if signals.shape[0] >= 3 else [0]
        powers = [self._band_power(signals[ch], fs, 8, 12)
                  for ch in channels if ch < signals.shape[0]]
        return float(np.mean(powers)) if powers else 0.0

    def _frontal_beta_power(self, signals: np.ndarray, fs: float) -> float:
        """Frontal beta (12-30 Hz)."""
        channels = [1, 2] if signals.shape[0] >= 3 else [0]
        powers = [self._band_power(signals[ch], fs, 12, 30)
                  for ch in channels if ch < signals.shape[0]]
        return float(np.mean(powers)) if powers else 0.0

    def _get_recommendations(self, state: str, empathy: float) -> List[str]:
        """Generate social cognition recommendations."""
        recs = []
        if state == "socially_disengaged":
            recs.append("engagement: try active listening or eye contact")
        if empathy < 0.3:
            recs.append("empathy_practice: perspective-taking exercises")
        if state in ("passively_observing", "socially_disengaged"):
            recs.append("connection: mirror neuron activation through social mirroring")
        return recs
