"""Decision confidence and risk-taking detection from EEG.

Detects decision-related brain states from 4-channel Muse 2 EEG using
validated biomarkers from the cognitive neuroscience literature:

1. Frontal theta (AF7/AF8, 4-8 Hz) -- increases during deliberation and
   conflict monitoring (Cavanagh & Frank 2014, midfrontal theta).
2. Beta desynchronization -- precedes confident decisions; high beta
   at frontal sites indicates motor preparation / resolved state.
3. Alpha power -- reflects uncertainty/disengagement; high alpha at
   frontal sites = idle/disengaged cortex.
4. Frontal alpha asymmetry (FAA) -- approach vs. avoidance motivation
   in risky decisions (Davidson 1992, Harmon-Jones).
5. Theta/beta ratio -- deliberation intensity marker.

Muse 2 channels: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10 (256 Hz).

Output:
    decision_confidence: 0-100 (how confident the brain state appears)
    risk_profile: risk_seeking / risk_neutral / risk_averse
    deliberation_intensity: 0-1 (frontal theta level)
    approach_motivation: -1 to 1 (FAA-based)
    cognitive_conflict: 0-1 (theta power at frontal sites)
    decision_readiness: ready / deliberating / uncertain / disengaged

References:
    Cavanagh & Frank (2014) -- Frontal theta as mechanism for cognitive control
    Davidson (1992) -- Anterior brain asymmetry and emotion/motivation
    Harmon-Jones et al. -- FAA indexes approach motivation, not pure valence
"""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import welch

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# Muse 2 channel indices
_CH_TP9, _CH_AF7, _CH_AF8, _CH_TP10 = 0, 1, 2, 3

# EEG frequency bands (Hz)
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "low_beta": (12.0, 20.0),
    "high_beta": (20.0, 30.0),
}

# History cap per user
_MAX_HISTORY = 500


class DecisionDetector:
    """Detect decision confidence, risk profile, and cognitive conflict from EEG.

    Designed for 4-channel Muse 2 (TP9, AF7, AF8, TP10).
    Degrades gracefully to single-channel operation.
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        self._history: Dict[str, List[Dict]] = {}
        self._baselines: Dict[str, Dict[str, float]] = {}

    # ── Public API ────────────────────────────────────────────

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline for per-user normalization.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate (defaults to constructor value).
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set (bool) and baseline_metrics (dict).
        """
        fs = fs or self._fs
        signals = self._ensure_2d(signals)
        metrics = self._extract_metrics(signals, fs)
        self._baselines[user_id] = metrics
        return {
            "baseline_set": True,
            "baseline_metrics": {k: round(v, 4) for k, v in metrics.items()},
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess decision-related brain state from EEG.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
                     For Muse 2: (4, n_samples) with TP9, AF7, AF8, TP10.
            fs: Sampling rate in Hz (defaults to constructor value).
            user_id: User identifier for multi-user support.

        Returns:
            Dict with decision_confidence (0-100), risk_profile, deliberation_intensity,
            approach_motivation, cognitive_conflict, decision_readiness, has_baseline.
        """
        fs = fs or self._fs
        signals = self._ensure_2d(signals)
        metrics = self._extract_metrics(signals, fs)
        has_baseline = user_id in self._baselines

        # Normalize against baseline if available
        if has_baseline:
            metrics = self._normalize_metrics(metrics, self._baselines[user_id])

        # Compute composite scores
        deliberation = self._compute_deliberation(metrics)
        conflict = self._compute_conflict(metrics)
        approach = self._compute_approach_motivation(metrics, signals.shape[0])
        confidence = self._compute_confidence(metrics, deliberation, conflict)
        risk_profile = self._classify_risk(approach, metrics)
        readiness = self._classify_readiness(confidence, conflict, metrics)

        result = {
            "decision_confidence": round(float(confidence), 1),
            "risk_profile": risk_profile,
            "deliberation_intensity": round(float(deliberation), 3),
            "approach_motivation": round(float(approach), 3),
            "cognitive_conflict": round(float(conflict), 3),
            "decision_readiness": readiness,
            "has_baseline": has_baseline,
        }

        # Store in history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > _MAX_HISTORY:
            self._history[user_id] = self._history[user_id][-_MAX_HISTORY:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Summary statistics for the session.

        Returns:
            Dict with n_epochs, mean_confidence, dominant_profile, dominant_readiness.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {
                "n_epochs": 0,
                "mean_confidence": 0.0,
                "dominant_profile": "risk_neutral",
                "dominant_readiness": "uncertain",
            }

        confidences = [h["decision_confidence"] for h in history]
        profiles = [h["risk_profile"] for h in history]
        readiness_vals = [h["decision_readiness"] for h in history]

        return {
            "n_epochs": len(history),
            "mean_confidence": round(float(np.mean(confidences)), 1),
            "dominant_profile": Counter(profiles).most_common(1)[0][0],
            "dominant_readiness": Counter(readiness_vals).most_common(1)[0][0],
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get assessment history for a user.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of assessment result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None:
            return history[-last_n:]
        return list(history)

    def reset(self, user_id: str = "default"):
        """Clear history and baseline for a user.

        Args:
            user_id: User identifier.
        """
        self._history.pop(user_id, None)
        self._baselines.pop(user_id, None)

    # ── Private: signal processing ────────────────────────────

    def _ensure_2d(self, signals: np.ndarray) -> np.ndarray:
        """Ensure signals are 2D (n_channels, n_samples)."""
        signals = np.asarray(signals, dtype=float)
        # Replace NaNs with zeros to avoid Welch failures
        if np.any(np.isnan(signals)):
            signals = np.nan_to_num(signals, nan=0.0)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        return signals

    def _band_power(
        self, signal: np.ndarray, fs: float, low: float, high: float
    ) -> float:
        """Compute relative band power using Welch PSD."""
        n = len(signal)
        if n < 16:
            return 0.0
        nperseg = min(n, int(fs * 1.0))
        nperseg = max(nperseg, 16)
        try:
            freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.0
        total = _trapezoid(psd, freqs)
        if total <= 0:
            return 0.0
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return float(_trapezoid(psd[mask], freqs[mask]) / total)

    def _extract_metrics(self, signals: np.ndarray, fs: float) -> Dict[str, float]:
        """Extract all decision-relevant metrics from multichannel EEG.

        Returns a flat dict of metric name -> float value.
        """
        n_ch = signals.shape[0]

        # Determine frontal channels
        if n_ch >= 3:
            af7 = signals[_CH_AF7]
            af8 = signals[_CH_AF8]
        else:
            af7 = signals[0]
            af8 = signals[0]

        # Average frontal signal for band power extraction
        frontal_avg = (af7 + af8) / 2.0

        # Band powers from frontal average
        theta = self._band_power(frontal_avg, fs, *_BANDS["theta"])
        alpha = self._band_power(frontal_avg, fs, *_BANDS["alpha"])
        beta = self._band_power(frontal_avg, fs, *_BANDS["beta"])
        low_beta = self._band_power(frontal_avg, fs, *_BANDS["low_beta"])
        high_beta = self._band_power(frontal_avg, fs, *_BANDS["high_beta"])
        delta = self._band_power(frontal_avg, fs, *_BANDS["delta"])

        # Per-channel alpha for FAA
        af7_alpha = self._band_power(af7, fs, *_BANDS["alpha"])
        af8_alpha = self._band_power(af8, fs, *_BANDS["alpha"])

        # Theta/beta ratio
        theta_beta_ratio = theta / max(beta, 1e-10)

        # Alpha/beta ratio
        alpha_beta_ratio = alpha / max(beta, 1e-10)

        # FAA = log(right_alpha) - log(left_alpha)
        # Positive = more right alpha = more left activation = approach
        faa = float(
            np.log(max(af8_alpha, 1e-10)) - np.log(max(af7_alpha, 1e-10))
        )

        return {
            "theta": theta,
            "alpha": alpha,
            "beta": beta,
            "low_beta": low_beta,
            "high_beta": high_beta,
            "delta": delta,
            "af7_alpha": af7_alpha,
            "af8_alpha": af8_alpha,
            "theta_beta_ratio": theta_beta_ratio,
            "alpha_beta_ratio": alpha_beta_ratio,
            "faa": faa,
            "n_channels": float(n_ch),
        }

    def _normalize_metrics(
        self, metrics: Dict[str, float], baseline: Dict[str, float]
    ) -> Dict[str, float]:
        """Z-score normalize metrics against baseline.

        For ratio and FAA metrics, subtract baseline mean.
        For power metrics, divide by baseline to get relative change.
        """
        normalized = dict(metrics)
        ratio_keys = {"theta_beta_ratio", "alpha_beta_ratio", "faa"}
        power_keys = {"theta", "alpha", "beta", "low_beta", "high_beta", "delta",
                      "af7_alpha", "af8_alpha"}

        for key in ratio_keys:
            if key in baseline and key in normalized:
                normalized[key] = normalized[key] - baseline[key]

        for key in power_keys:
            if key in baseline and key in normalized and baseline[key] > 1e-10:
                normalized[key] = normalized[key] / baseline[key]

        return normalized

    # ── Private: composite scores ─────────────────────────────

    def _compute_deliberation(self, metrics: Dict[str, float]) -> float:
        """Deliberation intensity from frontal theta and theta/beta ratio.

        High frontal theta = active conflict monitoring / deliberation.
        """
        theta = metrics.get("theta", 0.0)
        tbr = metrics.get("theta_beta_ratio", 0.0)

        # Theta power component (normalized to typical range)
        # Relative theta in 0.1-0.5 range is typical for frontal channels
        theta_score = float(np.clip(theta / 0.35, 0.0, 1.0))

        # Theta/beta ratio component — higher = more deliberation
        tbr_score = float(np.clip(tbr / 2.0, 0.0, 1.0))

        deliberation = 0.55 * theta_score + 0.45 * tbr_score
        return float(np.clip(deliberation, 0.0, 1.0))

    def _compute_conflict(self, metrics: Dict[str, float]) -> float:
        """Cognitive conflict from frontal theta power.

        Midfrontal theta (4-8 Hz) is the primary marker for cognitive conflict
        and error monitoring (Cavanagh & Frank 2014).
        """
        theta = metrics.get("theta", 0.0)
        high_beta = metrics.get("high_beta", 0.0)

        # Theta power is the primary conflict signal
        theta_component = float(np.clip(theta / 0.35, 0.0, 1.0))

        # High beta (anxiety/stress) amplifies conflict signal
        hb_component = float(np.clip(high_beta / 0.15, 0.0, 1.0))

        conflict = 0.70 * theta_component + 0.30 * hb_component
        return float(np.clip(conflict, 0.0, 1.0))

    def _compute_approach_motivation(
        self, metrics: Dict[str, float], n_channels: int
    ) -> float:
        """Approach motivation from FAA.

        Positive FAA = more right alpha = more left activation = approach.
        Without multichannel data, returns near zero.
        """
        faa = metrics.get("faa", 0.0)

        if n_channels < 2:
            # Cannot compute meaningful FAA with single channel
            return 0.0

        # Clip and scale FAA to [-1, 1] with tanh
        return float(np.clip(np.tanh(faa * 1.5), -1.0, 1.0))

    def _compute_confidence(
        self,
        metrics: Dict[str, float],
        deliberation: float,
        conflict: float,
    ) -> float:
        """Decision confidence from beta, alpha suppression, and deliberation.

        High confidence:
          - High beta (resolved/decided state)
          - Low alpha (engaged, not idle)
          - Low deliberation (not actively weighing options)
          - Low conflict

        Returns 0-100 score.
        """
        beta = metrics.get("beta", 0.0)
        alpha = metrics.get("alpha", 0.0)

        # Beta engagement: higher beta = more confident/decided
        beta_score = float(np.clip(beta / 0.30, 0.0, 1.0))

        # Alpha suppression: low alpha = engaged (inverse)
        alpha_suppression = float(np.clip(1.0 - alpha / 0.40, 0.0, 1.0))

        # Low deliberation = already decided
        settled_score = 1.0 - deliberation

        # Low conflict = clear decision
        clarity_score = 1.0 - conflict

        confidence = (
            0.30 * beta_score
            + 0.25 * alpha_suppression
            + 0.25 * settled_score
            + 0.20 * clarity_score
        )
        return float(np.clip(confidence * 100.0, 0.0, 100.0))

    def _classify_risk(
        self, approach: float, metrics: Dict[str, float]
    ) -> str:
        """Classify risk profile from approach motivation and band powers.

        risk_seeking: high approach (FAA > 0.3) + high beta + low theta
        risk_averse: negative approach (FAA < -0.3) + high theta + low beta
        risk_neutral: otherwise
        """
        theta = metrics.get("theta", 0.0)
        beta = metrics.get("beta", 0.0)

        # Approach-based component
        if approach > 0.3 and beta > theta:
            return "risk_seeking"
        elif approach < -0.3 and theta > beta * 0.5:
            return "risk_averse"
        else:
            return "risk_neutral"

    def _classify_readiness(
        self,
        confidence: float,
        conflict: float,
        metrics: Dict[str, float],
    ) -> str:
        """Classify decision readiness.

        ready: high confidence (>60) + low conflict (<0.4)
        deliberating: high theta + moderate conflict
        uncertain: high alpha + low beta
        disengaged: very high alpha + very low beta
        """
        alpha = metrics.get("alpha", 0.0)
        beta = metrics.get("beta", 0.0)
        theta = metrics.get("theta", 0.0)

        # Check disengaged first (very high alpha, very low beta)
        if alpha > 0.40 and beta < 0.10:
            return "disengaged"

        # Uncertain: high alpha + low beta (less extreme than disengaged)
        if alpha > 0.25 and beta < 0.15:
            return "uncertain"

        # Deliberating: high theta + moderate conflict
        if theta > 0.15 and conflict > 0.35:
            return "deliberating"

        # Ready: high confidence + low conflict
        if confidence > 60.0 and conflict < 0.4:
            return "ready"

        # Default: deliberating if theta is moderate, uncertain otherwise
        if theta > 0.12:
            return "deliberating"
        return "uncertain"
