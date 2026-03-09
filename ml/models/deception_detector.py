"""EEG-based deception/lie detection for 4-channel Muse 2.

Detects deceptive cognitive states via continuous EEG monitoring using
validated biomarkers from the deception neuroscience literature:

1. Frontal theta increase (AF7/AF8) -- cognitive load of lying
2. Beta power increase -- effortful processing during deception
3. Alpha suppression -- heightened engagement/vigilance
4. Theta/alpha ratio elevation -- cognitive load marker
5. Reduced frontal alpha asymmetry stability during deception

Muse 2 channel mapping (BrainFlow board_id 38):
    ch0 = TP9  (left temporal)
    ch1 = AF7  (left frontal)
    ch2 = AF8  (right frontal)
    ch3 = TP10 (right temporal)

DISCLAIMER: This is an experimental research tool only. EEG-based deception
detection is NOT admissible as legal evidence and should never be used for
forensic, employment, or any high-stakes decision-making.

References:
    Rosenfeld et al. (2008) -- P300-based concealed information test review
    Ganis et al. (2003) -- fMRI deception, frontal engagement
    Farwell & Donchin (1991) -- original P300 brain fingerprinting
    Hu et al. (2012) -- EEG theta/beta during deception
"""

from typing import Dict, List, Optional

import numpy as np

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz

from scipy.signal import welch

DISCLAIMER = (
    "Experimental research tool only. Not admissible as evidence. "
    "EEG-based deception detection has fundamental limitations and "
    "must never be used for legal, forensic, or employment decisions."
)

# Assessment thresholds
_TRUTHFUL_UPPER = 0.35
_DECEPTIVE_LOWER = 0.65


class DeceptionDetector:
    """EEG-based deception detection for 4-channel Muse 2 (256 Hz).

    Uses frontal theta, beta engagement, and alpha suppression relative
    to a truth-telling baseline to estimate deception likelihood.

    Multi-user support: each user_id tracks independent baseline,
    history, and session statistics.
    """

    def __init__(self, fs: float = 256.0):
        """Initialize the deception detector.

        Args:
            fs: Default sampling rate in Hz (Muse 2 = 256).
        """
        self._fs = fs
        # Per-user state: baseline metrics + assessment history
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    # ── Public API ────────────────────────────────────────────

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record a truth-telling baseline for a user.

        Should be collected during a known-truthful period (e.g., answering
        verifiable questions honestly). The baseline captures normal frontal
        theta, beta, and alpha levels for this user.

        Args:
            signals: EEG data -- shape (4, n_samples) for multichannel or
                     (n_samples,) for single-channel.
            fs: Sampling rate. Falls back to instance default.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set (bool) and baseline_metrics.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)

        frontal_theta = self._compute_frontal_theta(signals, fs)
        beta_power = self._compute_beta_power(signals, fs)
        alpha_power = self._compute_alpha_power(signals, fs)
        theta_alpha_ratio = frontal_theta / (alpha_power + 1e-12)

        metrics = {
            "frontal_theta": float(frontal_theta),
            "beta_power": float(beta_power),
            "alpha_power": float(alpha_power),
            "theta_alpha_ratio": float(theta_alpha_ratio),
        }

        self._baselines[user_id] = metrics

        return {
            "baseline_set": True,
            "baseline_metrics": metrics,
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess current EEG for deception likelihood.

        Computes cognitive load (frontal theta), beta engagement, and
        alpha suppression. When a baseline exists, values are normalized
        relative to the user's truthful baseline.

        Args:
            signals: EEG data -- (4, n_samples) or (n_samples,).
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with deception_likelihood, cognitive_load, confidence,
            assessment label, component scores, and disclaimer.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)

        has_baseline = user_id in self._baselines
        baseline = self._baselines.get(user_id)

        # Extract current features
        frontal_theta = self._compute_frontal_theta(signals, fs)
        beta_power = self._compute_beta_power(signals, fs)
        alpha_power = self._compute_alpha_power(signals, fs)

        # Compute component scores (0-1)
        cognitive_load = self._compute_cognitive_load(
            frontal_theta, baseline
        )
        beta_engagement = self._compute_beta_engagement(
            beta_power, baseline
        )
        alpha_suppression = self._compute_alpha_suppression(
            alpha_power, baseline
        )

        # Weighted combination: cognitive_load 40%, beta 30%, alpha 30%
        deception_likelihood = float(np.clip(
            0.40 * cognitive_load
            + 0.30 * beta_engagement
            + 0.30 * alpha_suppression,
            0.0,
            1.0,
        ))

        # Confidence: higher when baseline exists and features are
        # far from the uncertain zone
        confidence = self._compute_confidence(
            deception_likelihood, has_baseline
        )

        # Assessment label
        if deception_likelihood < _TRUTHFUL_UPPER:
            assessment = "truthful"
        elif deception_likelihood > _DECEPTIVE_LOWER:
            assessment = "deceptive"
        else:
            assessment = "uncertain"

        result = {
            "deception_likelihood": round(deception_likelihood, 4),
            "cognitive_load": round(cognitive_load, 4),
            "confidence": round(confidence, 4),
            "assessment": assessment,
            "frontal_theta_power": round(float(frontal_theta), 6),
            "beta_engagement": round(beta_engagement, 4),
            "alpha_suppression": round(alpha_suppression, 4),
            "disclaimer": DISCLAIMER,
            "has_baseline": has_baseline,
        }

        # Append to history (capped at 500 per user)
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > 500:
            self._history[user_id] = self._history[user_id][-500:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get aggregate session statistics for a user.

        Returns:
            Dict with n_epochs, mean_likelihood, and
            assessment_distribution counts.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {
                "n_epochs": 0,
                "mean_likelihood": 0.0,
                "assessment_distribution": {
                    "truthful": 0,
                    "uncertain": 0,
                    "deceptive": 0,
                },
            }

        likelihoods = [h["deception_likelihood"] for h in history]
        dist = {"truthful": 0, "uncertain": 0, "deceptive": 0}
        for h in history:
            label = h["assessment"]
            if label in dist:
                dist[label] += 1

        return {
            "n_epochs": len(history),
            "mean_likelihood": round(float(np.mean(likelihoods)), 4),
            "assessment_distribution": dist,
        }

    def get_history(
        self,
        user_id: str = "default",
        last_n: Optional[int] = None,
    ) -> List[Dict]:
        """Return assessment history for a user.

        Args:
            user_id: User identifier.
            last_n: If set, return only the last N entries.

        Returns:
            List of assessment result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None and last_n > 0:
            return history[-last_n:]
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear baseline and history for a user.

        Args:
            user_id: User identifier.
        """
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # ── Private feature extraction ────────────────────────────

    def _get_frontal_channels(self, signals: np.ndarray) -> np.ndarray:
        """Extract frontal channels (AF7, AF8) from multichannel data.

        Returns (2, n_samples) for multichannel, or (1, n_samples) for 1D.
        """
        if signals.ndim == 2 and signals.shape[0] >= 3:
            # ch1=AF7, ch2=AF8
            return signals[1:3]
        if signals.ndim == 2:
            return signals[:1]
        return signals.reshape(1, -1)

    def _compute_band_power(
        self,
        signal_1d: np.ndarray,
        fs: float,
        low: float,
        high: float,
    ) -> float:
        """Compute absolute band power via Welch PSD.

        Args:
            signal_1d: 1D signal array.
            fs: Sampling rate.
            low: Lower frequency bound (Hz).
            high: Upper frequency bound (Hz).

        Returns:
            Absolute power in the specified band.
        """
        n = len(signal_1d)
        if n < 4:
            return 0.0

        nperseg = min(n, int(fs))
        if nperseg < 4:
            nperseg = n

        try:
            freqs, psd = welch(signal_1d, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.0

        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0

        return float(_trapezoid(psd[mask], freqs[mask]))

    def _compute_frontal_theta(
        self, signals: np.ndarray, fs: float
    ) -> float:
        """Average frontal theta power (4-8 Hz) from AF7/AF8."""
        frontal = self._get_frontal_channels(signals)
        powers = []
        for ch in frontal:
            powers.append(self._compute_band_power(ch, fs, 4.0, 8.0))
        return float(np.mean(powers)) if powers else 0.0

    def _compute_beta_power(
        self, signals: np.ndarray, fs: float
    ) -> float:
        """Average beta power (12-30 Hz) from AF7/AF8."""
        frontal = self._get_frontal_channels(signals)
        powers = []
        for ch in frontal:
            powers.append(self._compute_band_power(ch, fs, 12.0, 30.0))
        return float(np.mean(powers)) if powers else 0.0

    def _compute_alpha_power(
        self, signals: np.ndarray, fs: float
    ) -> float:
        """Average alpha power (8-12 Hz) from AF7/AF8."""
        frontal = self._get_frontal_channels(signals)
        powers = []
        for ch in frontal:
            powers.append(self._compute_band_power(ch, fs, 8.0, 12.0))
        return float(np.mean(powers)) if powers else 0.0

    # ── Component scores (0-1) ────────────────────────────────

    def _compute_cognitive_load(
        self, frontal_theta: float, baseline: Optional[Dict]
    ) -> float:
        """Cognitive load from frontal theta elevation.

        With baseline: sigmoid of the relative theta increase.
        Without baseline: sigmoid of absolute theta (population average).
        """
        if baseline is not None:
            base_theta = baseline["frontal_theta"]
            if base_theta > 1e-12:
                ratio = frontal_theta / base_theta
                # ratio > 1 = theta elevated above baseline
                return float(np.clip(
                    1.0 / (1.0 + np.exp(-3.0 * (ratio - 1.2))),
                    0.0,
                    1.0,
                ))
        # No baseline: use absolute theta with population-average sigmoid
        return float(np.clip(
            1.0 / (1.0 + np.exp(-2.0 * (frontal_theta - 0.5))),
            0.0,
            1.0,
        ))

    def _compute_beta_engagement(
        self, beta_power: float, baseline: Optional[Dict]
    ) -> float:
        """Beta engagement from beta power increase.

        Deception increases effortful processing, elevating beta.
        """
        if baseline is not None:
            base_beta = baseline["beta_power"]
            if base_beta > 1e-12:
                ratio = beta_power / base_beta
                return float(np.clip(
                    1.0 / (1.0 + np.exp(-3.0 * (ratio - 1.2))),
                    0.0,
                    1.0,
                ))
        return float(np.clip(
            1.0 / (1.0 + np.exp(-2.0 * (beta_power - 0.3))),
            0.0,
            1.0,
        ))

    def _compute_alpha_suppression(
        self, alpha_power: float, baseline: Optional[Dict]
    ) -> float:
        """Alpha suppression during deception.

        Alpha decreases when engagement and vigilance increase.
        Score is inverted: lower alpha = higher suppression score.
        """
        if baseline is not None:
            base_alpha = baseline["alpha_power"]
            if base_alpha > 1e-12:
                ratio = alpha_power / base_alpha
                # ratio < 1 means alpha is suppressed
                suppression = 1.0 - ratio
                return float(np.clip(
                    1.0 / (1.0 + np.exp(-4.0 * suppression)),
                    0.0,
                    1.0,
                ))
        # No baseline: lower absolute alpha = more suppression
        exponent = 4.0 * (alpha_power - 0.3)
        # Clamp exponent to avoid overflow in exp()
        exponent = float(np.clip(exponent, -50.0, 50.0))
        return float(np.clip(
            1.0 / (1.0 + np.exp(exponent)),
            0.0,
            1.0,
        ))

    def _compute_confidence(
        self, likelihood: float, has_baseline: bool
    ) -> float:
        """Confidence score based on distance from uncertainty zone and baseline.

        Higher confidence when:
        - Assessment is far from the 0.35-0.65 uncertain band
        - A baseline is available
        """
        # Distance from center of uncertain zone (0.50)
        center = 0.50
        dist_from_center = abs(likelihood - center)
        # Max distance is 0.50 (at 0.0 or 1.0)
        distance_score = dist_from_center / 0.50  # 0-1

        # Baseline bonus
        baseline_factor = 0.8 if has_baseline else 0.5

        confidence = float(np.clip(
            distance_score * baseline_factor + (0.2 if has_baseline else 0.1),
            0.1,
            0.95,
        ))
        return confidence
