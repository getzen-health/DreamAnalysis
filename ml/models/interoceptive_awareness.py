"""Interoceptive awareness estimator from EEG markers.

Estimates interoceptive awareness (the ability to perceive internal
bodily signals) from EEG-only proxy metrics. The gold standard is
heartbeat-evoked potential (HEP) amplitude measured with concurrent
PPG/ECG, but since Muse 2 PPG data may not always be available,
this module uses three EEG-only proxies:

1. **Frontal theta power** (AF7/AF8, 4-8 Hz): Elevated during
   interoceptive tasks — reflects anterior insula / ACC engagement
   (Critchley et al., 2004; Pollatos et al., 2005).

2. **Alpha suppression**: Reduction in alpha power (8-12 Hz) from
   resting baseline during body-focused attention. Alpha desynchronizes
   when neural circuits actively process sensory input (Klimesch, 2012).

3. **Right-frontal activation**: Lower alpha at AF8 vs AF7 indicates
   right anterior insula engagement — the primary cortical hub for
   interoception (Craig, 2009; Critchley & Garfinkel, 2017).

Channel layout (Muse 2): ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10.

References:
    Craig (2009) — How do you feel now? The anterior insula and human awareness
    Critchley et al. (2004) — Neural systems supporting interoceptive awareness
    Pollatos et al. (2005) — Frontal cortex role in interoception
    Garfinkel et al. (2015) — Knowing your own heart: interoceptive accuracy
    Klimesch (2012) — Alpha-band oscillations, attention, and controlled access
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal

# numpy compat: trapezoid (numpy >= 2.0) or trapz (older)
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# Muse 2 channel indices
_CH_AF7 = 1
_CH_AF8 = 2

# Population-average baseline alpha power (used when no baseline recorded).
# Typical resting-state alpha at frontal sites is ~5-15 uV^2/Hz integrated
# over 8-12 Hz.  We pick a moderate default so scores are not wildly off.
_DEFAULT_BASELINE_ALPHA = 8.0  # uV^2 (approximate)

# History cap per user to avoid unbounded memory growth.
_MAX_HISTORY = 500

# Body-awareness level thresholds
_LEVEL_HIGH = 0.65
_LEVEL_MODERATE = 0.40
_LEVEL_LOW = 0.20


def _band_power(sig: np.ndarray, fs: float, low: float, high: float) -> float:
    """Compute absolute band power via Welch PSD for a single channel.

    Returns integrated PSD (uV^2) in the [low, high] Hz band.
    """
    nperseg = min(len(sig), int(fs * 2))
    if nperseg < 4:
        return 0.0
    try:
        freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=nperseg)
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return float(_trapezoid(psd[mask], freqs[mask]))
    except Exception:
        return 0.0


class InteroceptiveAwarenessTrainer:
    """Estimate interoceptive awareness from EEG markers.

    Maintains per-user baselines and assessment history.  Designed to be
    instantiated once and shared across requests (keyed by user_id).
    """

    def __init__(self, fs: float = 256.0) -> None:
        self._fs = fs
        # Per-user storage
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._history: Dict[str, List[Dict]] = {}

    # ── Public API ──────────────────────────────────────────────

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline for interoceptive metrics.

        Should be called during eyes-closed resting state (2 min minimum
        recommended) before any interoceptive task.

        Args:
            signals: (n_channels, n_samples) EEG array.
            fs: Sampling rate (defaults to self._fs).
            user_id: User identifier.

        Returns:
            Dict with baseline_set (bool) and n_channels (int).
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_ch = signals.shape[0]

        # Compute per-channel alpha and theta for baseline
        alpha_powers = []
        theta_powers = []
        for ch in range(n_ch):
            alpha_powers.append(_band_power(signals[ch], fs, 8.0, 12.0))
            theta_powers.append(_band_power(signals[ch], fs, 4.0, 8.0))

        # Store frontal alpha separately (for suppression calculation)
        frontal_alpha = []
        if n_ch > _CH_AF7:
            frontal_alpha.append(alpha_powers[_CH_AF7])
        if n_ch > _CH_AF8:
            frontal_alpha.append(alpha_powers[_CH_AF8])
        if not frontal_alpha:
            frontal_alpha.append(alpha_powers[0])

        self._baselines[user_id] = {
            "mean_alpha": float(np.mean(alpha_powers)),
            "mean_theta": float(np.mean(theta_powers)),
            "frontal_alpha": float(np.mean(frontal_alpha)),
            "af7_alpha": alpha_powers[_CH_AF7] if n_ch > _CH_AF7 else alpha_powers[0],
            "af8_alpha": alpha_powers[_CH_AF8] if n_ch > _CH_AF8 else alpha_powers[0],
        }

        return {
            "baseline_set": True,
            "n_channels": n_ch,
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess interoceptive awareness from a live EEG epoch.

        Best called during a heartbeat-focused attention task (e.g.,
        "count your heartbeats without touching your pulse").

        Args:
            signals: (n_channels, n_samples) EEG (4-sec minimum recommended).
            fs: Sampling rate (defaults to self._fs).
            user_id: User identifier.

        Returns:
            Dict with interoceptive_score (0-1), frontal_theta_power,
            alpha_suppression, right_frontal_activation,
            body_awareness_level, and has_baseline.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_ch = signals.shape[0]
        baseline = self._baselines.get(user_id)
        has_baseline = baseline is not None

        # ---- Metric 1: Frontal theta power ----
        # Higher theta at frontal sites = more interoceptive processing
        frontal_theta = self._compute_frontal_theta(signals, fs, n_ch)

        # Normalize theta to 0-1 using sigmoid-like mapping.
        # Typical frontal theta power during interoceptive task: 2-20 uV^2.
        # midpoint ~6, steepness ~0.4
        theta_norm = float(1.0 / (1.0 + np.exp(-0.4 * (frontal_theta - 6.0))))

        # ---- Metric 2: Alpha suppression from baseline ----
        alpha_suppression = self._compute_alpha_suppression(
            signals, fs, n_ch, baseline
        )

        # ---- Metric 3: Right-frontal activation (insula proxy) ----
        right_frontal = self._compute_right_frontal_activation(
            signals, fs, n_ch
        )

        # ---- Composite score ----
        # Weights: theta 35%, alpha suppression 35%, right frontal 30%
        composite = (
            0.35 * theta_norm
            + 0.35 * alpha_suppression
            + 0.30 * right_frontal
        )
        interoceptive_score = float(np.clip(composite, 0.0, 1.0))

        # Body awareness level
        if interoceptive_score >= _LEVEL_HIGH:
            level = "high"
        elif interoceptive_score >= _LEVEL_MODERATE:
            level = "moderate"
        elif interoceptive_score >= _LEVEL_LOW:
            level = "low"
        else:
            level = "minimal"

        result = {
            "interoceptive_score": round(interoceptive_score, 4),
            "frontal_theta_power": round(frontal_theta, 4),
            "alpha_suppression": round(alpha_suppression, 4),
            "right_frontal_activation": round(right_frontal, 4),
            "body_awareness_level": level,
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
        """Get session statistics for a user.

        Returns:
            Dict with n_epochs, mean_score, and improvement_trend.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {
                "n_epochs": 0,
                "mean_score": 0.0,
                "improvement_trend": "insufficient_data",
            }

        scores = [h["interoceptive_score"] for h in history]
        mean_score = float(np.mean(scores))

        # Improvement trend: compare first half vs second half
        if len(scores) < 4:
            trend = "insufficient_data"
        else:
            mid = len(scores) // 2
            first_half = float(np.mean(scores[:mid]))
            second_half = float(np.mean(scores[mid:]))
            diff = second_half - first_half
            if diff > 0.05:
                trend = "improving"
            elif diff < -0.05:
                trend = "declining"
            else:
                trend = "stable"

        return {
            "n_epochs": len(scores),
            "mean_score": round(mean_score, 4),
            "improvement_trend": trend,
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
        if last_n is not None and last_n > 0:
            return history[-last_n:]
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all data for a user (baseline + history)."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # ── Private helpers ─────────────────────────────────────────

    def _compute_frontal_theta(
        self, signals: np.ndarray, fs: float, n_ch: int
    ) -> float:
        """Average theta power at frontal channels (AF7, AF8)."""
        theta_powers = []
        if n_ch > _CH_AF7:
            theta_powers.append(_band_power(signals[_CH_AF7], fs, 4.0, 8.0))
        if n_ch > _CH_AF8:
            theta_powers.append(_band_power(signals[_CH_AF8], fs, 4.0, 8.0))
        if not theta_powers:
            # Fallback: use channel 0
            theta_powers.append(_band_power(signals[0], fs, 4.0, 8.0))
        return float(np.mean(theta_powers))

    def _compute_alpha_suppression(
        self,
        signals: np.ndarray,
        fs: float,
        n_ch: int,
        baseline: Optional[Dict[str, float]],
    ) -> float:
        """Alpha suppression from baseline (0-1, higher = more suppressed).

        Alpha desynchronization (power decrease) during body-focused
        attention indicates active interoceptive processing.
        """
        # Current frontal alpha
        alpha_powers = []
        if n_ch > _CH_AF7:
            alpha_powers.append(_band_power(signals[_CH_AF7], fs, 8.0, 12.0))
        if n_ch > _CH_AF8:
            alpha_powers.append(_band_power(signals[_CH_AF8], fs, 8.0, 12.0))
        if not alpha_powers:
            alpha_powers.append(_band_power(signals[0], fs, 8.0, 12.0))
        current_alpha = float(np.mean(alpha_powers))

        # Baseline alpha (or population default)
        if baseline is not None:
            baseline_alpha = baseline["frontal_alpha"]
        else:
            baseline_alpha = _DEFAULT_BASELINE_ALPHA

        # Suppress ratio: how much alpha dropped from baseline.
        # suppression = (baseline - current) / baseline, clipped to [0, 1].
        # A value of 0 = no suppression (alpha same or higher than baseline).
        # A value near 1 = strong suppression (alpha nearly gone).
        if baseline_alpha < 1e-10:
            return 0.0

        suppression = (baseline_alpha - current_alpha) / (baseline_alpha + 1e-10)
        return float(np.clip(suppression, 0.0, 1.0))

    def _compute_right_frontal_activation(
        self, signals: np.ndarray, fs: float, n_ch: int
    ) -> float:
        """Right-frontal activation index (0-1).

        Lower alpha at AF8 relative to AF7 indicates greater right-frontal
        cortical activation, which is a proxy for anterior insula engagement
        during interoceptive processing (Craig, 2009).

        Uses: (AF7_alpha - AF8_alpha) / (AF7_alpha + AF8_alpha).
        Positive = right more active. Normalized to [0, 1].
        """
        if n_ch <= _CH_AF8:
            # Not enough channels for laterality — return neutral 0.5
            return 0.5

        af7_alpha = _band_power(signals[_CH_AF7], fs, 8.0, 12.0)
        af8_alpha = _band_power(signals[_CH_AF8], fs, 8.0, 12.0)

        denom = af7_alpha + af8_alpha
        if denom < 1e-10:
            return 0.5

        # (AF7 - AF8) / (AF7 + AF8): positive when AF8 alpha is lower
        # (= right frontal more active). Map from [-1, 1] to [0, 1].
        asymmetry = (af7_alpha - af8_alpha) / denom
        return float(np.clip(0.5 + asymmetry * 0.5, 0.0, 1.0))
