"""Mood forecaster — predict future mood from longitudinal EEG trends.

Uses FAA (frontal alpha asymmetry) for valence and beta/alpha ratio for
arousal, tracked over multiple sessions. EWMA smoothing reduces noise in
EEG-derived mood signals. Linear regression on the smoothed series
extrapolates short-term forecasts.

Key science:
    - FAA trends over sessions predict upcoming mood states (Davidson, 1992)
    - Declining alpha/beta ratio trend predicts increasing stress
    - Rising theta/alpha ratio predicts fatigue / low mood
    - EWMA smooths noisy epoch-to-epoch variation in mood signals
    - Russell's circumplex: valence x arousal -> mood quadrant

References:
    Davidson (1992) — Anterior cerebral asymmetry and emotion
    Russell (1980) — A circumplex model of affect
    Coan & Allen (2004) — Frontal EEG asymmetry as a moderator
"""
from typing import Dict, List, Optional

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

from scipy import signal as scipy_signal


# Minimum records before forecast is meaningful
_MIN_RECORDS_FOR_FORECAST = 5

# Maximum history entries per user
_MAX_HISTORY = 500

# Band definitions (Hz)
_BANDS = {
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "theta": (4.0, 8.0),
}


def _compute_psd(signal_1d: np.ndarray, fs: float):
    """Compute Welch PSD for a 1D signal."""
    n = len(signal_1d)
    nperseg = min(n, int(fs * 2))
    if nperseg < 4:
        nperseg = n
    noverlap = nperseg // 2
    freqs, psd = scipy_signal.welch(signal_1d, fs=fs, nperseg=nperseg,
                                     noverlap=noverlap)
    return freqs, psd


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Compute power in a frequency band."""
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 1e-12
    return max(float(_trapezoid(psd[mask], freqs[mask])), 1e-12)


class MoodForecaster:
    """Track mood over time and forecast future mood from EEG trends.

    Uses 4-channel Muse 2 EEG (TP9, AF7, AF8, TP10) at 256 Hz.
    Computes valence from FAA and arousal from beta/alpha ratio,
    applies EWMA smoothing, and uses linear regression to forecast.
    """

    def __init__(self, fs: float = 256.0, ewma_alpha: float = 0.3):
        """Initialize mood forecaster.

        Args:
            fs: Default sampling rate in Hz.
            ewma_alpha: EWMA smoothing factor (0-1). Higher = less smoothing.
        """
        self._fs = fs
        self._ewma_alpha = ewma_alpha
        # Per-user history: user_id -> list of dicts
        self._history: Dict[str, List[Dict]] = {}
        # Per-user EWMA state: user_id -> {valence, arousal}
        self._ewma_state: Dict[str, Dict[str, float]] = {}

    def record(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        mood_label: Optional[str] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record an EEG epoch and compute current mood.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate override.
            mood_label: Optional ground-truth mood label for this epoch.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with recorded, current_valence, current_arousal,
            current_mood, n_records.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        valence = self._compute_valence(signals, fs)
        arousal = self._compute_arousal(signals, fs)

        # Apply EWMA smoothing
        valence, arousal = self._apply_ewma(valence, arousal, user_id)

        # Classify mood quadrant
        mood = self._classify_mood(valence, arousal)

        # Store in history
        entry = {
            "valence": float(valence),
            "arousal": float(arousal),
            "mood": mood,
        }
        if mood_label is not None:
            entry["mood_label"] = mood_label

        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(entry)

        # Cap history at _MAX_HISTORY
        if len(self._history[user_id]) > _MAX_HISTORY:
            self._history[user_id] = self._history[user_id][-_MAX_HISTORY:]

        return {
            "recorded": True,
            "current_valence": float(valence),
            "current_arousal": float(arousal),
            "current_mood": mood,
            "n_records": len(self._history[user_id]),
        }

    def forecast(
        self,
        user_id: str = "default",
        horizon: int = 3,
    ) -> Dict:
        """Forecast future mood from longitudinal trends.

        Uses EWMA-smoothed valence/arousal series and linear regression
        to extrapolate by `horizon` steps.

        Args:
            user_id: User identifier.
            horizon: Number of steps to forecast ahead.

        Returns:
            Dict with forecast_valence, forecast_arousal, forecast_mood,
            trend_valence, trend_arousal, confidence, n_records,
            sufficient_data.
        """
        history = self._history.get(user_id, [])
        n = len(history)

        if n < _MIN_RECORDS_FOR_FORECAST:
            return {
                "forecast_valence": 0.0,
                "forecast_arousal": 0.5,
                "forecast_mood": "neutral",
                "trend_valence": "stable",
                "trend_arousal": "stable",
                "confidence": 0.0,
                "n_records": n,
                "sufficient_data": False,
            }

        # Extract valence and arousal series
        valences = np.array([e["valence"] for e in history])
        arousals = np.array([e["arousal"] for e in history])

        # Use last 10 points (or all if fewer) for regression
        window = min(10, n)
        v_window = valences[-window:]
        a_window = arousals[-window:]

        # Linear regression: fit y = slope * x + intercept
        x = np.arange(window, dtype=float)
        v_slope, v_intercept = self._linear_fit(x, v_window)
        a_slope, a_intercept = self._linear_fit(x, a_window)

        # Extrapolate by horizon steps
        forecast_x = window - 1 + horizon
        forecast_valence = float(np.clip(v_slope * forecast_x + v_intercept, -1.0, 1.0))
        forecast_arousal = float(np.clip(a_slope * forecast_x + a_intercept, 0.0, 1.0))

        forecast_mood = self._classify_mood(forecast_valence, forecast_arousal)

        # Determine trend labels
        trend_valence = self._valence_trend(v_slope)
        trend_arousal = self._arousal_trend(a_slope)

        # Confidence: increases with more data, up to 1.0
        confidence = float(np.clip(n / 50.0, 0.0, 1.0))

        return {
            "forecast_valence": forecast_valence,
            "forecast_arousal": forecast_arousal,
            "forecast_mood": forecast_mood,
            "trend_valence": trend_valence,
            "trend_arousal": trend_arousal,
            "confidence": confidence,
            "n_records": n,
            "sufficient_data": True,
        }

    def get_mood_timeline(
        self,
        user_id: str = "default",
        last_n: Optional[int] = None,
    ) -> List[Dict]:
        """Get mood history for a user.

        Args:
            user_id: User identifier.
            last_n: If set, return only the last N entries.

        Returns:
            List of {valence, arousal, mood} dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None:
            return list(history[-last_n:])
        return list(history)

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get aggregate stats for a user's session.

        Args:
            user_id: User identifier.

        Returns:
            Dict with n_records, mean_valence, mean_arousal, dominant_mood.
        """
        history = self._history.get(user_id, [])
        n = len(history)
        if n == 0:
            return {
                "n_records": 0,
                "mean_valence": 0.0,
                "mean_arousal": 0.5,
                "dominant_mood": "neutral",
            }

        valences = [e["valence"] for e in history]
        arousals = [e["arousal"] for e in history]
        moods = [e["mood"] for e in history]

        # Find dominant mood by frequency
        mood_counts: Dict[str, int] = {}
        for m in moods:
            mood_counts[m] = mood_counts.get(m, 0) + 1
        dominant_mood = max(mood_counts, key=mood_counts.get)

        return {
            "n_records": n,
            "mean_valence": float(np.mean(valences)),
            "mean_arousal": float(np.mean(arousals)),
            "dominant_mood": dominant_mood,
        }

    def reset(self, user_id: str = "default") -> None:
        """Clear all history and EWMA state for a user.

        Args:
            user_id: User identifier.
        """
        self._history.pop(user_id, None)
        self._ewma_state.pop(user_id, None)

    # ── Private methods ─────────────────────────────────────────

    def _compute_valence(self, signals: np.ndarray, fs: float) -> float:
        """Compute valence from FAA + alpha/beta ratio.

        For multichannel: 50% FAA + 50% alpha/beta ratio.
        For single-channel: 100% alpha/beta ratio.
        """
        n_channels = signals.shape[0]

        # Alpha/beta ratio from first channel (or average across channels)
        abr_values = []
        for ch in range(n_channels):
            freqs, psd = _compute_psd(signals[ch], fs)
            alpha = _band_power(freqs, psd, *_BANDS["alpha"])
            beta = _band_power(freqs, psd, *_BANDS["beta"])
            ratio = alpha / beta
            abr_values.append(ratio)

        mean_abr = float(np.mean(abr_values))
        valence_abr = float(np.tanh((mean_abr - 0.7) * 2.0))

        # FAA if multichannel (need at least AF7=ch1 and AF8=ch2)
        if n_channels >= 3:
            # AF7 = ch1, AF8 = ch2 for Muse 2
            freqs_l, psd_l = _compute_psd(signals[1], fs)
            freqs_r, psd_r = _compute_psd(signals[2], fs)
            l_alpha = _band_power(freqs_l, psd_l, *_BANDS["alpha"])
            r_alpha = _band_power(freqs_r, psd_r, *_BANDS["alpha"])
            faa = float(np.log(r_alpha) - np.log(l_alpha))
            faa_valence = float(np.clip(np.tanh(faa * 2.0), -1.0, 1.0))
            valence = float(np.clip(0.50 * valence_abr + 0.50 * faa_valence, -1.0, 1.0))
        else:
            valence = float(np.clip(valence_abr, -1.0, 1.0))

        return valence

    def _compute_arousal(self, signals: np.ndarray, fs: float) -> float:
        """Compute arousal from beta/(alpha+beta) ratio, averaged across channels."""
        arousal_values = []
        for ch in range(signals.shape[0]):
            freqs, psd = _compute_psd(signals[ch], fs)
            alpha = _band_power(freqs, psd, *_BANDS["alpha"])
            beta = _band_power(freqs, psd, *_BANDS["beta"])
            ratio = beta / (alpha + beta)
            arousal_values.append(ratio)

        arousal = float(np.clip(np.mean(arousal_values), 0.0, 1.0))
        return arousal

    def _apply_ewma(
        self, valence: float, arousal: float, user_id: str
    ) -> tuple:
        """Apply exponentially weighted moving average smoothing.

        Returns:
            Tuple of (smoothed_valence, smoothed_arousal).
        """
        alpha = self._ewma_alpha
        state = self._ewma_state.get(user_id)

        if state is None:
            # First observation: initialize EWMA to current value
            smoothed_v = valence
            smoothed_a = arousal
        else:
            smoothed_v = alpha * valence + (1 - alpha) * state["valence"]
            smoothed_a = alpha * arousal + (1 - alpha) * state["arousal"]

        self._ewma_state[user_id] = {
            "valence": smoothed_v,
            "arousal": smoothed_a,
        }
        return smoothed_v, smoothed_a

    @staticmethod
    def _classify_mood(valence: float, arousal: float) -> str:
        """Classify mood quadrant from Russell's circumplex model.

        Returns:
            One of: positive_high, positive_low, negative_high,
            negative_low, neutral.
        """
        if abs(valence) < 0.1:
            return "neutral"
        if valence > 0:
            return "positive_high" if arousal > 0.5 else "positive_low"
        return "negative_high" if arousal > 0.5 else "negative_low"

    @staticmethod
    def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple:
        """Simple least-squares linear fit.

        Returns:
            Tuple of (slope, intercept).
        """
        n = len(x)
        if n < 2:
            return 0.0, float(y[0]) if n == 1 else 0.0
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx == 0:
            return 0.0, float(y_mean)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        slope = float(ss_xy / ss_xx)
        intercept = float(y_mean - slope * x_mean)
        return slope, intercept

    @staticmethod
    def _valence_trend(slope: float) -> str:
        """Classify valence trend from regression slope."""
        if slope > 0.01:
            return "improving"
        if slope < -0.01:
            return "declining"
        return "stable"

    @staticmethod
    def _arousal_trend(slope: float) -> str:
        """Classify arousal trend from regression slope."""
        if slope > 0.005:
            return "increasing"
        if slope < -0.005:
            return "decreasing"
        return "stable"
