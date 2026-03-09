"""Closed-loop emotion regulation biofeedback via frontal alpha asymmetry.

Trains users to shift FAA (frontal alpha asymmetry) toward positive valence
by increasing relative left-frontal activation. The Muse 2 headband provides
AF7 (ch1, left frontal) and AF8 (ch2, right frontal) for FAA computation.

Protocol:
1. Baseline: record 2 min resting FAA from AF7/AF8
2. Training: real-time feedback on FAA deviation from baseline
3. Reward when FAA shifts toward target state (positive or neutral)
4. Cognitive reappraisal cues guide the user based on current state

References:
    Ochsner & Gross (2005) - Cognitive control of emotion
    Davidson et al. (2003) - Alterations in brain and immune function
        produced by mindfulness meditation
    Harmon-Jones (2003) - Clarifying the emotive functions of asymmetrical
        frontal cortical activity
"""
import threading
from typing import Dict, List, Optional

import numpy as np

try:
    from scipy import signal as scipy_signal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# Cognitive reappraisal strategies keyed by current emotional state
_STRATEGIES = {
    "negative_high_arousal": [
        "Try slow diaphragmatic breathing: inhale 4 sec, hold 4, exhale 6.",
        "Reframe the situation: what would you advise a friend feeling this?",
        "Notice physical tension and consciously release your jaw and shoulders.",
    ],
    "negative_low_arousal": [
        "Focus on a specific positive memory and hold it in vivid detail.",
        "Practice gratitude reflection: name three things you appreciate right now.",
        "Engage body scan relaxation: move attention slowly from feet to head.",
    ],
    "neutral": [
        "Maintain gentle awareness of your breath without changing it.",
        "Visualize a calm place you have been -- notice colors, sounds, temperature.",
        "Softly smile and notice any shift in how you feel.",
    ],
    "positive": [
        "You are doing well. Stay with this feeling and deepen it.",
        "Notice what thoughts or images accompany this positive state.",
        "Savor this moment -- let the feeling expand through your body.",
    ],
}

# Feedback messages by regulation performance
_FEEDBACK = {
    "strong_success": "Excellent regulation -- FAA shifted strongly toward target.",
    "moderate_success": "Good progress -- FAA is moving in the right direction.",
    "mild_success": "Slight positive shift detected. Keep focusing.",
    "no_change": "FAA is near baseline. Try a different strategy.",
    "mild_failure": "FAA drifted slightly away from target. Refocus gently.",
    "strong_failure": "FAA moved away from target. Pause, breathe, and try again.",
}


class EmotionRegulationTrainer:
    """Closed-loop FAA neurofeedback for emotion regulation training.

    Target: train users to shift FAA toward positive valence
    (left-frontal activation > right-frontal activation).

    FAA = log(AF8_alpha) - log(AF7_alpha)
    Positive FAA -> approach motivation / positive affect.
    """

    def __init__(
        self,
        success_threshold: float = 0.05,
        fs: float = 256.0,
    ):
        """Initialize the emotion regulation trainer.

        Args:
            success_threshold: Minimum FAA shift from baseline to count as
                regulation success. Default 0.05 log-units (small but
                meaningful per Davidson 2003).
            fs: Default sampling rate in Hz.
        """
        self._success_threshold = success_threshold
        self._fs = fs
        # Per-user state: baseline FAA and session history
        self._baselines: Dict[str, float] = {}
        self._histories: Dict[str, List[Dict]] = {}

    # ── Public API ──────────────────────────────────────────────

    def set_baseline(
        self,
        eeg_signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record baseline resting-state FAA.

        Args:
            eeg_signals: (n_channels, n_samples) EEG array. Needs at least
                2 channels where ch1=AF7 and ch2=AF8.
            fs: Sampling rate (defaults to self._fs).
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_faa, af7_alpha, af8_alpha, baseline_set.
        """
        fs = fs or self._fs
        eeg_signals = np.asarray(eeg_signals, dtype=float)
        if eeg_signals.ndim == 1:
            eeg_signals = eeg_signals.reshape(1, -1)

        af7_alpha, af8_alpha = self._frontal_alpha_powers(eeg_signals, fs)

        # FAA = log(right) - log(left)
        eps = 1e-12
        faa = float(np.log(af8_alpha + eps) - np.log(af7_alpha + eps))

        self._baselines[user_id] = faa

        return {
            "baseline_faa": round(faa, 6),
            "af7_alpha": round(af7_alpha, 6),
            "af8_alpha": round(af8_alpha, 6),
            "baseline_set": True,
        }

    def evaluate(
        self,
        eeg_signals: np.ndarray,
        fs: Optional[float] = None,
        target_state: str = "positive",
        user_id: str = "default",
    ) -> Dict:
        """Evaluate a training epoch and provide regulation feedback.

        Args:
            eeg_signals: (n_channels, n_samples) EEG epoch.
            fs: Sampling rate.
            target_state: 'positive' (shift FAA up) or 'neutral' (return
                FAA to baseline).
            user_id: User identifier.

        Returns:
            Dict with current_faa, baseline_faa, regulation_success,
            regulation_score, effort_index, feedback_message,
            target_state, strategy_suggestion.
        """
        fs = fs or self._fs
        eeg_signals = np.asarray(eeg_signals, dtype=float)
        if eeg_signals.ndim == 1:
            eeg_signals = eeg_signals.reshape(1, -1)

        baseline_faa = self._baselines.get(user_id, 0.0)
        af7_alpha, af8_alpha = self._frontal_alpha_powers(eeg_signals, fs)

        eps = 1e-12
        current_faa = float(np.log(af8_alpha + eps) - np.log(af7_alpha + eps))

        # FAA deviation from baseline
        faa_shift = current_faa - baseline_faa

        # Determine regulation success based on target state
        if target_state == "positive":
            regulation_success = faa_shift >= self._success_threshold
        elif target_state == "neutral":
            regulation_success = abs(faa_shift) <= self._success_threshold
        else:
            # Default to positive
            regulation_success = faa_shift >= self._success_threshold

        # Regulation score: 0-100 scale
        regulation_score = self._compute_regulation_score(
            faa_shift, target_state
        )

        # Effort index from alpha variability (0-1)
        effort_index = self._compute_effort_index(eeg_signals, fs)

        # Feedback message
        feedback_message = self._get_feedback_message(faa_shift, target_state)

        # Strategy suggestion based on current state
        strategy_suggestion = self._get_strategy(current_faa, faa_shift)

        result = {
            "current_faa": round(current_faa, 6),
            "baseline_faa": round(baseline_faa, 6),
            "faa_shift": round(faa_shift, 6),
            "regulation_success": regulation_success,
            "regulation_score": round(regulation_score, 2),
            "effort_index": round(effort_index, 4),
            "feedback_message": feedback_message,
            "target_state": target_state,
            "strategy_suggestion": strategy_suggestion,
            "has_baseline": user_id in self._baselines,
        }

        # Track history
        if user_id not in self._histories:
            self._histories[user_id] = []
        self._histories[user_id].append(result)
        if len(self._histories[user_id]) > 2000:
            self._histories[user_id] = self._histories[user_id][-2000:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get cumulative session statistics.

        Returns:
            Dict with n_epochs, success_rate, mean_score, mean_faa_shift,
            trend, has_baseline.
        """
        history = self._histories.get(user_id, [])
        if not history:
            return {
                "n_epochs": 0,
                "success_rate": 0.0,
                "mean_score": 0.0,
                "mean_faa_shift": 0.0,
                "trend": "insufficient_data",
                "has_baseline": user_id in self._baselines,
            }

        successes = [h["regulation_success"] for h in history]
        scores = [h["regulation_score"] for h in history]
        shifts = [h["faa_shift"] for h in history]

        success_rate = sum(successes) / len(successes)

        return {
            "n_epochs": len(history),
            "success_rate": round(success_rate, 4),
            "mean_score": round(float(np.mean(scores)), 2),
            "mean_faa_shift": round(float(np.mean(shifts)), 6),
            "max_score": round(float(np.max(scores)), 2),
            "trend": self._compute_trend(scores),
            "has_baseline": user_id in self._baselines,
        }

    def get_strategies(self, state: Optional[str] = None) -> Dict[str, List[str]]:
        """Get cognitive reappraisal strategies.

        Args:
            state: If provided, return strategies for this state only.
                Options: 'negative_high_arousal', 'negative_low_arousal',
                'neutral', 'positive'.

        Returns:
            Dict mapping state names to lists of strategy strings.
        """
        if state is not None and state in _STRATEGIES:
            return {state: _STRATEGIES[state]}
        return dict(_STRATEGIES)

    def get_history(
        self,
        user_id: str = "default",
        last_n: Optional[int] = None,
    ) -> List[Dict]:
        """Get evaluation history.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of evaluation result dicts.
        """
        history = self._histories.get(user_id, [])
        if last_n is not None and last_n > 0:
            return history[-last_n:]
        return list(history)

    def reset(self, user_id: str = "default"):
        """Clear baseline and session history for a user."""
        self._baselines.pop(user_id, None)
        self._histories.pop(user_id, None)

    # ── Private helpers ──────────────────────────────────────────

    def _alpha_power(self, signal: np.ndarray, fs: float) -> float:
        """Compute alpha (8-12 Hz) band power via Welch PSD."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return 0.0

        try:
            if _SCIPY_AVAILABLE:
                freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
            else:
                freqs, psd = _numpy_welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.0

        mask = (freqs >= 8) & (freqs <= 12)
        if not np.any(mask):
            return 0.0

        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(psd[mask], freqs[mask]))
        return float(np.trapz(psd[mask], freqs[mask]))

    def _frontal_alpha_powers(
        self, signals: np.ndarray, fs: float
    ) -> tuple:
        """Extract alpha power from AF7 (ch1) and AF8 (ch2).

        If fewer than 3 channels are available, uses ch0 for AF7 and
        ch1 for AF8 (or ch0 only if single channel).

        Returns:
            (af7_alpha, af8_alpha) tuple of floats.
        """
        n_ch = signals.shape[0]

        if n_ch >= 3:
            # Standard Muse 2 layout: ch1=AF7, ch2=AF8
            af7_alpha = self._alpha_power(signals[1], fs)
            af8_alpha = self._alpha_power(signals[2], fs)
        elif n_ch == 2:
            af7_alpha = self._alpha_power(signals[0], fs)
            af8_alpha = self._alpha_power(signals[1], fs)
        else:
            # Single channel -- FAA will be ~0
            af7_alpha = self._alpha_power(signals[0], fs)
            af8_alpha = af7_alpha
        return af7_alpha, af8_alpha

    def _compute_regulation_score(
        self, faa_shift: float, target_state: str
    ) -> float:
        """Compute regulation score 0-100.

        For 'positive' target: score scales with positive FAA shift.
        For 'neutral' target: score scales with proximity to zero shift.
        """
        if target_state == "neutral":
            # Closer to zero shift = higher score
            raw = max(0.0, 1.0 - abs(faa_shift) / 0.3)
        else:
            # Positive shift = higher score; negative shift = 0
            raw = float(np.clip(faa_shift / 0.3, 0.0, 1.0))

        return round(raw * 100.0, 2)

    def _compute_effort_index(
        self, signals: np.ndarray, fs: float
    ) -> float:
        """Compute regulation effort from alpha variability.

        Higher alpha power variance across short sub-windows indicates
        more effortful regulation (the brain is actively modulating).

        Returns:
            Float 0-1, where 1 = maximum detected effort.
        """
        n_ch = signals.shape[0]
        # Use AF7/AF8 channels
        if n_ch >= 3:
            channels_to_use = [1, 2]
        elif n_ch == 2:
            channels_to_use = [0, 1]
        else:
            channels_to_use = [0]

        window_samples = int(fs * 0.5)  # 500ms sub-windows
        if window_samples < 4 or signals.shape[1] < window_samples:
            return 0.0

        all_variances = []
        for ch_idx in channels_to_use:
            channel = signals[ch_idx]
            n_windows = max(1, len(channel) // window_samples)
            sub_powers = []
            for i in range(n_windows):
                start = i * window_samples
                end = start + window_samples
                if end > len(channel):
                    break
                sub_powers.append(self._alpha_power(channel[start:end], fs))

            if len(sub_powers) >= 2:
                all_variances.append(float(np.std(sub_powers)))

        if not all_variances:
            return 0.0

        mean_var = float(np.mean(all_variances))
        # Normalize: 0.5 uV^2 std -> effort ~0.5; clip at 1.0
        effort = float(np.clip(mean_var / 1.0, 0.0, 1.0))
        return effort

    def _get_feedback_message(
        self, faa_shift: float, target_state: str
    ) -> str:
        """Select feedback message based on FAA shift magnitude."""
        if target_state == "neutral":
            deviation = abs(faa_shift)
            if deviation <= 0.02:
                return _FEEDBACK["strong_success"]
            elif deviation <= 0.05:
                return _FEEDBACK["moderate_success"]
            elif deviation <= 0.10:
                return _FEEDBACK["mild_success"]
            else:
                return _FEEDBACK["no_change"]

        # Positive target
        if faa_shift >= 0.15:
            return _FEEDBACK["strong_success"]
        elif faa_shift >= 0.08:
            return _FEEDBACK["moderate_success"]
        elif faa_shift >= 0.03:
            return _FEEDBACK["mild_success"]
        elif faa_shift >= -0.03:
            return _FEEDBACK["no_change"]
        elif faa_shift >= -0.10:
            return _FEEDBACK["mild_failure"]
        else:
            return _FEEDBACK["strong_failure"]

    def _get_strategy(
        self, current_faa: float, faa_shift: float
    ) -> str:
        """Select cognitive reappraisal strategy based on current FAA state."""
        if current_faa > 0.1:
            state = "positive"
        elif current_faa > -0.1:
            state = "neutral"
        elif faa_shift < -0.05:
            # Negative and getting worse -- high arousal negative
            state = "negative_high_arousal"
        else:
            state = "negative_low_arousal"

        strategies = _STRATEGIES[state]
        # Rotate through strategies based on history length
        # Use a simple hash of the faa values to pick varied suggestions
        idx = int(abs(current_faa * 1000 + faa_shift * 1000)) % len(strategies)
        return strategies[idx]

    def _compute_trend(self, scores: List[float]) -> str:
        """Compute score trend over session."""
        if len(scores) < 10:
            return "insufficient_data"
        mid = len(scores) // 2
        first_half = float(np.mean(scores[:mid]))
        second_half = float(np.mean(scores[mid:]))
        diff = second_half - first_half
        if diff > 3.0:
            return "improving"
        elif diff < -3.0:
            return "declining"
        return "stable"


# ── numpy-only Welch fallback ─────────────────────────────────────────────────

def _numpy_welch(
    x: np.ndarray, fs: float, nperseg: int
):
    """Minimal Welch PSD using numpy FFT (used when scipy is unavailable)."""
    step = nperseg // 2
    n = len(x)
    window = np.hanning(nperseg)
    segments = []
    start = 0
    while start + nperseg <= n:
        seg = x[start: start + nperseg] * window
        segments.append(seg)
        start += step
    if not segments:
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        return freqs, np.zeros(len(freqs))
    psds = [np.abs(np.fft.rfft(s)) ** 2 for s in segments]
    psd = np.mean(psds, axis=0)
    win_power = np.sum(window ** 2)
    psd = psd / (fs * win_power)
    psd[1:-1] *= 2  # single-sided
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, psd


def _band_power(signal: np.ndarray, fs: float, flo: float, fhi: float) -> float:
    """Compute power in a frequency band via Welch PSD."""
    nperseg = min(len(signal), int(fs * 2))
    if nperseg < 8:
        # Too short: fall back to variance-scaled heuristic
        return float(np.var(signal)) + 1e-12
    try:
        if _SCIPY_AVAILABLE:
            freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
        else:
            freqs, psd = _numpy_welch(signal, fs=fs, nperseg=nperseg)
    except Exception:
        return float(np.var(signal)) + 1e-12
    mask = (freqs >= flo) & (freqs <= fhi)
    if not np.any(mask):
        return 1e-12
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd[mask], freqs[mask])) + 1e-12
    return float(np.trapz(psd[mask], freqs[mask])) + 1e-12


# ── EmotionRegulationBiofeedback ──────────────────────────────────────────────

_VALID_EMOTION_STATES = frozenset(
    ["anxious", "calm", "focused", "stressed", "neutral"]
)
_VALID_BIOFEEDBACK_CUES = frozenset(
    ["increase_alpha", "decrease_beta", "balanced"]
)
_VALID_REGULATION_TRENDS = frozenset(["improving", "declining", "stable"])


class EmotionRegulationBiofeedback:
    """Closed-loop neurofeedback for real-time emotion regulation.

    Tracks three EEG biomarkers via exponential moving averages (EMA):
      - Alpha asymmetry (FAA): log(AF8_alpha) - log(AF7_alpha)
      - Anxiety index: theta/alpha ratio (frontal theta / frontal alpha)
      - Arousal regulation: beta/alpha ratio

    All raw data is discarded; only EMA state is retained.

    Args:
        ema_alpha: EMA decay factor (0–1). Smaller = more smoothing.
            Default 0.2 gives a 5-frame effective window.
        trend_window: Number of regulation_score updates used to classify
            the regulation trend as improving/declining/stable.
    """

    def __init__(
        self,
        ema_alpha: float = 0.2,
        trend_window: int = 10,
    ):
        self._ema_alpha = ema_alpha
        self._trend_window = trend_window

        # EMA state (per-instance, shared across calls for one user)
        self._ema_faa: Optional[float] = None
        self._ema_theta_alpha: Optional[float] = None
        self._ema_beta_alpha: Optional[float] = None

        # Session accumulators (lightweight — no raw data)
        self._session_scores: List[float] = []
        self._session_states: List[str] = []
        self._session_count: int = 0
        self._session_duration: int = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Analyze one EEG epoch and return biofeedback result.

        Args:
            eeg: (n_channels, n_samples) or (n_samples,) array.
                For Muse 2: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10.
            fs: Sampling rate in Hz.

        Returns:
            Dict with keys:
                emotion_state (str): one of "anxious"|"calm"|"focused"|"stressed"|"neutral"
                regulation_score (float): 0–1
                biofeedback_cue (str): "increase_alpha"|"decrease_beta"|"balanced"
                alpha_asymmetry (float): FAA value
                anxiety_index (float): 0–1
                regulation_trend (str): "improving"|"declining"|"stable"
                session_duration (int): seconds accumulated via update_session()
        """
        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        faa = self._compute_faa(eeg, fs)
        theta_alpha = self._compute_theta_alpha(eeg, fs)
        beta_alpha = self._compute_beta_alpha(eeg, fs)

        # Update EMAs
        self._ema_faa = self._update_ema(self._ema_faa, faa)
        self._ema_theta_alpha = self._update_ema(self._ema_theta_alpha, theta_alpha)
        self._ema_beta_alpha = self._update_ema(self._ema_beta_alpha, beta_alpha)

        smooth_faa = self._ema_faa
        smooth_theta_alpha = self._ema_theta_alpha
        smooth_beta_alpha = self._ema_beta_alpha

        anxiety_index = self._anxiety_from_theta_alpha(smooth_theta_alpha)
        regulation_score = self._compute_regulation_score(
            smooth_faa, anxiety_index, smooth_beta_alpha
        )
        emotion_state = self._classify_state(
            smooth_faa, anxiety_index, smooth_beta_alpha
        )
        biofeedback_cue = self._choose_cue(
            emotion_state, smooth_faa, smooth_beta_alpha
        )
        regulation_trend = self._compute_trend()

        return {
            "emotion_state": emotion_state,
            "regulation_score": round(float(regulation_score), 4),
            "biofeedback_cue": biofeedback_cue,
            "alpha_asymmetry": round(float(smooth_faa), 6),
            "anxiety_index": round(float(anxiety_index), 4),
            "regulation_trend": regulation_trend,
            "session_duration": int(self._session_duration),
        }

    def update_session(
        self,
        eeg: np.ndarray,
        fs: float = 256.0,
        duration_sec: int = 0,
    ) -> Dict:
        """Accumulate a session epoch and update session counters.

        Args:
            eeg: EEG epoch for this update.
            fs: Sampling rate in Hz.
            duration_sec: Seconds to add to the session clock.

        Returns:
            Same dict as predict() with session_count included.
        """
        result = self.predict(eeg, fs)
        self._session_count += 1
        self._session_duration += max(0, int(duration_sec))
        self._session_scores.append(result["regulation_score"])
        self._session_states.append(result["emotion_state"])
        # Keep only last trend_window * 4 entries to bound memory
        max_keep = max(self._trend_window * 4, 40)
        if len(self._session_scores) > max_keep:
            self._session_scores = self._session_scores[-max_keep:]
            self._session_states = self._session_states[-max_keep:]
        result["session_count"] = self._session_count
        return result

    def get_session_summary(self) -> Dict:
        """Return summary statistics for the current session.

        Returns:
            Dict with:
                mean_regulation_score (float): average score 0–1
                peak_score (float): highest score seen
                session_count (int): number of update_session() calls
                dominant_state (str): most frequent emotion_state
        """
        if not self._session_scores:
            return {
                "mean_regulation_score": 0.0,
                "peak_score": 0.0,
                "session_count": 0,
                "dominant_state": "neutral",
            }

        dominant = "neutral"
        if self._session_states:
            counts: Dict[str, int] = {}
            for s in self._session_states:
                counts[s] = counts.get(s, 0) + 1
            dominant = max(counts, key=lambda k: counts[k])

        return {
            "mean_regulation_score": round(
                float(np.mean(self._session_scores)), 4
            ),
            "peak_score": round(float(np.max(self._session_scores)), 4),
            "session_count": self._session_count,
            "dominant_state": dominant,
        }

    def reset(self) -> None:
        """Clear all EMA state and session history."""
        self._ema_faa = None
        self._ema_theta_alpha = None
        self._ema_beta_alpha = None
        self._session_scores = []
        self._session_states = []
        self._session_count = 0
        self._session_duration = 0

    # ── Private helpers ──────────────────────────────────────────────────────

    def _update_ema(self, current: Optional[float], new: float) -> float:
        """Update exponential moving average."""
        if current is None:
            return new
        return self._ema_alpha * new + (1.0 - self._ema_alpha) * current

    def _compute_faa(self, eeg: np.ndarray, fs: float) -> float:
        """Compute frontal alpha asymmetry: log(AF8_alpha) - log(AF7_alpha)."""
        n_ch = eeg.shape[0]
        eps = 1e-12
        if n_ch >= 3:
            af7_alpha = _band_power(eeg[1], fs, 8.0, 12.0)
            af8_alpha = _band_power(eeg[2], fs, 8.0, 12.0)
        elif n_ch == 2:
            af7_alpha = _band_power(eeg[0], fs, 8.0, 12.0)
            af8_alpha = _band_power(eeg[1], fs, 8.0, 12.0)
        else:
            af7_alpha = _band_power(eeg[0], fs, 8.0, 12.0)
            af8_alpha = af7_alpha
        return float(np.log(af8_alpha + eps) - np.log(af7_alpha + eps))

    def _compute_theta_alpha(self, eeg: np.ndarray, fs: float) -> float:
        """Compute theta/alpha ratio (frontal channel)."""
        ch = eeg[1] if eeg.shape[0] >= 2 else eeg[0]
        theta = _band_power(ch, fs, 4.0, 8.0)
        alpha = _band_power(ch, fs, 8.0, 12.0)
        return float(theta / (alpha + 1e-12))

    def _compute_beta_alpha(self, eeg: np.ndarray, fs: float) -> float:
        """Compute beta/alpha ratio (frontal channel)."""
        ch = eeg[1] if eeg.shape[0] >= 2 else eeg[0]
        beta = _band_power(ch, fs, 12.0, 30.0)
        alpha = _band_power(ch, fs, 8.0, 12.0)
        return float(beta / (alpha + 1e-12))

    def _anxiety_from_theta_alpha(self, theta_alpha: float) -> float:
        """Map theta/alpha ratio to anxiety_index in [0, 1].

        theta/alpha > 2.5 → highly anxious; < 0.8 → calm.
        Sigmoid-like mapping clipped to [0, 1].
        """
        # Logistic: centre at 1.5, scale 0.8
        raw = 1.0 / (1.0 + np.exp(-0.8 * (theta_alpha - 1.5)))
        return float(np.clip(raw, 0.0, 1.0))

    def _compute_regulation_score(
        self,
        faa: float,
        anxiety_index: float,
        beta_alpha: float,
    ) -> float:
        """Compute regulation_score in [0, 1].

        Higher score = better regulated state:
        - positive FAA (left activation) → higher
        - lower anxiety_index → higher
        - moderate beta/alpha (not too high) → higher
        """
        # FAA component: sigmoid around 0, positive = good
        faa_score = float(np.clip(0.5 + faa * 1.5, 0.0, 1.0))
        # Anxiety component: lower anxiety = higher score
        anxiety_score = 1.0 - anxiety_index
        # Arousal component: beta/alpha > 3.0 is stressed; < 1.0 is too drowsy
        arousal_score = float(np.clip(1.0 - (beta_alpha - 1.5) / 3.0, 0.0, 1.0))
        # Weighted blend
        score = 0.4 * faa_score + 0.4 * anxiety_score + 0.2 * arousal_score
        return float(np.clip(score, 0.0, 1.0))

    def _classify_state(
        self,
        faa: float,
        anxiety_index: float,
        beta_alpha: float,
    ) -> str:
        """Map EEG features to one of the five emotion_state labels."""
        if anxiety_index > 0.65:
            # High theta/alpha → anxious or stressed
            if beta_alpha > 2.5:
                return "stressed"
            return "anxious"
        if faa > 0.15 and anxiety_index < 0.40:
            # Positive FAA + low anxiety → calm
            if beta_alpha > 1.8:
                return "focused"
            return "calm"
        if beta_alpha > 2.0 and anxiety_index < 0.50:
            return "focused"
        if faa < -0.15 or anxiety_index > 0.55:
            return "stressed"
        return "neutral"

    def _choose_cue(
        self,
        state: str,
        faa: float,
        beta_alpha: float,
    ) -> str:
        """Choose the biofeedback cue based on current state."""
        if state in ("calm", "focused") and faa > 0.0 and beta_alpha < 2.0:
            return "balanced"
        if state in ("anxious", "stressed") and faa < 0.0:
            return "increase_alpha"
        if beta_alpha > 2.5:
            return "decrease_beta"
        if faa < -0.1:
            return "increase_alpha"
        return "balanced"

    def _compute_trend(self) -> str:
        """Classify regulation trend from recent session scores."""
        scores = self._session_scores
        n = len(scores)
        if n < self._trend_window:
            return "stable"
        recent = scores[-self._trend_window:]
        # Linear regression slope via numpy
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent, dtype=float)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-12:
            return "stable"
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
        if slope > 0.01:
            return "improving"
        if slope < -0.01:
            return "declining"
        return "stable"


# ── Per-user singleton registry ───────────────────────────────────────────────

_instances: Dict[str, "EmotionRegulationBiofeedback"] = {}
_instances_lock = threading.Lock()


def get_emotion_regulation_biofeedback(
    user_id: str = "default",
) -> "EmotionRegulationBiofeedback":
    """Return a singleton EmotionRegulationBiofeedback for the given user_id.

    Different user_ids return different instances.  Same user_id always
    returns the same instance (thread-safe).
    """
    global _instances
    with _instances_lock:
        if user_id not in _instances:
            _instances[user_id] = EmotionRegulationBiofeedback()
        return _instances[user_id]
