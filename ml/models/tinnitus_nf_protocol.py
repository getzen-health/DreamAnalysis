"""Alpha neurofeedback protocol for tinnitus relief.

Trains the user to increase alpha power at temporal sites (TP9/TP10),
which reduces tinnitus perception. The Muse 2's TP9/TP10 electrodes
sit directly over the auditory cortex — ideal for this protocol.

Protocol:
1. Baseline: record 2 min resting alpha at TP9/TP10
2. Training: visual/auditory feedback when alpha exceeds baseline by threshold
3. Reward when temporal alpha > baseline * reward_threshold

References:
    Crocetti et al. (2011) — Alpha neurofeedback for tinnitus
    Dohrmann et al. (2007) — EEG neurofeedback for tinnitus treatment
"""
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


class TinnitusNFProtocol:
    """Alpha up-training neurofeedback for tinnitus relief.

    Target: increase alpha (8-12 Hz) power at TP9 (ch0) and TP10 (ch3).
    """

    def __init__(
        self,
        reward_threshold: float = 1.1,
        target_channels: tuple = (0, 3),
        fs: float = 256.0,
    ):
        self._reward_threshold = reward_threshold
        self._target_channels = target_channels
        self._fs = fs
        self._baselines: Dict[str, float] = {}
        self._sessions: Dict[str, List[Dict]] = {}

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record baseline alpha power from resting state.

        Args:
            signals: (n_channels, n_samples) EEG array.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with baseline_alpha and channel contributions.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        alpha_powers = []
        for ch in self._target_channels:
            if ch < signals.shape[0]:
                power = self._alpha_power(signals[ch], fs)
                alpha_powers.append(power)

        if not alpha_powers:
            return {"baseline_alpha": 0.0, "error": "no valid channels"}

        baseline = float(np.mean(alpha_powers))
        self._baselines[user_id] = baseline

        return {
            "baseline_alpha": round(baseline, 6),
            "channel_powers": [round(p, 6) for p in alpha_powers],
            "baseline_set": True,
        }

    def evaluate(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Evaluate a training epoch — give reward feedback.

        Args:
            signals: (n_channels, n_samples) EEG epoch.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with reward (bool), alpha_ratio, feedback_intensity,
            current_alpha, and session statistics.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        baseline = self._baselines.get(user_id, 0)

        # Current alpha at temporal channels
        alpha_powers = []
        for ch in self._target_channels:
            if ch < signals.shape[0]:
                alpha_powers.append(self._alpha_power(signals[ch], fs))

        current_alpha = float(np.mean(alpha_powers)) if alpha_powers else 0.0

        # Alpha ratio vs baseline
        if baseline > 1e-10:
            alpha_ratio = current_alpha / baseline
        else:
            alpha_ratio = 1.0

        # Reward
        reward = alpha_ratio >= self._reward_threshold

        # Feedback intensity (0-1 scale, proportional to alpha increase)
        feedback_intensity = float(np.clip((alpha_ratio - 1.0) / 0.5, 0, 1))

        # Feedback tone (higher alpha = higher pitch)
        feedback_tone_hz = 440.0 + (alpha_ratio - 1.0) * 200 if reward else None

        result = {
            "reward": reward,
            "alpha_ratio": round(alpha_ratio, 4),
            "current_alpha": round(current_alpha, 6),
            "baseline_alpha": round(baseline, 6),
            "feedback_intensity": round(feedback_intensity, 4),
            "feedback_tone_hz": round(feedback_tone_hz, 1) if feedback_tone_hz else None,
            "has_baseline": baseline > 1e-10,
        }

        # Track session
        if user_id not in self._sessions:
            self._sessions[user_id] = []
        self._sessions[user_id].append(result)
        if len(self._sessions[user_id]) > 1000:
            self._sessions[user_id] = self._sessions[user_id][-1000:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get training session statistics."""
        history = self._sessions.get(user_id, [])
        if not history:
            return {"n_epochs": 0, "has_baseline": user_id in self._baselines}

        rewards = [h["reward"] for h in history]
        ratios = [h["alpha_ratio"] for h in history]

        return {
            "n_epochs": len(history),
            "reward_rate": round(sum(rewards) / len(rewards), 4),
            "mean_alpha_ratio": round(float(np.mean(ratios)), 4),
            "max_alpha_ratio": round(float(np.max(ratios)), 4),
            "has_baseline": user_id in self._baselines,
            "trend": self._compute_trend(ratios),
        }

    def reset(self, user_id: str = "default"):
        """Clear session and baseline."""
        self._baselines.pop(user_id, None)
        self._sessions.pop(user_id, None)

    # ── Private helpers ──────────────────────────────────────────

    def _alpha_power(self, signal: np.ndarray, fs: float) -> float:
        """Compute alpha (8-12 Hz) band power via Welch."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return 0.0

        try:
            freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.0

        mask = (freqs >= 8) & (freqs <= 12)
        if not np.any(mask):
            return 0.0

        return float(np.trapezoid(psd[mask], freqs[mask]) if hasattr(np, 'trapezoid')
                     else np.trapz(psd[mask], freqs[mask]))

    def _compute_trend(self, ratios: List[float]) -> str:
        """Compute alpha ratio trend over session."""
        if len(ratios) < 10:
            return "insufficient_data"
        first_half = np.mean(ratios[:len(ratios)//2])
        second_half = np.mean(ratios[len(ratios)//2:])
        diff = second_half - first_half
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"
