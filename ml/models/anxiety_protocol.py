"""Neurofeedback protocol for anxiety reduction.

Implements alpha up-training and high-beta down-training at frontal
sites (AF7/AF8) to reduce anxiety symptoms. High-beta (20-30 Hz)
power at frontal sites is the primary EEG marker of anxiety; training
users to suppress it while increasing alpha reduces subjective anxiety.

Protocol:
1. Baseline: record 2 min resting state
2. Training: reward when frontal alpha increases AND high-beta decreases
3. Track anxiety index over session (should decline)

References:
    Hammond (2005) — Neurofeedback treatment of anxiety disorders
    Hardt & Kamiya (1978) — Alpha biofeedback for anxiety
    Kerson et al. (2009) — Alpha asymmetry neurofeedback for anxiety
"""
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


class AnxietyProtocol:
    """Frontal alpha/high-beta neurofeedback for anxiety reduction.

    Target: increase frontal alpha (8-12 Hz) and decrease high-beta
    (20-30 Hz) at AF7 (ch1) and AF8 (ch2).
    """

    def __init__(
        self,
        alpha_threshold: float = 1.1,
        hbeta_threshold: float = 0.9,
        fs: float = 256.0,
    ):
        """
        Args:
            alpha_threshold: Alpha must exceed baseline * this for reward.
            hbeta_threshold: High-beta must be below baseline * this for reward.
            fs: Sampling rate.
        """
        self._alpha_threshold = alpha_threshold
        self._hbeta_threshold = hbeta_threshold
        self._fs = fs
        self._baselines: Dict[str, Dict] = {}
        self._sessions: Dict[str, List[Dict]] = {}

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record baseline alpha and high-beta at frontal sites.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with baseline alpha and high-beta powers.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        alpha_power = self._frontal_band_power(signals, fs, 8, 12)
        hbeta_power = self._frontal_band_power(signals, fs, 20, 30)
        anxiety_index = self._compute_anxiety_index(alpha_power, hbeta_power)

        baseline = {
            "alpha": alpha_power,
            "high_beta": hbeta_power,
            "anxiety_index": anxiety_index,
        }
        self._baselines[user_id] = baseline

        return {
            "baseline_set": True,
            "baseline_alpha": round(alpha_power, 6),
            "baseline_high_beta": round(hbeta_power, 6),
            "baseline_anxiety_index": round(anxiety_index, 4),
        }

    def evaluate(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Evaluate a training epoch — give anxiety reduction feedback.

        Args:
            signals: (n_channels, n_samples) EEG epoch.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with reward, alpha/hbeta ratios, anxiety_index,
            feedback parameters, and anxiety level.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        baseline = self._baselines.get(user_id, {})
        bl_alpha = baseline.get("alpha", 0)
        bl_hbeta = baseline.get("high_beta", 0)

        # Current powers
        alpha_power = self._frontal_band_power(signals, fs, 8, 12)
        hbeta_power = self._frontal_band_power(signals, fs, 20, 30)
        anxiety_index = self._compute_anxiety_index(alpha_power, hbeta_power)

        # Ratios vs baseline
        has_baseline = bl_alpha > 1e-10 and bl_hbeta > 1e-10
        if has_baseline:
            alpha_ratio = alpha_power / bl_alpha
            hbeta_ratio = hbeta_power / bl_hbeta
        else:
            alpha_ratio = 1.0
            hbeta_ratio = 1.0

        # Reward: alpha up AND high-beta down
        alpha_success = alpha_ratio >= self._alpha_threshold
        hbeta_success = hbeta_ratio <= self._hbeta_threshold
        reward = alpha_success and hbeta_success

        # Partial reward for either condition met
        partial_score = 0.0
        if alpha_success:
            partial_score += 0.5
        if hbeta_success:
            partial_score += 0.5

        # Feedback intensity (0-1)
        alpha_improvement = max(0, alpha_ratio - 1.0) / 0.5
        hbeta_reduction = max(0, 1.0 - hbeta_ratio) / 0.5
        feedback_intensity = float(np.clip(
            0.5 * alpha_improvement + 0.5 * hbeta_reduction, 0, 1
        ))

        # Anxiety level classification
        if anxiety_index >= 0.7:
            anxiety_level = "high"
        elif anxiety_index >= 0.5:
            anxiety_level = "moderate"
        elif anxiety_index >= 0.3:
            anxiety_level = "mild"
        else:
            anxiety_level = "low"

        # Calming instruction based on current state
        if anxiety_index >= 0.6:
            instruction = "slow_breathing: inhale 4s, hold 4s, exhale 6s"
        elif anxiety_index >= 0.4:
            instruction = "body_scan: notice and release tension in shoulders"
        elif anxiety_index >= 0.2:
            instruction = "maintain: keep this calm, relaxed state"
        else:
            instruction = "excellent: deeply relaxed state achieved"

        # Feedback tone (lower anxiety = lower, calming tone)
        if reward:
            feedback_tone_hz = 220.0 + (1.0 - anxiety_index) * 220
        else:
            feedback_tone_hz = None

        result = {
            "reward": reward,
            "partial_score": round(partial_score, 2),
            "alpha_ratio": round(alpha_ratio, 4),
            "hbeta_ratio": round(hbeta_ratio, 4),
            "alpha_success": alpha_success,
            "hbeta_success": hbeta_success,
            "anxiety_index": round(anxiety_index, 4),
            "anxiety_level": anxiety_level,
            "feedback_intensity": round(feedback_intensity, 4),
            "feedback_tone_hz": round(feedback_tone_hz, 1) if feedback_tone_hz else None,
            "instruction": instruction,
            "has_baseline": has_baseline,
            "current_alpha": round(alpha_power, 6),
            "current_high_beta": round(hbeta_power, 6),
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
        anxiety = [h["anxiety_index"] for h in history]

        # Anxiety trend
        if len(anxiety) >= 10:
            first_half = np.mean(anxiety[:len(anxiety)//2])
            second_half = np.mean(anxiety[len(anxiety)//2:])
            diff = second_half - first_half
            if diff < -0.05:
                trend = "improving"
            elif diff > 0.05:
                trend = "worsening"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "n_epochs": len(history),
            "reward_rate": round(sum(rewards) / len(rewards), 4),
            "mean_anxiety": round(float(np.mean(anxiety)), 4),
            "anxiety_start": round(anxiety[0], 4),
            "anxiety_end": round(anxiety[-1], 4),
            "trend": trend,
            "has_baseline": user_id in self._baselines,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get epoch history."""
        history = self._sessions.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return history

    def reset(self, user_id: str = "default"):
        """Clear session and baseline."""
        self._baselines.pop(user_id, None)
        self._sessions.pop(user_id, None)

    # ── Private helpers ─────────────────────────────────────────

    def _frontal_band_power(
        self, signals: np.ndarray, fs: float, low: float, high: float
    ) -> float:
        """Compute band power at frontal channels (AF7=ch1, AF8=ch2)."""
        frontal_channels = [1, 2] if signals.shape[0] >= 3 else [0]
        powers = []
        for ch in frontal_channels:
            if ch >= signals.shape[0]:
                continue
            nperseg = min(len(signals[ch]), int(fs * 2))
            if nperseg < 4:
                continue
            try:
                freqs, psd = scipy_signal.welch(
                    signals[ch], fs=fs, nperseg=nperseg
                )
                mask = (freqs >= low) & (freqs <= high)
                if np.any(mask):
                    if hasattr(np, "trapezoid"):
                        powers.append(float(np.trapezoid(psd[mask], freqs[mask])))
                    else:
                        powers.append(float(np.trapz(psd[mask], freqs[mask])))
            except Exception:
                pass
        return float(np.mean(powers)) if powers else 0.0

    def _compute_anxiety_index(self, alpha: float, high_beta: float) -> float:
        """Compute anxiety index from alpha and high-beta.

        Higher high-beta relative to alpha = more anxiety.
        """
        total = alpha + high_beta + 1e-10
        # High-beta fraction is the anxiety signal
        anxiety_raw = high_beta / total
        return float(np.clip(anxiety_raw * 2, 0, 1))
