"""Real-time seizure detection from 4-channel consumer EEG.

Feature-based heuristic approach using validated biomarkers:
1. High-frequency power surge (beta/gamma elevation)
2. Rhythmic spike-wave patterns (amplitude + periodicity)
3. Cross-channel hypersynchrony (seizures increase inter-channel correlation)
4. Suppression of normal alpha rhythm

IMPORTANT DISCLAIMER: This is NOT a medical device. It is a research tool
and must never be used as a primary seizure monitor. Users with epilepsy
should consult their neurologist. False negatives can occur.

References:
    AES 2025 — 4-channel wearable algorithm approaches expert-level (76.8% PPA)
    Scientific Reports 2025 — 1D CNN-LSTM: 96.94% on CHB-MIT
    Frontiers in Neurology 2024 — Pre-ictal state detection review
"""
from typing import Dict, List, Optional

import numpy as np
_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
from scipy import signal as scipy_signal

MEDICAL_DISCLAIMER = (
    "NOT a medical device. Research use only. "
    "Consult a neurologist for clinical seizure monitoring."
)


class SeizureDetector:
    """Real-time seizure detection from 4-channel EEG.

    Uses feature-based heuristics on Muse 2 channels:
    TP9 (ch0), AF7 (ch1), AF8 (ch2), TP10 (ch3).
    Temporal channels (TP9/TP10) are primary for temporal lobe epilepsy.
    """

    def __init__(
        self,
        alarm_threshold: float = 0.7,
        alarm_trigger_count: int = 3,
        fs: float = 256.0,
    ):
        self._alarm_threshold = float(np.clip(alarm_threshold, 0.1, 0.99))
        self._alarm_trigger_count = max(1, alarm_trigger_count)
        self._fs = fs
        self._consecutive: Dict[str, int] = {}
        self._event_log: Dict[str, List[Dict]] = {}
        self._baselines: Dict[str, Dict] = {}

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state EEG baseline for comparison.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate (defaults to instance fs).
            user_id: User identifier.

        Returns:
            Dict with baseline band powers per channel.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        baseline = {}
        for ch in range(signals.shape[0]):
            powers = self._band_powers(signals[ch], fs)
            baseline[f"ch{ch}"] = powers

        avg_beta = float(np.mean([baseline[k]["beta"] for k in baseline]))
        avg_alpha = float(np.mean([baseline[k]["alpha"] for k in baseline]))
        baseline["avg_beta"] = avg_beta
        baseline["avg_alpha"] = avg_alpha
        self._baselines[user_id] = baseline
        return {"baseline_set": True, "n_channels": signals.shape[0], "user_id": user_id}

    def detect(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Detect seizure activity in an EEG epoch.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with seizure_probability, is_seizure, alarm_triggered,
            severity, channel_scores, cross_channel_synchronization,
            and medical_disclaimer.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_channels = signals.shape[0]
        n_samples = signals.shape[1]

        if n_samples < int(fs * 0.5):
            return self._empty_result(user_id)

        # Per-channel seizure scoring
        channel_scores = []
        for ch in range(n_channels):
            score = self._channel_seizure_score(signals[ch], fs, ch, user_id)
            channel_scores.append(score)

        # Cross-channel synchronization (seizures increase sync)
        mean_sync = self._cross_channel_sync(signals)

        # Overall seizure probability
        channel_mean = float(np.mean(channel_scores))
        sync_bonus = 0.3 if mean_sync > 0.8 else 0.15 * max(0, mean_sync - 0.5) / 0.3
        seizure_prob = float(np.clip(channel_mean * 0.7 + sync_bonus, 0, 1))

        # Consecutive detection logic
        if user_id not in self._consecutive:
            self._consecutive[user_id] = 0

        if seizure_prob > self._alarm_threshold:
            self._consecutive[user_id] += 1
        else:
            self._consecutive[user_id] = max(0, self._consecutive[user_id] - 1)

        alarm = self._consecutive[user_id] >= self._alarm_trigger_count

        if seizure_prob > self._alarm_threshold:
            severity = "critical" if alarm else "warning"
        elif seizure_prob > 0.4:
            severity = "elevated"
        else:
            severity = "normal"

        result = {
            "seizure_probability": round(seizure_prob, 4),
            "is_seizure": seizure_prob > self._alarm_threshold,
            "alarm_triggered": alarm,
            "consecutive_detections": self._consecutive[user_id],
            "channel_scores": [round(s, 4) for s in channel_scores],
            "cross_channel_synchronization": round(mean_sync, 4),
            "severity": severity,
            "n_channels": n_channels,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }

        # Log events
        if user_id not in self._event_log:
            self._event_log[user_id] = []
        self._event_log[user_id].append(result)
        if len(self._event_log[user_id]) > 1000:
            self._event_log[user_id] = self._event_log[user_id][-1000:]

        return result

    def get_event_log(self, user_id: str = "default", last_n: Optional[int] = None) -> List[Dict]:
        """Get seizure detection history."""
        log = self._event_log.get(user_id, [])
        if last_n:
            log = log[-last_n:]
        return log

    def get_status(self, user_id: str = "default") -> Dict:
        """Get current monitoring status."""
        log = self._event_log.get(user_id, [])
        if not log:
            return {
                "monitoring": False,
                "total_epochs": 0,
                "seizure_events": 0,
                "alarms_triggered": 0,
            }

        seizure_count = sum(1 for e in log if e["is_seizure"])
        alarm_count = sum(1 for e in log if e["alarm_triggered"])
        return {
            "monitoring": True,
            "total_epochs": len(log),
            "seizure_events": seizure_count,
            "alarms_triggered": alarm_count,
            "last_severity": log[-1]["severity"],
            "has_baseline": user_id in self._baselines,
            "consecutive_detections": self._consecutive.get(user_id, 0),
        }

    def set_alarm_threshold(self, threshold: float):
        """Update alarm threshold (0.1-0.99)."""
        self._alarm_threshold = float(np.clip(threshold, 0.1, 0.99))

    def reset(self, user_id: str = "default"):
        """Clear all data for a user."""
        self._consecutive.pop(user_id, None)
        self._event_log.pop(user_id, None)
        self._baselines.pop(user_id, None)

    # ── Private helpers ──────────────────────────────────────────

    def _channel_seizure_score(
        self, signal: np.ndarray, fs: float, ch_idx: int, user_id: str
    ) -> float:
        """Score a single channel for seizure-like activity (0-1)."""
        powers = self._band_powers(signal, fs)
        beta = powers["beta"]
        alpha = powers["alpha"]
        theta = powers["theta"]
        delta = powers["delta"]
        total = beta + alpha + theta + delta + 1e-10

        # Feature 1: High-frequency dominance (beta surge)
        # Seizures elevate beta and sometimes gamma relative to baseline
        beta_frac = beta / total
        hf_score = float(np.clip((beta_frac - 0.25) / 0.25, 0, 1))

        # Feature 2: Alpha suppression (normal alpha drops during seizure)
        alpha_frac = alpha / total
        alpha_suppression = float(np.clip(1.0 - alpha_frac / 0.3, 0, 1))

        # Feature 3: Amplitude spike (seizures have high amplitude)
        amplitude = float(np.std(signal))
        amp_score = float(np.clip((amplitude - 50) / 100, 0, 1))

        # Feature 4: Rhythmicity (seizures are more rhythmic than background)
        rhythm_score = self._rhythmicity(signal, fs)

        # Baseline comparison boost
        baseline_boost = 0.0
        baseline = self._baselines.get(user_id)
        if baseline and f"ch{ch_idx}" in baseline:
            bl_beta = baseline[f"ch{ch_idx}"]["beta"]
            if bl_beta > 1e-10:
                beta_increase = (beta - bl_beta) / bl_beta
                baseline_boost = float(np.clip(beta_increase * 0.3, 0, 0.2))

        # Temporal channels (TP9=0, TP10=3) weighted higher for TLE
        temporal_weight = 1.15 if ch_idx in (0, 3) else 1.0

        score = (
            0.30 * hf_score
            + 0.25 * alpha_suppression
            + 0.25 * amp_score
            + 0.20 * rhythm_score
            + baseline_boost
        ) * temporal_weight

        return float(np.clip(score, 0, 1))

    def _band_powers(self, signal: np.ndarray, fs: float) -> Dict[str, float]:
        """Extract band powers via Welch PSD."""
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return {"delta": 0, "theta": 0, "alpha": 0, "beta": 0}

        try:
            freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return {"delta": 0, "theta": 0, "alpha": 0, "beta": 0}

        def band_power(low, high):
            mask = (freqs >= low) & (freqs <= high)
            if not np.any(mask):
                return 0.0
            return float(_trapezoid(psd[mask], freqs[mask]) if hasattr(np, 'trapezoid')
                         else np.trapz(psd[mask], freqs[mask]))

        return {
            "delta": band_power(0.5, 4),
            "theta": band_power(4, 8),
            "alpha": band_power(8, 12),
            "beta": band_power(12, 30),
        }

    def _cross_channel_sync(self, signals: np.ndarray) -> float:
        """Compute mean pairwise correlation across channels."""
        n_channels = signals.shape[0]
        if n_channels < 2:
            return 0.5

        correlations = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                min_len = min(len(signals[i]), len(signals[j]))
                if min_len < 10:
                    continue
                s1 = signals[i, :min_len]
                s2 = signals[j, :min_len]
                std1, std2 = np.std(s1), np.std(s2)
                if std1 < 1e-10 or std2 < 1e-10:
                    correlations.append(0.0)
                    continue
                r = float(np.corrcoef(s1, s2)[0, 1])
                correlations.append(abs(r))

        return float(np.mean(correlations)) if correlations else 0.5

    def _rhythmicity(self, signal: np.ndarray, fs: float) -> float:
        """Measure signal rhythmicity via autocorrelation peak."""
        if len(signal) < int(fs * 0.5):
            return 0.0

        # Normalize
        sig = signal - np.mean(signal)
        std = np.std(sig)
        if std < 1e-10:
            return 0.0
        sig = sig / std

        # Autocorrelation at lags corresponding to 1-30 Hz
        min_lag = max(1, int(fs / 30))
        max_lag = min(len(sig) // 2, int(fs / 1))
        if min_lag >= max_lag:
            return 0.0

        autocorr = np.correlate(sig, sig, mode="full")
        mid = len(autocorr) // 2
        lags = autocorr[mid + min_lag: mid + max_lag]

        if len(lags) == 0:
            return 0.0

        peak = float(np.max(lags)) / len(sig)
        return float(np.clip(peak, 0, 1))

    def _empty_result(self, user_id: str) -> Dict:
        """Return result for too-short signals."""
        return {
            "seizure_probability": 0.0,
            "is_seizure": False,
            "alarm_triggered": False,
            "consecutive_detections": self._consecutive.get(user_id, 0),
            "channel_scores": [],
            "cross_channel_synchronization": 0.0,
            "severity": "normal",
            "n_channels": 0,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }
