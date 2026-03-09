"""Real-time seizure detection from consumer EEG.

Feature-based binary classification: ictal (seizure) vs interictal (non-seizure).

Validated biomarkers used:
1. High-frequency oscillations (HFO): power ratio 80-250 Hz / total power
2. Delta suppression: delta power drops during ictal activity
3. Rhythmic spiking: autocorrelation at 2-4 Hz (spike-wave discharge)
4. Line length: sum of |diff| — increases sharply during seizure
5. Band powers: delta/theta/alpha/beta/high_gamma

IMPORTANT DISCLAIMER: This is NOT a medical device. Research use only.
Never use as a primary seizure monitor. Consult a neurologist.

References:
    Scientific Reports 2025 — 1D CNN-LSTM: 96.94% on CHB-MIT
    AES 2025 — 4-channel wearable algorithm approaches expert-level (76.8% PPA)
    Frontiers in Neurology 2024 — Pre-ictal state detection review
"""
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal

MEDICAL_DISCLAIMER = (
    "NOT a medical device. Research use only. "
    "Consult a neurologist for clinical seizure monitoring."
)

_ALERT_LEVELS = ("none", "warning", "alert", "critical")

_singleton: Optional["SeizureDetector"] = None


def get_seizure_detector() -> "SeizureDetector":
    """Return the module-level SeizureDetector singleton."""
    global _singleton
    if _singleton is None:
        _singleton = SeizureDetector()
    return _singleton


class SeizureDetector:
    """Real-time seizure detector using feature-based heuristics.

    Accepts raw EEG windows of shape (n_channels, n_samples) or (n_samples,).
    Performs binary classification: 'ictal' vs 'interictal'.

    Rolling alarm buffer reduces false positives: alarm is only raised when
    N consecutive windows exceed the detection threshold.

    Args:
        alarm_threshold: Seizure probability required to increment consecutive
            counter (default 0.7).
        alarm_trigger_count: Number of consecutive ictal windows before alarm
            fires (default 3).
        fs: Default sampling rate in Hz (default 256).
    """

    def __init__(
        self,
        alarm_threshold: float = 0.7,
        alarm_trigger_count: int = 3,
        fs: float = 256.0,
    ) -> None:
        self._alarm_threshold = float(np.clip(alarm_threshold, 0.1, 0.99))
        self._alarm_trigger_count = max(1, int(alarm_trigger_count))
        self._fs = float(fs)
        # Per-user rolling alarm state
        self._consecutive_ictal: int = 0
        self._alarm_active: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        eeg: np.ndarray,
        fs: Optional[float] = None,
        window_seconds: float = 4.0,
    ) -> Dict:
        """Classify an EEG window as ictal or interictal.

        Args:
            eeg: Raw EEG array, shape (n_channels, n_samples) or (n_samples,).
            fs: Sampling rate in Hz. Defaults to instance fs.
            window_seconds: Expected window length (informational, not enforced).

        Returns:
            Dict with keys:
                detection       — 'ictal' or 'interictal'
                probability     — seizure probability 0.0–1.0 (alias of seizure_probability)
                seizure_probability — seizure probability 0.0–1.0
                alert_level     — 'none' | 'warning' | 'alert' | 'critical'
                features        — dict of extracted EEG features
                confidence      — confidence score 0.0–1.0
                alarm_active    — bool, True when consecutive threshold met
                consecutive_ictal — int, current consecutive ictal count
                threshold       — current alarm threshold
                medical_disclaimer — safety disclaimer string
        """
        fs = float(fs) if fs is not None else self._fs
        eeg = np.asarray(eeg, dtype=float)

        # Normalise to 2-D (n_channels, n_samples)
        if eeg.ndim == 1:
            signals = eeg.reshape(1, -1)
        else:
            signals = eeg

        n_channels, n_samples = signals.shape
        min_samples = max(4, int(fs * 0.5))

        if n_samples < min_samples:
            return self._empty_predict()

        # Extract features per channel and aggregate
        channel_features = [self._extract_channel_features(signals[ch], fs) for ch in range(n_channels)]
        agg = self._aggregate_features(channel_features)

        # Score
        seizure_prob = self._score(agg, signals)
        seizure_prob = float(np.clip(seizure_prob, 0.0, 1.0))

        # Update rolling alarm buffer
        if seizure_prob >= self._alarm_threshold:
            self._consecutive_ictal += 1
        else:
            self._consecutive_ictal = 0

        self._alarm_active = self._consecutive_ictal >= self._alarm_trigger_count

        detection = "ictal" if seizure_prob >= self._alarm_threshold else "interictal"

        alert_level = self._get_alert_level(seizure_prob, self._alarm_active)

        # Confidence: distance from 0.5 decision boundary, scaled to [0,1]
        confidence = float(min(1.0, abs(seizure_prob - 0.5) * 2.0))

        return {
            "detection": detection,
            "probability": round(seizure_prob, 4),
            "seizure_probability": round(seizure_prob, 4),
            "alert_level": alert_level,
            "features": agg,
            "confidence": round(confidence, 4),
            "alarm_active": self._alarm_active,
            "consecutive_ictal": self._consecutive_ictal,
            "threshold": self._alarm_threshold,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }

    def get_status(self) -> Dict:
        """Return current alarm state and configuration."""
        return {
            "alarm_active": self._alarm_active,
            "consecutive_ictal": self._consecutive_ictal,
            "threshold": self._alarm_threshold,
            "alarm_trigger_count": self._alarm_trigger_count,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }

    def reset_alarm(self) -> Dict:
        """Reset the consecutive ictal counter and alarm state."""
        self._consecutive_ictal = 0
        self._alarm_active = False
        return {
            "reset": True,
            "consecutive_ictal": 0,
            "alarm_active": False,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _extract_channel_features(self, signal: np.ndarray, fs: float) -> Dict:
        """Extract seizure-relevant features from a single channel."""
        features: Dict = {}

        # Band powers via Welch PSD
        nperseg = min(len(signal), int(fs * 2))
        nperseg = max(nperseg, 4)
        try:
            freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            freqs = np.array([0.0])
            psd = np.array([0.0])

        def _bp(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs <= hi)
            if not np.any(mask):
                return 0.0
            if hasattr(np, "trapezoid"):
                return float(np.trapezoid(psd[mask], freqs[mask]))
            return float(np.trapz(psd[mask], freqs[mask]))

        total_power = _bp(0.5, min(fs / 2 - 1, 250))
        total_power = max(total_power, 1e-12)

        delta = _bp(0.5, 4.0)
        theta = _bp(4.0, 8.0)
        alpha = _bp(8.0, 12.0)
        beta = _bp(12.0, 30.0)
        high_gamma = _bp(60.0, min(fs / 2 - 1, 120.0))

        # HFO: high-frequency oscillations (80–250 Hz); proxy via high_gamma if fs<500
        hfo_hi = min(fs / 2 - 1, 250.0)
        hfo = _bp(80.0, hfo_hi) if hfo_hi > 80.0 else high_gamma
        hfo_ratio = hfo / total_power

        # Delta suppression ratio (delta/total — drops during seizure)
        delta_ratio = delta / total_power

        # Line length: captures amplitude + morphology of seizure waveforms
        if len(signal) > 1:
            ll = float(np.sum(np.abs(np.diff(signal))))
            ll_per_sample = ll / max(len(signal) - 1, 1)
        else:
            ll = 0.0
            ll_per_sample = 0.0

        # Rhythmic spiking: autocorrelation peak at lags corresponding to 2-4 Hz
        rhythmic_score = self._rhythmic_spiking_score(signal, fs)

        features["delta"] = delta
        features["theta"] = theta
        features["alpha"] = alpha
        features["beta"] = beta
        features["high_gamma"] = high_gamma
        features["hfo_ratio"] = round(hfo_ratio, 6)
        features["delta_ratio"] = round(delta_ratio, 6)
        features["line_length"] = round(ll_per_sample, 6)
        features["rhythmic_score"] = round(rhythmic_score, 6)
        features["amplitude_std"] = round(float(np.std(signal)), 4)

        return features

    def _rhythmic_spiking_score(self, signal: np.ndarray, fs: float) -> float:
        """Measure autocorrelation peak at 2-4 Hz lags (spike-wave frequency)."""
        if len(signal) < int(fs * 0.5):
            return 0.0

        sig = signal - np.mean(signal)
        std = np.std(sig)
        if std < 1e-10:
            return 0.0
        sig = sig / std

        # Lags for 2-4 Hz: period = 0.25-0.5 s
        lag_lo = max(1, int(fs * 0.25))  # 4 Hz
        lag_hi = min(len(sig) // 2, int(fs * 0.5))  # 2 Hz

        if lag_lo >= lag_hi:
            return 0.0

        autocorr = np.correlate(sig, sig, mode="full")
        mid = len(autocorr) // 2
        lags = autocorr[mid + lag_lo: mid + lag_hi]

        if len(lags) == 0:
            return 0.0

        peak = float(np.max(lags)) / len(sig)
        return float(np.clip(peak, 0.0, 1.0))

    def _aggregate_features(self, channel_features: List[Dict]) -> Dict:
        """Average features across all channels."""
        if not channel_features:
            return {}

        keys = channel_features[0].keys()
        agg: Dict = {}
        for key in keys:
            values = [ch[key] for ch in channel_features]
            agg[key] = round(float(np.mean(values)), 6)
        agg["n_channels"] = len(channel_features)
        return agg

    def _score(self, features: Dict, signals: np.ndarray) -> float:
        """Compute overall seizure probability from aggregated features."""
        hfo_ratio = features.get("hfo_ratio", 0.0)
        delta_ratio = features.get("delta_ratio", 0.3)
        ll = features.get("line_length", 0.0)
        rhythmic = features.get("rhythmic_score", 0.0)
        amplitude_std = features.get("amplitude_std", 0.0)

        # Feature 1: HFO ratio (higher → more seizure-like)
        # Normal resting HFO ratio ~0.01-0.05; seizure can reach 0.2+
        hfo_score = float(np.clip((hfo_ratio - 0.02) / 0.18, 0.0, 1.0))

        # Feature 2: Delta suppression (normal delta_ratio ~0.3-0.6; drops during seizure)
        # Lower delta_ratio → higher seizure probability
        delta_suppression = float(np.clip(1.0 - delta_ratio / 0.4, 0.0, 1.0))

        # Feature 3: Line length (seizure EEG has much higher line length per sample)
        # Threshold: ~5 µV/sample at rest, >30 during seizure
        ll_score = float(np.clip((ll - 5.0) / 45.0, 0.0, 1.0))

        # Feature 4: Rhythmic spiking at 2-4 Hz
        rhythmic_score = float(np.clip(rhythmic, 0.0, 1.0))

        # Feature 5: Amplitude (seizures produce high-amplitude activity)
        amp_score = float(np.clip((amplitude_std - 30.0) / 70.0, 0.0, 1.0))

        # Cross-channel synchrony (seizures increase sync)
        sync = self._cross_channel_sync(signals)
        sync_score = float(np.clip((sync - 0.5) / 0.5, 0.0, 1.0))

        # Weighted combination
        seizure_prob = (
            0.20 * hfo_score
            + 0.15 * delta_suppression
            + 0.30 * ll_score
            + 0.20 * rhythmic_score
            + 0.10 * amp_score
            + 0.05 * sync_score
        )

        return float(np.clip(seizure_prob, 0.0, 1.0))

    def _cross_channel_sync(self, signals: np.ndarray) -> float:
        """Mean pairwise absolute correlation across channels."""
        n_ch = signals.shape[0]
        if n_ch < 2:
            return 0.5

        corrs: List[float] = []
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                s1, s2 = signals[i], signals[j]
                std1, std2 = np.std(s1), np.std(s2)
                if std1 < 1e-10 or std2 < 1e-10:
                    corrs.append(0.0)
                    continue
                r = float(np.corrcoef(s1, s2)[0, 1])
                corrs.append(abs(r))

        return float(np.mean(corrs)) if corrs else 0.5

    def _get_alert_level(self, prob: float, alarm_active: bool) -> str:
        """Map probability + alarm state to alert level string."""
        if alarm_active and prob >= self._alarm_threshold:
            return "critical"
        if prob >= self._alarm_threshold:
            return "alert"
        if prob >= 0.4:
            return "warning"
        return "none"

    def _empty_predict(self) -> Dict:
        """Return a safe default result for too-short signals."""
        return {
            "detection": "interictal",
            "probability": 0.0,
            "seizure_probability": 0.0,
            "alert_level": "none",
            "features": {},
            "confidence": 0.0,
            "alarm_active": self._alarm_active,
            "consecutive_ictal": self._consecutive_ictal,
            "threshold": self._alarm_threshold,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }
