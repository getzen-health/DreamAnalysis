"""Anomaly & Seizure Detection Module.

Uses Isolation Forest for unsupervised anomaly detection on EEG features,
plus specialized detectors for spikes and seizure patterns.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, List, Optional
from collections import deque


class AnomalyDetector:
    """Real-time EEG anomaly and seizure pattern detector.

    Combines unsupervised anomaly detection (Isolation Forest) with
    rule-based spike and seizure pattern recognition.
    """

    def __init__(self):
        self.forest = None
        self.baseline_features: Optional[np.ndarray] = None
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit_baseline(self, features_list: List[Dict[str, float]]) -> Dict:
        """Train anomaly detector on normal EEG features.

        Should be called with the first ~5 minutes of a session.

        Args:
            features_list: List of feature dicts from extract_features().

        Returns:
            Dict with fitting results.
        """
        if len(features_list) < 5:
            return {"fitted": False, "reason": "Need at least 5 samples"}

        keys = list(features_list[0].keys())
        X = np.array([[f.get(k, 0.0) for k in keys] for f in features_list])

        self.baseline_mean = np.mean(X, axis=0)
        self.baseline_std = np.std(X, axis=0) + 1e-10
        self.baseline_features = X

        try:
            from sklearn.ensemble import IsolationForest
            self.forest = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42,
            )
            self.forest.fit(X)
            self.is_fitted = True
        except ImportError:
            # Fallback: use z-score based detection
            self.is_fitted = True

        return {
            "fitted": True,
            "n_samples": len(features_list),
            "n_features": len(keys),
        }

    def detect_anomaly(self, features: Dict[str, float]) -> Dict:
        """Detect if current features are anomalous.

        Returns:
            Dict with is_anomaly, anomaly_score, anomaly_type.
        """
        if not self.is_fitted:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "anomaly_type": "none",
            }

        keys = list(features.keys())
        x = np.array([features.get(k, 0.0) for k in keys]).reshape(1, -1)

        if self.forest is not None:
            score = self.forest.decision_function(x)[0]
            is_anomaly = self.forest.predict(x)[0] == -1
        else:
            # Z-score fallback
            if self.baseline_mean is not None and len(x[0]) == len(self.baseline_mean):
                z_scores = np.abs((x[0] - self.baseline_mean) / self.baseline_std)
                score = -float(np.max(z_scores))  # More negative = more anomalous
                is_anomaly = bool(np.any(z_scores > 3.0))
            else:
                score = 0.0
                is_anomaly = False

        # Determine anomaly type
        anomaly_type = "none"
        if is_anomaly:
            if features.get("band_power_delta", 0) > 0.7:
                anomaly_type = "excessive_delta"
            elif features.get("band_power_gamma", 0) > 0.5:
                anomaly_type = "excessive_gamma"
            elif features.get("hjorth_activity", 0) > 10000:
                anomaly_type = "high_amplitude"
            else:
                anomaly_type = "general_anomaly"

        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(score),
            "anomaly_type": anomaly_type,
        }

    @staticmethod
    def detect_spike(
        signal_data: np.ndarray, fs: float = 256.0, threshold: float = 4.0
    ) -> List[Dict]:
        """Detect sharp transient spikes in EEG.

        Bandpass 20-70 Hz, then find amplitude peaks > threshold * rolling_std.

        Returns:
            List of dicts with 'time', 'amplitude' for each spike.
        """
        # Bandpass 20-70 Hz
        nyq = 0.5 * fs
        if 70.0 / nyq >= 1.0:
            high = 0.95 * nyq
        else:
            high = 70.0

        b, a = scipy_signal.butter(4, [20.0 / nyq, high / nyq], btype="band")
        filtered = scipy_signal.filtfilt(b, a, signal_data)

        # Rolling standard deviation (1-second window)
        window = int(fs)
        rolling_std = np.array([
            np.std(filtered[max(0, i - window):i + 1])
            for i in range(len(filtered))
        ])
        rolling_std = np.maximum(rolling_std, 1e-10)

        # Find peaks above threshold
        z_scores = np.abs(filtered) / rolling_std
        peaks, props = scipy_signal.find_peaks(z_scores, height=threshold, distance=int(fs * 0.1))

        spikes = []
        for i, peak_idx in enumerate(peaks):
            spikes.append({
                "time": float(peak_idx / fs),
                "amplitude": float(filtered[peak_idx]),
                "z_score": float(props["peak_heights"][i]),
            })

        return spikes

    @staticmethod
    def detect_seizure_pattern(
        signal_data: np.ndarray, fs: float = 256.0
    ) -> Dict:
        """Detect seizure-like patterns using multiple indicators.

        Indicators:
        - Sustained rhythmic activity (>3s of >3 Hz rhythmic pattern)
        - Sudden amplitude increase (>3x baseline)
        - Frequency evolution (progressive slowing)

        Returns:
            Dict with seizure_probability, pattern_type, duration_sec.
        """
        n_samples = len(signal_data)
        duration = n_samples / fs

        indicators = []

        # 1. Rhythmic activity detection
        # Check autocorrelation for periodicity
        if n_samples > int(fs * 3):
            segment = signal_data[-int(fs * 5):] if n_samples > int(fs * 5) else signal_data
            autocorr = np.correlate(segment, segment, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)

            # Look for strong periodicity between 3-20 Hz
            for freq in range(3, 21):
                lag = int(fs / freq)
                if lag < len(autocorr) and autocorr[lag] > 0.5:
                    indicators.append(("rhythmic", 0.4))
                    break

        # 2. Amplitude increase detection
        if n_samples > int(fs * 6):
            baseline_rms = np.sqrt(np.mean(signal_data[:int(fs * 3)] ** 2))
            recent_rms = np.sqrt(np.mean(signal_data[-int(fs * 3):] ** 2))
            if baseline_rms > 0 and recent_rms / baseline_rms > 3.0:
                indicators.append(("amplitude_surge", 0.3))

        # 3. Frequency evolution (progressive slowing)
        if n_samples > int(fs * 6):
            # Compare spectral peak of first vs second half
            half = n_samples // 2
            f1, p1 = scipy_signal.welch(signal_data[:half], fs=fs, nperseg=min(half, int(fs)))
            f2, p2 = scipy_signal.welch(signal_data[half:], fs=fs, nperseg=min(n_samples - half, int(fs)))

            peak1 = f1[np.argmax(p1)] if len(p1) > 0 else 0
            peak2 = f2[np.argmax(p2)] if len(p2) > 0 else 0

            if peak1 > 0 and peak2 < peak1 * 0.7:
                indicators.append(("frequency_evolution", 0.3))

        # Combine indicators
        seizure_prob = sum(score for _, score in indicators)
        seizure_prob = min(1.0, seizure_prob)

        pattern_types = [t for t, _ in indicators]
        pattern_type = "+".join(pattern_types) if pattern_types else "none"

        return {
            "seizure_probability": float(seizure_prob),
            "pattern_type": pattern_type,
            "duration_sec": float(duration),
            "indicators": [{"type": t, "score": float(s)} for t, s in indicators],
        }

    @staticmethod
    def get_alert_level(anomaly_score: float, seizure_prob: float) -> str:
        """Determine alert level from anomaly and seizure scores.

        Returns: "normal", "watch", "warning", or "critical"
        """
        if seizure_prob > 0.7:
            return "critical"
        if seizure_prob > 0.4 or anomaly_score < -0.5:
            return "warning"
        if seizure_prob > 0.2 or anomaly_score < -0.3:
            return "watch"
        return "normal"
