"""Drowsiness / Alertness Detector from EEG signals.

Detects 3 alertness states from EEG biomarkers:
  0: alert     — fully awake, high beta, low theta/delta
  1: drowsy    — microsleep onset, increasing theta/alpha, decreasing beta
  2: sleepy    — near-sleep, high delta/theta, very low beta

Scientific basis:
- Drowsiness increases theta (4-8 Hz) and alpha (8-12 Hz) power
- Alertness correlates with beta (12-30 Hz) activity
- Theta/beta ratio is the gold standard drowsiness index
- Alpha attenuation coefficient drops during drowsiness
- Slow eye movements appear as frontal delta increases

Reference: Borghini et al. (2014), Lin et al. (2013), Jap et al. (2009)
"""

import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import (
    extract_band_powers, extract_features, preprocess, compute_hjorth_parameters
)

ALERTNESS_STATES = ["alert", "drowsy", "sleepy"]


class DrowsinessDetector:
    """EEG-based drowsiness/alertness detector."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_type = "feature-based"
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.baseline_alpha = None
        self.baseline_beta = None
        self.baseline_theta = None
        # Running average for trend detection
        self._theta_beta_history = []
        # Microsleep detection: per-user consecutive drowsy epoch counters
        self._microsleep_counters: Dict[str, int] = {}

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        if model_path.endswith(".pkl"):
            try:
                import joblib
                data = joblib.load(model_path)
                self.sklearn_model = data["model"]
                self.feature_names = data["feature_names"]
                self.scaler = data.get("scaler")
                self.model_type = "sklearn"
            except Exception:
                pass

    def calibrate(self, alert_eeg: np.ndarray, fs: float = 256.0):
        """Calibrate with alert-state EEG for personalized baseline."""
        processed = preprocess(alert_eeg, fs)
        bands = extract_band_powers(processed, fs)
        self.baseline_alpha = bands.get("alpha", 0.2)
        self.baseline_beta = bands.get("beta", 0.2)
        self.baseline_theta = bands.get("theta", 0.15)

    def detect_microsleep(self, features: Dict, user_id: str = 'default') -> Dict:
        """Detect microsleep: theta dominates alpha for >= 3 consecutive seconds.

        Microsleep episodes occur when the brain briefly enters a sleep-like
        state while a person is nominally awake. The hallmark EEG signature is
        theta power replacing alpha power for several consecutive seconds.

        Args:
            features: Dict containing at least 'theta' and 'alpha' band powers.
            user_id: Identifier for per-user tracking (different users maintain
                     independent streak counters).

        Returns:
            Dict with:
              - microsleep_alert: bool — True if microsleep detected
              - drowsy_streak_seconds: int — consecutive drowsy time (seconds)
              - theta_alpha_ratio: float — current theta/alpha ratio
              - drowsiness_level: str — 'microsleep', 'drowsy', or 'alert'
        """
        theta = features.get('theta', 0)
        alpha = features.get('alpha', 0)

        theta_alpha_ratio = theta / (alpha + 1e-8)
        is_drowsy_epoch = theta_alpha_ratio > 1.5

        if user_id not in self._microsleep_counters:
            self._microsleep_counters[user_id] = 0

        if is_drowsy_epoch:
            self._microsleep_counters[user_id] += 1
        else:
            self._microsleep_counters[user_id] = 0

        # Microsleep: drowsy for >= 3 consecutive epochs.
        # At a 2-sec hop (4-sec sliding window, 50% overlap), 3 epochs ~ 6 sec.
        # At 1-sec epochs, 3 epochs ~ 3 sec. Either way, 3 is the safety threshold.
        microsleep = self._microsleep_counters[user_id] >= 3

        if microsleep:
            level = 'microsleep'
        elif is_drowsy_epoch:
            level = 'drowsy'
        else:
            level = 'alert'

        return {
            'microsleep_alert': microsleep,
            'drowsy_streak_seconds': self._microsleep_counters[user_id] * 2,
            'theta_alpha_ratio': float(theta_alpha_ratio),
            'drowsiness_level': level,
        }

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Detect drowsiness state from EEG signal.

        Returns:
            Dict with 'state', 'alertness_score' (0-1, 1=fully alert),
            'drowsiness_index', 'confidence', component scores.
        """
        if self.sklearn_model is not None:
            try:
                return self._predict_sklearn(eeg, fs)
            except Exception:
                pass  # fall through to feature-based

        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        hjorth = compute_hjorth_parameters(processed)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        delta = bands.get("delta", 0)
        gamma = bands.get("gamma", 0)

        base_alpha = self.baseline_alpha or 0.2
        base_beta = self.baseline_beta or 0.2
        base_theta = self.baseline_theta or 0.15

        # === Drowsiness Indices ===

        # 1. Theta/Beta Ratio (primary drowsiness marker)
        theta_beta_ratio = theta / (beta + 1e-10)
        self._theta_beta_history.append(theta_beta_ratio)
        if len(self._theta_beta_history) > 30:
            self._theta_beta_history = self._theta_beta_history[-30:]

        # 2. Alpha Attenuation Coefficient
        alpha_change = alpha / (base_alpha + 1e-10)
        # Alpha INCREASES then DECREASES during drowsiness progression
        alpha_attenuation = float(np.clip(1.0 - abs(alpha_change - 1.0), 0, 1))

        # 3. Beta Suppression (alertness drops = beta drops)
        beta_suppression = 1.0 - (beta / (base_beta + 1e-10))
        beta_suppression = float(np.clip(beta_suppression, 0, 1))

        # 4. Delta/Theta Increase (sleepiness)
        slow_wave_increase = (delta + theta) / (alpha + beta + gamma + 1e-10)

        # 5. Theta trend (increasing theta = increasing drowsiness)
        if len(self._theta_beta_history) >= 5:
            recent = np.mean(self._theta_beta_history[-5:])
            older = np.mean(self._theta_beta_history[:5])
            theta_trend = float(np.clip((recent - older) / (older + 1e-10), -1, 1))
        else:
            theta_trend = 0.0

        # === Drowsiness Score (0 = alert, 1 = very drowsy) ===
        drowsiness_index = float(np.clip(
            0.30 * np.tanh(theta_beta_ratio - 1.0) +
            0.25 * beta_suppression +
            0.20 * np.tanh(slow_wave_increase - 0.8) +
            0.15 * max(0, theta_trend) +
            0.10 * (1.0 - alpha_attenuation),
            0, 1
        ))

        # Alertness is inverse of drowsiness
        alertness_score = 1.0 - drowsiness_index

        # === Classify State ===
        if drowsiness_index >= 0.6:
            state_idx = 2  # sleepy
        elif drowsiness_index >= 0.3:
            state_idx = 1  # drowsy
        else:
            state_idx = 0  # alert

        # Confidence
        thresholds = [0.0, 0.3, 0.6, 1.0]
        mid = (thresholds[state_idx] + thresholds[state_idx + 1]) / 2
        dist = abs(drowsiness_index - mid)
        range_size = thresholds[state_idx + 1] - thresholds[state_idx]
        confidence = float(np.clip(1.0 - dist / (range_size / 2 + 1e-10), 0.35, 0.95))

        # === Microsleep Detection ===
        microsleep_info = self.detect_microsleep(bands)

        result = {
            "state": ALERTNESS_STATES[state_idx],
            "state_index": state_idx,
            "alertness_score": round(float(alertness_score), 3),
            "drowsiness_index": round(float(drowsiness_index), 3),
            "confidence": round(confidence, 3),
            "components": {
                "theta_beta_ratio": round(float(theta_beta_ratio), 3),
                "alpha_attenuation": round(float(alpha_attenuation), 3),
                "beta_suppression": round(float(beta_suppression), 3),
                "slow_wave_increase": round(float(slow_wave_increase), 3),
                "theta_trend": round(float(theta_trend), 3),
            },
            "band_powers": bands,
            "microsleep": microsleep_info,
        }

        return result

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        processed = preprocess(eeg, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        fv = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)
        if self.scaler is not None:
            fv = self.scaler.transform(fv)

        probs = self.sklearn_model.predict_proba(fv)[0]
        state_idx = int(np.argmax(probs))
        drowsiness_index = float(probs[1] * 0.5 + probs[2] * 1.0) if len(probs) >= 3 else 0.0

        return {
            "state": ALERTNESS_STATES[state_idx],
            "state_index": state_idx,
            "alertness_score": round(1.0 - drowsiness_index, 3),
            "drowsiness_index": round(drowsiness_index, 3),
            "confidence": round(float(probs[state_idx]), 3),
            "components": {
                "theta_beta_ratio": round(float(bands.get("theta", 0) / (bands.get("beta", 0.01) + 1e-10)), 3),
                "alpha_attenuation": 0.5,
                "beta_suppression": round(float(1.0 - bands.get("beta", 0.2) / 0.2), 3),
                "slow_wave_increase": round(float((bands.get("delta", 0) + bands.get("theta", 0)) / 0.5), 3),
                "theta_trend": 0.0,
            },
            "band_powers": bands,
        }
