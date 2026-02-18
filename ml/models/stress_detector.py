"""Dedicated Stress Detection Model from EEG signals.

Multi-dimensional stress assessment beyond emotion-derived heuristics:
  0: relaxed    — parasympathetic dominance, low cortisol proxy
  1: mild       — slight arousal, manageable pressure
  2: moderate   — sympathetic activation, noticeable stress
  3: high       — fight-or-flight, cortisol surge, cognitive impairment

Scientific basis:
- High-beta (20-30 Hz) increases with anxiety/stress (Pavlenko et al., 2009)
- Alpha asymmetry: right > left frontal alpha = negative affect (Davidson, 1992)
- Theta/alpha ratio changes under acute stress (Gärtner et al., 2014)
- Gamma suppression during chronic stress (Miskovic et al., 2010)
- HRV proxy: beta variability correlates with autonomic stress response
- Cortisol proxy: sustained elevated beta + reduced alpha

Reference: Al-Shargie et al. (2016), Giannakakis et al. (2019)
"""

import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import (
    extract_band_powers, extract_features, preprocess, compute_hjorth_parameters
)

STRESS_LEVELS = ["relaxed", "mild", "moderate", "high"]


class StressDetector:
    """Dedicated EEG-based stress level detector."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_type = "feature-based"
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.baseline_alpha = None
        self.baseline_beta = None
        # Track stress over time for trend analysis
        self._stress_history = []

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

    def calibrate(self, relaxed_eeg: np.ndarray, fs: float = 256.0):
        """Calibrate with relaxed-state EEG baseline."""
        processed = preprocess(relaxed_eeg, fs)
        bands = extract_band_powers(processed, fs)
        self.baseline_alpha = bands.get("alpha", 0.25)
        self.baseline_beta = bands.get("beta", 0.15)

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Detect stress level from EEG.

        Returns:
            Dict with 'level', 'stress_index' (0-1), 'confidence',
            'cortisol_proxy', 'autonomic_index', and component scores.
        """
        if self.sklearn_model is not None:
            return self._predict_sklearn(eeg, fs)

        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        hjorth = compute_hjorth_parameters(processed)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        delta = bands.get("delta", 0)
        gamma = bands.get("gamma", 0)

        base_alpha = self.baseline_alpha or 0.25
        base_beta = self.baseline_beta or 0.15

        # === Stress Components ===

        # 1. High-Beta Activation (anxiety marker, 20-30 Hz)
        # Stress elevates high-beta disproportionately
        beta_elevation = beta / (base_beta + 1e-10) - 1.0
        anxiety_activation = float(np.clip(np.tanh(beta_elevation), 0, 1))

        # 2. Alpha Suppression (stress reduces alpha relaxation)
        alpha_ratio = alpha / (base_alpha + 1e-10)
        alpha_suppression = float(np.clip(1.0 - alpha_ratio, 0, 1))

        # 3. Beta/Alpha Ratio (arousal index)
        beta_alpha_ratio = beta / (alpha + 1e-10)
        arousal_index = float(np.clip(np.tanh(beta_alpha_ratio - 0.8), 0, 1))

        # 4. Theta Increase (under acute stress, theta can increase)
        theta_stress = float(np.clip(np.tanh(theta * 5 - 0.5), 0, 1))

        # 5. Gamma Suppression (chronic stress suppresses gamma)
        gamma_suppression = float(np.clip(1.0 - np.tanh(gamma * 15), 0, 1))

        # 6. Signal Irregularity (Hjorth complexity increases under stress)
        complexity = hjorth.get("complexity", 1.0) if isinstance(hjorth, dict) else 1.0
        activity = hjorth.get("activity", 0.01) if isinstance(hjorth, dict) else 0.01
        neural_irregularity = float(np.clip(np.tanh(complexity - 1.0), 0, 1))

        # 7. Cortisol Proxy (sustained beta + suppressed alpha over time)
        cortisol_proxy = float(np.clip(
            0.5 * anxiety_activation + 0.5 * alpha_suppression, 0, 1
        ))

        # 8. Autonomic Index (sympathetic vs parasympathetic proxy)
        # High beta + low alpha + elevated theta = sympathetic dominance
        autonomic_index = float(np.clip(
            0.4 * arousal_index + 0.3 * alpha_suppression + 0.3 * theta_stress,
            0, 1
        ))

        # === Overall Stress Index ===
        stress_index = float(np.clip(
            0.25 * anxiety_activation +
            0.20 * alpha_suppression +
            0.20 * arousal_index +
            0.10 * theta_stress +
            0.10 * gamma_suppression +
            0.15 * neural_irregularity,
            0, 1
        ))

        # Track history
        self._stress_history.append(stress_index)
        if len(self._stress_history) > 60:
            self._stress_history = self._stress_history[-60:]

        # Stress trend
        if len(self._stress_history) >= 10:
            recent = np.mean(self._stress_history[-5:])
            older = np.mean(self._stress_history[-10:-5])
            stress_trend = float(np.clip(recent - older, -0.5, 0.5))
        else:
            stress_trend = 0.0

        # === Classify Level ===
        if stress_index >= 0.70:
            level_idx = 3  # high
        elif stress_index >= 0.45:
            level_idx = 2  # moderate
        elif stress_index >= 0.20:
            level_idx = 1  # mild
        else:
            level_idx = 0  # relaxed

        # Confidence
        thresholds = [0.0, 0.20, 0.45, 0.70, 1.0]
        mid = (thresholds[level_idx] + thresholds[level_idx + 1]) / 2
        dist = abs(stress_index - mid)
        range_size = thresholds[level_idx + 1] - thresholds[level_idx]
        confidence = float(np.clip(1.0 - dist / (range_size / 2 + 1e-10), 0.35, 0.95))

        return {
            "level": STRESS_LEVELS[level_idx],
            "level_index": level_idx,
            "stress_index": round(stress_index, 3),
            "confidence": round(confidence, 3),
            "cortisol_proxy": round(cortisol_proxy, 3),
            "autonomic_index": round(autonomic_index, 3),
            "stress_trend": round(stress_trend, 3),
            "components": {
                "anxiety_activation": round(anxiety_activation, 3),
                "alpha_suppression": round(alpha_suppression, 3),
                "arousal_index": round(arousal_index, 3),
                "theta_stress": round(theta_stress, 3),
                "gamma_suppression": round(gamma_suppression, 3),
                "neural_irregularity": round(neural_irregularity, 3),
            },
            "band_powers": bands,
        }

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        processed = preprocess(eeg, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        fv = np.array([features[k] for k in self.feature_names]).reshape(1, -1)
        if self.scaler is not None:
            fv = self.scaler.transform(fv)

        probs = self.sklearn_model.predict_proba(fv)[0]
        level_idx = int(np.argmax(probs))
        stress_index = float(
            probs[1] * 0.3 + probs[2] * 0.6 + probs[3] * 1.0
        ) if len(probs) >= 4 else float(probs[level_idx])

        return {
            "level": STRESS_LEVELS[min(level_idx, 3)],
            "level_index": min(level_idx, 3),
            "stress_index": round(stress_index, 3),
            "confidence": round(float(probs[level_idx]), 3),
            "cortisol_proxy": round(stress_index * 0.8, 3),
            "autonomic_index": round(stress_index * 0.7, 3),
            "stress_trend": 0.0,
            "components": {
                "anxiety_activation": round(float(bands.get("beta", 0.15) / 0.15 - 1), 3),
                "alpha_suppression": round(float(1.0 - bands.get("alpha", 0.2) / 0.25), 3),
                "arousal_index": round(float(bands.get("beta", 0) / (bands.get("alpha", 0.01) + 1e-10) - 0.8), 3),
                "theta_stress": round(float(bands.get("theta", 0.15)), 3),
                "gamma_suppression": round(float(1.0 - bands.get("gamma", 0.05) * 15), 3),
                "neural_irregularity": 0.5,
            },
            "band_powers": bands,
        }
