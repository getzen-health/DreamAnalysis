"""Cognitive Load / Mental Workload Estimator from EEG signals.

Estimates mental workload level from EEG biomarkers:
  0: low       — resting, minimal cognitive demand
  1: moderate  — engaged, manageable workload
  2: high      — overloaded, near capacity

Scientific basis:
- Frontal theta increases with working memory load (Gevins et al., 1997)
- Parietal alpha decreases with task difficulty (Klimesch, 1999)
- Theta/alpha ratio correlates with cognitive load (Holm et al., 2009)
- Gamma bursts during high-demand processing
- Pupil dilation proxy: beta variability increases under load

Reference: Antonenko et al. (2010), Paas et al. (2003)
"""

import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import (
    extract_band_powers, extract_features, preprocess, compute_hjorth_parameters
)

LOAD_LEVELS = ["low", "moderate", "high"]


class CognitiveLoadEstimator:
    """EEG-based mental workload estimator."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_type = "feature-based"
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.baseline_theta = None
        self.baseline_alpha = None

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

    def calibrate(self, resting_eeg: np.ndarray, fs: float = 256.0):
        """Calibrate with resting EEG baseline."""
        processed = preprocess(resting_eeg, fs)
        bands = extract_band_powers(processed, fs)
        self.baseline_theta = bands.get("theta", 0.15)
        self.baseline_alpha = bands.get("alpha", 0.25)

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Estimate cognitive load from EEG.

        Returns:
            Dict with 'level', 'load_index' (0-1), 'confidence',
            and component scores.
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

        base_theta = self.baseline_theta or 0.15
        base_alpha = self.baseline_alpha or 0.25

        # === Cognitive Load Components ===

        # 1. Frontal Theta Increase (working memory load)
        theta_increase = theta / (base_theta + 1e-10) - 1.0
        working_memory_load = float(np.clip(np.tanh(theta_increase), 0, 1))

        # 2. Parietal Alpha Suppression (task engagement)
        alpha_suppression = 1.0 - (alpha / (base_alpha + 1e-10))
        task_engagement = float(np.clip(alpha_suppression, 0, 1))

        # 3. Theta/Alpha Ratio (cognitive demand index)
        theta_alpha_ratio = theta / (alpha + 1e-10)
        cognitive_demand = float(np.clip(np.tanh(theta_alpha_ratio - 0.5), 0, 1))

        # 4. Beta Activity (active processing)
        processing_intensity = float(np.clip(np.tanh(beta * 5 - 0.5), 0, 1))

        # 5. Gamma Bursts (high-demand processing)
        gamma_activity = float(np.clip(np.tanh(gamma * 15), 0, 1))

        # 6. Complexity (Hjorth — more complex = more processing)
        complexity = hjorth.get("complexity", 1.0) if isinstance(hjorth, dict) else 1.0
        signal_complexity = float(np.clip(np.tanh(complexity - 1.0), 0, 1))

        # === Overall Load Index ===
        load_index = float(np.clip(
            0.25 * working_memory_load +
            0.20 * task_engagement +
            0.20 * cognitive_demand +
            0.15 * processing_intensity +
            0.10 * gamma_activity +
            0.10 * signal_complexity,
            0, 1
        ))

        # === Classify Level ===
        if load_index >= 0.6:
            level_idx = 2  # high
        elif load_index >= 0.3:
            level_idx = 1  # moderate
        else:
            level_idx = 0  # low

        # Confidence
        thresholds = [0.0, 0.3, 0.6, 1.0]
        mid = (thresholds[level_idx] + thresholds[level_idx + 1]) / 2
        dist = abs(load_index - mid)
        range_size = thresholds[level_idx + 1] - thresholds[level_idx]
        confidence = float(np.clip(1.0 - dist / (range_size / 2 + 1e-10), 0.35, 0.95))

        return {
            "level": LOAD_LEVELS[level_idx],
            "level_index": level_idx,
            "load_index": round(load_index, 3),
            "confidence": round(confidence, 3),
            "components": {
                "working_memory_load": round(working_memory_load, 3),
                "task_engagement": round(task_engagement, 3),
                "cognitive_demand": round(cognitive_demand, 3),
                "processing_intensity": round(processing_intensity, 3),
                "gamma_activity": round(gamma_activity, 3),
                "signal_complexity": round(signal_complexity, 3),
            },
            "band_powers": bands,
        }

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        processed = preprocess(eeg, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        fv = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)
        if self.scaler is not None:
            fv = self.scaler.transform(fv)

        probs = self.sklearn_model.predict_proba(fv)[0]
        level_idx = int(np.argmax(probs))
        load_index = float(probs[1] * 0.5 + probs[2] * 1.0) if len(probs) >= 3 else 0.5

        return {
            "level": LOAD_LEVELS[level_idx],
            "level_index": level_idx,
            "load_index": round(load_index, 3),
            "confidence": round(float(probs[level_idx]), 3),
            "components": {
                "working_memory_load": round(float(bands.get("theta", 0.15) / 0.15 - 1), 3),
                "task_engagement": round(float(1.0 - bands.get("alpha", 0.2) / 0.25), 3),
                "cognitive_demand": round(float(bands.get("theta", 0) / (bands.get("alpha", 0.01) + 1e-10)), 3),
                "processing_intensity": round(float(bands.get("beta", 0.15)), 3),
                "gamma_activity": round(float(bands.get("gamma", 0.05)), 3),
                "signal_complexity": 0.5,
            },
            "band_powers": bands,
        }
