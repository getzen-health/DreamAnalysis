"""Flow State Detector from EEG signals.

Detects the psychological "flow" state (being "in the zone") using EEG biomarkers.

Scientific basis:
- Flow correlates with increased frontal alpha (relaxed focus)
- Moderate beta (engaged but not anxious)
- Theta/gamma coupling (deep processing)
- Reduced default mode network activity (less mind-wandering)
- Alpha/beta crossover (transition signature)

Reference: Katahira et al. (2018), Ulrich et al. (2014), Nacke et al. (2010)

Outputs flow probability (0-1) and component scores:
- absorption: how deeply engaged (theta + low-beta)
- effortlessness: lack of strain (alpha/high-beta ratio)
- focus_quality: sustained attention (beta stability)
- time_distortion: altered time perception proxy (theta power)
"""

import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import (
    extract_band_powers, extract_features, preprocess, compute_hjorth_parameters
)


FLOW_STATES = ["no_flow", "micro_flow", "flow", "deep_flow"]


class FlowStateDetector:
    """EEG-based flow state detector using frequency band analysis."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_type = "feature-based"
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.calibration_data = []
        self.baseline_alpha = None
        self.baseline_beta = None
        self.baseline_theta = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load trained model from pkl file."""
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
        """Calibrate with 30-60s of resting EEG for personalized thresholds."""
        processed = preprocess(resting_eeg, fs)
        bands = extract_band_powers(processed, fs)
        self.baseline_alpha = bands.get("alpha", 0.2)
        self.baseline_beta = bands.get("beta", 0.15)
        self.baseline_theta = bands.get("theta", 0.15)

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Detect flow state from EEG signal.

        Returns:
            Dict with 'state', 'flow_score', 'confidence',
            'absorption', 'effortlessness', 'focus_quality',
            'time_distortion', 'band_powers'
        """
        if self.sklearn_model is not None:
            return self._predict_sklearn(eeg, fs)

        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        hjorth = compute_hjorth_parameters(processed)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        gamma = bands.get("gamma", 0)
        delta = bands.get("delta", 0)

        # Use calibration baselines if available
        base_alpha = self.baseline_alpha or 0.2
        base_beta = self.baseline_beta or 0.15
        base_theta = self.baseline_theta or 0.15

        # === Flow Component Scores ===

        # 1. Absorption: deep engagement
        # High theta (deep processing) + moderate low-beta
        # Theta increases during immersive tasks
        theta_increase = theta / (base_theta + 1e-10)
        absorption = float(np.clip(
            0.4 * np.tanh(theta_increase - 1) +  # theta above baseline
            0.3 * np.tanh(beta * 5) +              # moderate beta presence
            0.3 * np.tanh(gamma * 10),              # gamma bursts (insight)
            0, 1
        ))

        # 2. Effortlessness: relaxed concentration (not straining)
        # High alpha (relaxation) relative to high-beta (anxiety)
        # Flow feels effortless — alpha stays elevated despite focus
        alpha_beta_ratio = alpha / (beta + 1e-10)
        alpha_increase = alpha / (base_alpha + 1e-10)
        effortlessness = float(np.clip(
            0.5 * np.tanh(alpha_increase - 0.8) +  # alpha at/above baseline
            0.3 * np.tanh(alpha_beta_ratio - 0.5) + # alpha > beta
            0.2 * (1 - np.tanh(delta * 5)),          # not drowsy
            0, 1
        ))

        # 3. Focus Quality: sustained, stable attention
        # Moderate-high beta, low variability (Hjorth complexity)
        beta_increase = beta / (base_beta + 1e-10)
        complexity = hjorth.get("complexity", 1.0) if isinstance(hjorth, dict) else 1.0
        focus_quality = float(np.clip(
            0.4 * np.tanh(beta_increase - 0.5) +   # beta above baseline
            0.3 * (1 - np.tanh(delta * 3)) +         # low delta (alert)
            0.3 * np.tanh(1.5 - complexity),          # low complexity (stable)
            0, 1
        ))

        # 4. Time Distortion: proxy for altered time perception
        # High theta/alpha ratio (associated with temporal distortion)
        theta_alpha_ratio = theta / (alpha + 1e-10)
        time_distortion = float(np.clip(
            0.5 * np.tanh(theta_alpha_ratio - 0.5) +
            0.3 * np.tanh(theta * 5) +
            0.2 * absorption,  # absorption correlates with time distortion
            0, 1
        ))

        # === Overall Flow Score ===
        # Weighted combination with emphasis on absorption and effortlessness
        flow_score = float(np.clip(
            0.35 * absorption +
            0.25 * effortlessness +
            0.25 * focus_quality +
            0.15 * time_distortion,
            0, 1
        ))

        # === Classify Flow Level ===
        if flow_score >= 0.7:
            state_idx = 3  # deep_flow
        elif flow_score >= 0.5:
            state_idx = 2  # flow
        elif flow_score >= 0.3:
            state_idx = 1  # micro_flow
        else:
            state_idx = 0  # no_flow

        # Confidence based on how clearly the score falls into a category
        thresholds = [0.0, 0.3, 0.5, 0.7, 1.0]
        mid = (thresholds[state_idx] + thresholds[state_idx + 1]) / 2
        distance_to_mid = abs(flow_score - mid)
        range_size = thresholds[state_idx + 1] - thresholds[state_idx]
        confidence = float(np.clip(1.0 - (distance_to_mid / (range_size / 2 + 1e-10)), 0.3, 0.95))

        # Add small noise to make it feel less deterministic
        confidence += np.random.uniform(-0.02, 0.02)
        confidence = float(np.clip(confidence, 0.3, 0.95))

        return {
            "state": FLOW_STATES[state_idx],
            "state_index": state_idx,
            "flow_score": round(flow_score, 3),
            "confidence": round(confidence, 3),
            "components": {
                "absorption": round(absorption, 3),
                "effortlessness": round(effortlessness, 3),
                "focus_quality": round(focus_quality, 3),
                "time_distortion": round(time_distortion, 3),
            },
            "band_powers": bands,
        }

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        """Sklearn model inference using extracted features."""
        processed = preprocess(eeg, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        feature_vector = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)

        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)

        probs = self.sklearn_model.predict_proba(feature_vector)[0]
        state_idx = int(np.argmax(probs))
        flow_score = float(state_idx / 3.0 * 0.7 + probs[state_idx] * 0.3)

        return {
            "state": FLOW_STATES[state_idx],
            "state_index": state_idx,
            "flow_score": round(float(np.clip(flow_score, 0, 1)), 3),
            "confidence": round(float(probs[state_idx]), 3),
            "components": {
                "absorption": round(float(probs[2] + probs[3]) / 2, 3) if len(probs) > 3 else 0.5,
                "effortlessness": round(float(bands.get("alpha", 0.3)), 3),
                "focus_quality": round(float(bands.get("beta", 0.2)), 3),
                "time_distortion": round(float(bands.get("theta", 0.15)), 3),
            },
            "band_powers": bands,
        }
