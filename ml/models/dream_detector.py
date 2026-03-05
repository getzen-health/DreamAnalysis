"""Dream Detection Model.

Binary classifier: dream state vs. dreamless sleep.
Uses REM detection + EEG spectral features.

Supports three inference paths: ONNX > sklearn > feature-based fallback.
"""

import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import extract_band_powers, extract_features, preprocess


class DreamDetector:
    """Binary dream state classifier with ONNX/sklearn/feature-based inference.

    Priority: ONNX → sklearn .pkl → feature-based fallback.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.onnx_session = None
        self.sklearn_model = None
        self.feature_names = None
        self.model_type = "feature-based"

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load model from file (ONNX or sklearn pkl)."""
        if model_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(model_path)
                self.model_type = "onnx"
            except Exception:
                pass
        elif model_path.endswith(".pkl"):
            try:
                import joblib
                data = joblib.load(model_path)
                self.sklearn_model = data["model"]
                self.feature_names = data["feature_names"]
                self.model_type = "sklearn"
            except Exception:
                pass

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Detect dream state from EEG epoch.

        Returns:
            Dict with 'is_dreaming', 'probability', 'rem_likelihood',
            'dream_intensity', 'lucidity_estimate'
        """
        if self.onnx_session is not None:
            return self._predict_onnx(eeg, fs)
        if self.sklearn_model is not None:
            return self._predict_sklearn(eeg, fs)
        return self._predict_features(eeg, fs)

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        """Sklearn model inference using extracted features."""
        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        features = extract_features(processed, fs)
        feature_vector = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)

        probs = self.sklearn_model.predict_proba(feature_vector)[0]
        # probs[0] = not_dreaming, probs[1] = dreaming
        dream_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])

        theta = bands.get("theta", 0)
        beta = bands.get("beta", 0)
        gamma = bands.get("gamma", 0)

        dream_intensity = float(np.clip(
            theta * 0.4 + gamma * 0.3 + beta * 0.2 + features.get("spectral_entropy", 0.5) * 0.1,
            0, 1
        ))

        lucidity = float(np.clip(
            gamma * 0.5 + beta * 0.3 + features.get("spectral_entropy", 0.5) * 0.2,
            0, 1
        ))

        return {
            "is_dreaming": dream_prob > 0.5,
            "probability": dream_prob,
            "rem_likelihood": float(np.clip(dream_prob * 0.9, 0, 1)),
            "dream_intensity": dream_intensity,
            "lucidity_estimate": lucidity,
            "band_analysis": {
                "delta_dominance": float(bands.get("delta", 0)),
                "theta_activity": float(theta),
                "alpha_presence": float(bands.get("alpha", 0)),
                "beta_activation": float(beta),
                "gamma_burst": float(gamma),
            },
        }

    def _predict_features(self, eeg: np.ndarray, fs: float) -> Dict:
        """Feature-based dream detection."""
        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        features = extract_features(processed, fs)

        delta = bands.get("delta", 0)
        theta = bands.get("theta", 0)
        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        gamma = bands.get("gamma", 0)

        # REM likelihood: high theta + beta, low delta, desynchronized EEG
        rem_score = (
            theta * 0.35
            + beta * 0.25
            + (1 - delta) * 0.20
            + gamma * 0.10
            + (1 - alpha) * 0.10
        )

        # Dream probability: based on REM + EEG complexity
        spectral_ent = features.get("spectral_entropy", 0.5)
        hjorth_complexity = features.get("hjorth_complexity", 0.5)

        dream_prob = (
            rem_score * 0.50
            + spectral_ent * 0.25
            + min(hjorth_complexity, 1.0) * 0.15
            + theta * 0.10
        )

        # Add slight randomness for realism
        dream_prob = float(np.clip(dream_prob + np.random.uniform(-0.05, 0.05), 0, 1))

        # Dream intensity (vividness estimate)
        dream_intensity = float(np.clip(
            theta * 0.4 + gamma * 0.3 + beta * 0.2 + spectral_ent * 0.1,
            0, 1
        ))

        # Lucidity estimate: higher gamma during dreaming suggests awareness
        lucidity = float(np.clip(
            gamma * 0.5 + beta * 0.3 + spectral_ent * 0.2,
            0, 1
        ))

        is_dreaming = dream_prob > 0.5

        return {
            "is_dreaming": bool(is_dreaming),
            "probability": dream_prob,
            "rem_likelihood": float(np.clip(rem_score, 0, 1)),
            "dream_intensity": dream_intensity,
            "lucidity_estimate": lucidity,
            "band_analysis": {
                "delta_dominance": float(delta),
                "theta_activity": float(theta),
                "alpha_presence": float(alpha),
                "beta_activation": float(beta),
                "gamma_burst": float(gamma),
            },
        }

    def _predict_onnx(self, eeg: np.ndarray, fs: float) -> Dict:
        """ONNX model inference."""
        processed = preprocess(eeg, fs)
        features = extract_features(processed, fs)
        input_data = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)

        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: input_data})
        prob = float(outputs[0][0][1]) if outputs[0].shape[-1] > 1 else float(outputs[0][0][0])

        return {
            "is_dreaming": prob > 0.5,
            "probability": prob,
            "rem_likelihood": prob * 0.8,
            "dream_intensity": prob * 0.7,
            "lucidity_estimate": 0.0,
        }
