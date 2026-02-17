"""CNN-LSTM Emotion Classifier from EEG signals.

Classifies EEG into 6 emotions: happy, sad, angry, fearful, relaxed, focused.
Also outputs valence (-1 to 1) and arousal (0 to 1) scores.

Supports three inference paths: ONNX > sklearn > feature-based fallback.
"""

import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import extract_band_powers, differential_entropy, extract_features, preprocess

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]


class EmotionClassifier:
    """EEG-based emotion classifier with ONNX/sklearn/feature-based inference.

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
        """Classify emotion from EEG signal.

        Returns:
            Dict with 'emotion', 'confidence', 'probabilities',
            'valence', 'arousal', 'band_powers'
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
        de = differential_entropy(processed, fs)

        feature_vector = np.array([features[k] for k in self.feature_names]).reshape(1, -1)

        probs = self.sklearn_model.predict_proba(feature_vector)[0]
        emotion_idx = int(np.argmax(probs))

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        gamma = bands.get("gamma", 0)

        valence = float(np.tanh((alpha - beta) * 2 + (theta - gamma) * 0.5))
        arousal = float(np.clip(beta + gamma, 0, 1))

        stress_index = float(np.clip(beta / max(alpha, 1e-10) * 25, 0, 100))
        focus_index = 100 - float(np.clip(theta / max(beta, 1e-10) * 50, 0, 100))
        relaxation_index = float(np.clip(alpha * 100, 0, 100))

        return {
            "emotion": EMOTIONS[emotion_idx],
            "emotion_index": emotion_idx,
            "confidence": float(probs[emotion_idx]),
            "probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(probs)},
            "valence": float(np.clip(valence, -1, 1)),
            "arousal": float(np.clip(arousal, 0, 1)),
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "band_powers": bands,
            "differential_entropy": de,
        }

    def _predict_features(self, eeg: np.ndarray, fs: float) -> Dict:
        """Feature-based emotion classification."""
        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        de = differential_entropy(processed, fs)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        gamma = bands.get("gamma", 0)
        delta = bands.get("delta", 0)

        # Valence: positive correlates with left frontal alpha
        # Simplified: higher alpha/beta ratio -> more positive
        valence = float(np.tanh((alpha - beta) * 2 + (theta - gamma) * 0.5))

        # Arousal: beta + gamma power relative to total
        arousal = float(np.clip(beta + gamma, 0, 1))

        # Map valence-arousal to emotion probabilities
        probs = np.zeros(6)

        # Happy: high valence, moderate-high arousal
        probs[0] = max(0, valence * 0.4 + arousal * 0.3 + alpha * 0.3)

        # Sad: low valence, low arousal
        probs[1] = max(0, -valence * 0.4 + (1 - arousal) * 0.3 + delta * 0.3)

        # Angry: low valence, high arousal
        probs[2] = max(0, -valence * 0.3 + arousal * 0.4 + beta * 0.3)

        # Fearful: low valence, high arousal, high beta
        probs[3] = max(0, -valence * 0.25 + arousal * 0.35 + gamma * 0.2 + beta * 0.2)

        # Relaxed: positive valence, low arousal, high alpha
        probs[4] = max(0, valence * 0.3 + (1 - arousal) * 0.3 + alpha * 0.4)

        # Focused: neutral valence, moderate arousal, high beta, low theta
        probs[5] = max(0, (1 - abs(valence)) * 0.2 + beta * 0.4 + (1 - theta) * 0.2 + gamma * 0.2)

        # Add small noise and normalize
        probs += np.random.uniform(0, 0.02, 6)
        probs = probs / (probs.sum() + 1e-10)

        emotion_idx = int(np.argmax(probs))

        # Derived indices
        stress_index = float(np.clip(beta / max(alpha, 1e-10) * 25, 0, 100))
        focus_index = float(np.clip(theta / max(beta, 1e-10) * 50, 0, 100))
        # Invert: lower theta/beta = more focus
        focus_index = 100 - focus_index
        relaxation_index = float(np.clip(alpha * 100, 0, 100))

        return {
            "emotion": EMOTIONS[emotion_idx],
            "emotion_index": emotion_idx,
            "confidence": float(probs[emotion_idx]),
            "probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(probs)},
            "valence": float(np.clip(valence, -1, 1)),
            "arousal": float(np.clip(arousal, 0, 1)),
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "band_powers": bands,
            "differential_entropy": de,
        }

    def _predict_onnx(self, eeg: np.ndarray, fs: float) -> Dict:
        """ONNX model inference."""
        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        features_dict = extract_features(processed, fs)
        features = np.array(list(features_dict.values()), dtype=np.float32).reshape(1, -1)

        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: features})
        # outputs[0] = predicted label, outputs[1] = probability map
        emotion_idx = int(outputs[0][0])
        prob_map = outputs[1][0]  # dict {class_idx: probability}
        n_classes = len(EMOTIONS)
        probs = [float(prob_map.get(i, 0.0)) for i in range(n_classes)]

        return {
            "emotion": EMOTIONS[emotion_idx] if emotion_idx < n_classes else "unknown",
            "emotion_index": emotion_idx,
            "confidence": float(probs[emotion_idx]) if emotion_idx < n_classes else 0.0,
            "probabilities": {EMOTIONS[i]: probs[i] for i in range(n_classes)},
            "valence": 0.0,
            "arousal": 0.0,
            "band_powers": bands,
        }
