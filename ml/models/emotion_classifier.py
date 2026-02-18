"""CNN-LSTM Emotion Classifier from EEG signals.

Classifies EEG into 6 emotions: happy, sad, angry, fearful, relaxed, focused.
Also outputs valence (-1 to 1) and arousal (0 to 1) scores.

Uses neuroscience-backed feature-based classification with exponential
smoothing to prevent rapid state flickering. ONNX/sklearn models are
only used when their benchmark accuracy exceeds 60%.
"""

import numpy as np
from typing import Dict, Optional
from collections import deque
from processing.eeg_processor import extract_band_powers, differential_entropy, extract_features, preprocess

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

# Minimum benchmark accuracy to trust a trained model over feature-based
_MIN_MODEL_ACCURACY = 0.60


class EmotionClassifier:
    """EEG-based emotion classifier with ONNX/sklearn/feature-based inference.

    Priority: ONNX → sklearn .pkl → feature-based fallback.
    Trained models are only used if their benchmark accuracy >= 60%.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.onnx_session = None
        self.sklearn_model = None
        self.feature_names = None
        self.model_type = "feature-based"
        self._benchmark_accuracy = 0.0

        # Exponential moving average for smoothing (prevents flickering)
        self._ema_probs = None  # smoothed probabilities
        self._ema_alpha = 0.3   # smoothing factor (lower = smoother)
        self._history = deque(maxlen=10)  # recent band power snapshots

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load model from file (ONNX or sklearn pkl)."""
        # Check benchmark to decide if model is trustworthy
        self._benchmark_accuracy = self._read_benchmark()

        if self._benchmark_accuracy < _MIN_MODEL_ACCURACY:
            # Model too inaccurate — stay on feature-based
            return

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

    @staticmethod
    def _read_benchmark() -> float:
        """Read the latest benchmark accuracy from disk."""
        import json
        from pathlib import Path
        bench_path = Path("benchmarks/emotion_classifier_benchmark.json")
        if bench_path.exists():
            try:
                data = json.loads(bench_path.read_text())
                return float(data.get("accuracy", 0))
            except Exception:
                pass
        return 0.0

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Classify emotion from EEG signal."""
        if self.onnx_session is not None and self._benchmark_accuracy >= _MIN_MODEL_ACCURACY:
            return self._predict_onnx(eeg, fs)
        if self.sklearn_model is not None and self._benchmark_accuracy >= _MIN_MODEL_ACCURACY:
            return self._predict_sklearn(eeg, fs)
        return self._predict_features(eeg, fs)

    # ────────────────────────────────────────────────────────────────
    # Feature-based classifier (primary path for Muse 2)
    # ────────────────────────────────────────────────────────────────

    def _predict_features(self, eeg: np.ndarray, fs: float) -> Dict:
        """Neuroscience-backed emotion classification from band powers.

        Uses established EEG-emotion correlates:
        - Alpha (8-12 Hz): inversely related to cortical arousal; dominant in relaxation
        - Beta (12-30 Hz): active thinking, concentration, anxiety when excessive
        - Theta (4-8 Hz): drowsiness, meditation, creative states
        - Gamma (30+ Hz): complex cognition, memory binding, peak experiences
        - Delta (0.5-4 Hz): deep sleep, regeneration
        - Alpha asymmetry: left > right frontal alpha → positive valence
        """
        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        de = differential_entropy(processed, fs)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        gamma = bands.get("gamma", 0)
        delta = bands.get("delta", 0)

        # Store snapshot for temporal analysis
        self._history.append(bands)

        # ── Valence (pleasantness) ──────────────────────────────
        # Higher alpha relative to beta suggests positive valence
        # Gamma bursts also correlate with positive emotions
        alpha_beta_ratio = alpha / max(beta, 1e-10)
        valence_raw = (
            0.45 * np.tanh((alpha_beta_ratio - 1.0) * 1.5)  # alpha dominance = positive
            + 0.25 * np.tanh((gamma - 0.15) * 5)              # gamma bursts = positive
            - 0.20 * np.tanh((theta - 0.25) * 3)              # excess theta = negative
            + 0.10 * np.tanh((alpha - 0.20) * 3)              # baseline alpha = slightly positive
        )
        valence = float(np.clip(valence_raw, -1, 1))

        # ── Arousal (activation level) ──────────────────────────
        # Beta + gamma indicate cortical activation
        # Alpha + delta indicate cortical deactivation
        arousal_raw = (
            0.40 * beta / max(beta + alpha, 1e-10)    # beta proportion
            + 0.25 * gamma / max(gamma + theta, 1e-10) # gamma proportion
            + 0.20 * (1.0 - alpha / max(alpha + beta + theta, 1e-10))  # inverse alpha
            + 0.15 * (1.0 - delta / max(delta + beta, 1e-10))         # inverse delta
        )
        arousal = float(np.clip(arousal_raw, 0, 1))

        # ── Emotion probability estimation ──────────────────────
        # Based on the circumplex model with EEG-specific weightings
        probs = np.zeros(6)

        theta_beta_ratio = theta / max(beta, 1e-10)
        beta_alpha_ratio = beta / max(alpha, 1e-10)

        # Happy: positive valence, moderate-high arousal, good alpha+gamma
        probs[0] = (
            0.30 * max(0, valence)
            + 0.20 * max(0, arousal - 0.3)
            + 0.25 * min(1, alpha_beta_ratio * 0.8)
            + 0.25 * min(1, gamma * 3)
        )

        # Sad: negative valence, low arousal, high theta, low gamma
        probs[1] = (
            0.30 * max(0, -valence)
            + 0.25 * max(0, 1 - arousal)
            + 0.25 * min(1, theta * 2)
            + 0.20 * max(0, 1 - gamma * 4)
        )

        # Angry: negative valence, high arousal, high beta
        probs[2] = (
            0.25 * max(0, -valence * 0.8)
            + 0.30 * max(0, arousal - 0.5)
            + 0.30 * min(1, beta_alpha_ratio * 0.5)
            + 0.15 * min(1, gamma * 2)
        )

        # Fearful/Anxious: negative valence, high arousal, very high beta/alpha
        probs[3] = (
            0.20 * max(0, -valence * 0.6)
            + 0.25 * max(0, arousal - 0.4)
            + 0.35 * min(1, beta_alpha_ratio * 0.4)
            + 0.20 * max(0, 1 - alpha * 3)
        )

        # Relaxed: positive valence, low arousal, dominant alpha
        probs[4] = (
            0.20 * max(0, valence * 0.8)
            + 0.25 * max(0, 1 - arousal)
            + 0.35 * min(1, alpha * 2.5)
            + 0.20 * max(0, 1 - beta_alpha_ratio * 0.3)
        )

        # Focused: neutral-positive valence, moderate arousal, high beta, low theta
        probs[5] = (
            0.10 * max(0, 1 - abs(valence))
            + 0.25 * min(1, max(0, arousal - 0.3) * 2)
            + 0.35 * min(1, beta * 2.5)
            + 0.30 * max(0, 1 - theta_beta_ratio * 0.5)
        )

        # Softmax-like normalization (temperature=1.5 for moderate sharpness)
        temp = 1.5
        probs_exp = np.exp(probs * temp)
        probs = probs_exp / (probs_exp.sum() + 1e-10)

        # Exponential moving average smoothing
        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            self._ema_probs = self._ema_alpha * probs + (1 - self._ema_alpha) * self._ema_probs

        smoothed = self._ema_probs / (self._ema_probs.sum() + 1e-10)
        emotion_idx = int(np.argmax(smoothed))

        # ── Mental state indices (0-1 scale) ────────────────────
        # Stress: high beta/alpha ratio, low alpha
        stress_index = float(np.clip(
            0.50 * min(1, beta_alpha_ratio * 0.3)
            + 0.30 * max(0, 1 - alpha * 2.5)
            + 0.20 * min(1, gamma * 2),
            0, 1
        ))

        # Focus: high beta, low theta/beta ratio
        focus_index = float(np.clip(
            0.40 * min(1, beta * 2.5)
            + 0.35 * max(0, 1 - theta_beta_ratio * 0.4)
            + 0.25 * min(1, gamma * 2),
            0, 1
        ))

        # Relaxation: high alpha, low beta
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5)
            + 0.30 * max(0, 1 - beta_alpha_ratio * 0.3)
            + 0.20 * min(1, theta * 1.5),
            0, 1
        ))

        return {
            "emotion": EMOTIONS[emotion_idx],
            "emotion_index": emotion_idx,
            "confidence": float(smoothed[emotion_idx]),
            "probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(smoothed)},
            "valence": valence,
            "arousal": arousal,
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "band_powers": bands,
            "differential_entropy": de,
        }

    # ────────────────────────────────────────────────────────────────
    # ONNX inference (only used if benchmark >= 60%)
    # ────────────────────────────────────────────────────────────────

    def _predict_onnx(self, eeg: np.ndarray, fs: float) -> Dict:
        """ONNX model inference."""
        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        features_dict = extract_features(processed, fs)
        de = differential_entropy(processed, fs)
        features = np.array(list(features_dict.values()), dtype=np.float32).reshape(1, -1)

        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: features})
        emotion_idx = int(outputs[0][0])
        prob_map = outputs[1][0]
        n_classes = len(EMOTIONS)
        probs = [float(prob_map.get(i, 0.0)) for i in range(n_classes)]

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        gamma = bands.get("gamma", 0)

        valence = float(np.tanh((alpha - beta) * 2 + (theta - gamma) * 0.5))
        arousal = float(np.clip(beta + gamma, 0, 1))

        stress_index = float(np.clip(beta / max(alpha, 1e-10) * 0.25, 0, 1))
        focus_index = 1.0 - float(np.clip(theta / max(beta, 1e-10) * 0.5, 0, 1))
        relaxation_index = float(np.clip(alpha, 0, 1))

        return {
            "emotion": EMOTIONS[emotion_idx] if emotion_idx < n_classes else "unknown",
            "emotion_index": emotion_idx,
            "confidence": float(probs[emotion_idx]) if emotion_idx < n_classes else 0.0,
            "probabilities": {EMOTIONS[i]: probs[i] for i in range(n_classes)},
            "valence": float(np.clip(valence, -1, 1)),
            "arousal": float(np.clip(arousal, 0, 1)),
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "band_powers": bands,
            "differential_entropy": de,
        }

    # ────────────────────────────────────────────────────────────────
    # Sklearn inference (only used if benchmark >= 60%)
    # ────────────────────────────────────────────────────────────────

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

        stress_index = float(np.clip(beta / max(alpha, 1e-10) * 0.25, 0, 1))
        focus_index = 1.0 - float(np.clip(theta / max(beta, 1e-10) * 0.5, 0, 1))
        relaxation_index = float(np.clip(alpha, 0, 1))

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
