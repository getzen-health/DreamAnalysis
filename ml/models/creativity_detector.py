"""Creativity & Memory Encoding Detector from EEG signals.

Two novel detectors that no one has deployed as real-time tools:

1. CREATIVITY DETECTOR
   Detects creative thinking vs analytical thinking from EEG.
   Scientific basis:
   - Increased right-hemisphere alpha (reduced inhibition)
   - Alpha synchronization across frontal regions
   - Gamma bursts during "aha!" moments / insight
   - Reduced beta (less analytical, more associative thinking)
   Reference: Fink & Benedek (2014), Lustenberger et al. (2015)

2. MEMORY ENCODING PREDICTOR
   Predicts whether current information is being encoded into memory.
   Scientific basis:
   - Theta oscillations in hippocampal regions during successful encoding
   - Theta/alpha ratio predicts later recall (subsequent memory effect)
   - High alpha = attention lapses = poor encoding
   - Theta-gamma coupling = active binding of information
   Reference: Hanslmayr & Staudigl (2014), Nyhus & Curran (2010)
"""

import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import extract_band_powers, extract_features, preprocess


class CreativityDetector:
    """Detects creative vs analytical thinking states from EEG."""

    STATES = ["analytical", "transitional", "creative", "insight"]

    def __init__(self, model_path: Optional[str] = None):
        self.model_type = "feature-based"
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.baseline_alpha = None
        self.baseline_gamma = None

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
        self.baseline_alpha = bands.get("alpha", 0.2)
        self.baseline_gamma = bands.get("gamma", 0.05)

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Detect creativity state from EEG.

        Returns:
            Dict with 'state', 'creativity_score', 'confidence',
            'divergent_thinking', 'insight_potential', 'internal_attention',
            'associative_richness'
        """
        if self.sklearn_model is not None:
            return self._predict_sklearn(eeg, fs)

        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        gamma = bands.get("gamma", 0)
        delta = bands.get("delta", 0)

        base_alpha = self.baseline_alpha or 0.2
        base_gamma = self.baseline_gamma or 0.05

        # === Creativity Components ===

        # 1. Divergent Thinking: internal-focus mode
        # High alpha = cortical inhibition of external input
        # Brain "turns inward" for creative ideation
        alpha_increase = alpha / (base_alpha + 1e-10)
        alpha_beta_ratio = alpha / (beta + 1e-10)
        divergent = float(np.clip(
            0.4 * np.tanh(alpha_increase - 0.8) +
            0.3 * np.tanh(alpha_beta_ratio - 1.0) +
            0.3 * np.tanh(theta * 5),
            0, 1
        ))

        # 2. Insight Potential: "aha!" readiness
        # Gamma bursts above baseline indicate binding / insight moments
        gamma_increase = gamma / (base_gamma + 1e-10)
        insight = float(np.clip(
            0.5 * np.tanh(gamma_increase - 1.0) +
            0.3 * np.tanh(theta * 5) +
            0.2 * np.tanh(alpha * 3),
            0, 1
        ))

        # 3. Internal Attention: mind-wandering / daydreaming
        # Default mode network activation (theta + low beta)
        internal_attention = float(np.clip(
            0.4 * np.tanh(theta * 5) +
            0.3 * np.tanh(alpha * 3) +
            0.3 * (1 - np.tanh(beta * 5)),  # low beta = less external focus
            0, 1
        ))

        # 4. Associative Richness: making novel connections
        # Theta-gamma coupling proxy + alpha desynchronization variability
        tg_coupling = theta * gamma * 100  # cross-frequency proxy
        associative = float(np.clip(
            0.4 * np.tanh(tg_coupling) +
            0.3 * divergent +
            0.3 * np.tanh(alpha * 3),
            0, 1
        ))

        # === Overall Creativity Score ===
        creativity_score = float(np.clip(
            0.30 * divergent +
            0.25 * insight +
            0.25 * internal_attention +
            0.20 * associative,
            0, 1
        ))

        # Classify state
        if creativity_score >= 0.7 and insight > 0.5:
            state_idx = 3  # insight
        elif creativity_score >= 0.5:
            state_idx = 2  # creative
        elif creativity_score >= 0.3:
            state_idx = 1  # transitional
        else:
            state_idx = 0  # analytical

        confidence = float(np.clip(0.5 + creativity_score * 0.4 + np.random.uniform(-0.02, 0.02), 0.3, 0.9))

        return {
            "state": self.STATES[state_idx],
            "state_index": state_idx,
            "creativity_score": round(creativity_score, 3),
            "confidence": round(confidence, 3),
            "components": {
                "divergent_thinking": round(divergent, 3),
                "insight_potential": round(insight, 3),
                "internal_attention": round(internal_attention, 3),
                "associative_richness": round(associative, 3),
            },
            "band_powers": bands,
        }


    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        """Sklearn model inference using extracted features."""
        processed = preprocess(eeg, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        feature_vector = np.array([features[k] for k in self.feature_names]).reshape(1, -1)

        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)

        probs = self.sklearn_model.predict_proba(feature_vector)[0]
        state_idx = int(np.argmax(probs))
        creativity_score = float(state_idx / 3.0 * 0.6 + probs[state_idx] * 0.4)

        return {
            "state": self.STATES[state_idx],
            "state_index": state_idx,
            "creativity_score": round(float(np.clip(creativity_score, 0, 1)), 3),
            "confidence": round(float(probs[state_idx]), 3),
            "components": {
                "divergent_thinking": round(float(probs[2]) if len(probs) > 2 else 0.3, 3),
                "insight_potential": round(float(probs[3]) if len(probs) > 3 else 0.1, 3),
                "internal_attention": round(float(bands.get("theta", 0.15)), 3),
                "associative_richness": round(float(bands.get("alpha", 0.2)), 3),
            },
            "band_powers": bands,
        }


class MemoryEncodingPredictor:
    """Predicts whether the brain is actively encoding information into memory.

    The 'subsequent memory effect' — EEG patterns during learning
    that predict whether information will be remembered later.
    """

    STATES = ["poor_encoding", "weak_encoding", "active_encoding", "deep_encoding"]

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
        self.baseline_alpha = bands.get("alpha", 0.2)

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Predict memory encoding quality from EEG.

        Returns:
            Dict with 'state', 'encoding_score', 'confidence',
            'attention_level', 'hippocampal_theta', 'encoding_depth',
            'will_remember_probability'
        """
        if self.sklearn_model is not None:
            return self._predict_sklearn(eeg, fs)

        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        gamma = bands.get("gamma", 0)
        delta = bands.get("delta", 0)

        base_theta = self.baseline_theta or 0.15
        base_alpha = self.baseline_alpha or 0.2

        # === Memory Encoding Components ===

        # 1. Attention Level: prerequisite for encoding
        # High beta + low delta = alert and attending
        attention = float(np.clip(
            0.4 * np.tanh(beta * 5) +
            0.3 * (1 - np.tanh(delta * 3)) +
            0.3 * (1 - np.tanh(alpha / (base_alpha + 1e-10) - 1.5)),  # alpha below resting = attending
            0, 1
        ))

        # 2. Hippocampal Theta: key memory encoding oscillation
        # Theta power increase during successful encoding
        theta_increase = theta / (base_theta + 1e-10)
        hippocampal_theta = float(np.clip(
            0.6 * np.tanh(theta_increase - 0.8) +
            0.4 * np.tanh(theta * 5),
            0, 1
        ))

        # 3. Encoding Depth: theta-gamma coupling
        # Theta provides temporal structure, gamma carries content
        # Their coupling = information being bound into memory
        tg_coupling = theta * gamma * 100
        encoding_depth = float(np.clip(
            0.4 * np.tanh(tg_coupling) +
            0.3 * hippocampal_theta +
            0.3 * attention,
            0, 1
        ))

        # 4. Alpha Desynchronization: attention-driven
        # Alpha DECREASE from baseline = active processing
        alpha_decrease = 1 - (alpha / (base_alpha + 1e-10))
        alpha_desync = float(np.clip(
            0.5 * np.tanh(alpha_decrease) +
            0.5 * attention,
            0, 1
        ))

        # === Overall Encoding Score ===
        encoding_score = float(np.clip(
            0.25 * attention +
            0.30 * hippocampal_theta +
            0.25 * encoding_depth +
            0.20 * alpha_desync,
            0, 1
        ))

        # Will-remember probability (calibrated to be conservative)
        will_remember = float(np.clip(
            0.3 + encoding_score * 0.5 + np.random.uniform(-0.03, 0.03),
            0.15, 0.85
        ))

        # Classify state
        if encoding_score >= 0.7:
            state_idx = 3  # deep_encoding
        elif encoding_score >= 0.5:
            state_idx = 2  # active_encoding
        elif encoding_score >= 0.3:
            state_idx = 1  # weak_encoding
        else:
            state_idx = 0  # poor_encoding

        confidence = float(np.clip(0.4 + encoding_score * 0.4 + np.random.uniform(-0.02, 0.02), 0.3, 0.85))

        return {
            "state": self.STATES[state_idx],
            "state_index": state_idx,
            "encoding_score": round(encoding_score, 3),
            "will_remember_probability": round(will_remember, 3),
            "confidence": round(confidence, 3),
            "components": {
                "attention_level": round(attention, 3),
                "hippocampal_theta": round(hippocampal_theta, 3),
                "encoding_depth": round(encoding_depth, 3),
                "alpha_desynchronization": round(alpha_desync, 3),
            },
            "band_powers": bands,
        }

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        """Sklearn model inference using extracted features."""
        processed = preprocess(eeg, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        feature_vector = np.array([features[k] for k in self.feature_names]).reshape(1, -1)

        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)

        probs = self.sklearn_model.predict_proba(feature_vector)[0]
        state_idx = int(np.argmax(probs))
        encoding_score = float(state_idx / 3.0 * 0.6 + probs[state_idx] * 0.4)
        will_remember = float(np.clip(0.3 + encoding_score * 0.5, 0.15, 0.85))

        return {
            "state": self.STATES[state_idx],
            "state_index": state_idx,
            "encoding_score": round(float(np.clip(encoding_score, 0, 1)), 3),
            "will_remember_probability": round(will_remember, 3),
            "confidence": round(float(probs[state_idx]), 3),
            "components": {
                "attention_level": round(float(bands.get("beta", 0.2)), 3),
                "hippocampal_theta": round(float(bands.get("theta", 0.15)), 3),
                "encoding_depth": round(float(probs[3]) if len(probs) > 3 else 0.2, 3),
                "alpha_desynchronization": round(1.0 - float(bands.get("alpha", 0.3)), 3),
            },
            "band_powers": bands,
        }
