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

# Minimum benchmark accuracy to trust a trained model over feature-based.
# 60% threshold: the DEAP model at ~51% has severe class-imbalance bias toward "sad"
# (11 165 sad samples vs 1 769 focused) — below this threshold we use feature-based.
_MIN_MODEL_ACCURACY = 0.60


class EmotionClassifier:
    """EEG-based emotion classifier with ONNX/sklearn/feature-based inference.

    Priority: multichannel DEAP model → ONNX → sklearn .pkl → feature-based.
    Trained models are only used if their benchmark accuracy >= 60%.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.onnx_session = None
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.model_type = "feature-based"
        self._benchmark_accuracy = 0.0

        # Exponential moving average for smoothing (prevents flickering)
        self._ema_probs = None  # smoothed probabilities
        self._ema_alpha = 0.3   # smoothing factor (lower = smoother)
        self._history = deque(maxlen=10)  # recent band power snapshots

        # Try loading the DEAP-trained multichannel model first
        self._try_load_deap_model()

        if model_path and self.model_type == "feature-based":
            self._load_model(model_path)

    def _try_load_deap_model(self):
        """Try loading the DEAP-trained Muse 2 model (best accuracy)."""
        from pathlib import Path
        pkl_path = Path("models/saved/emotion_deap_muse.pkl")
        if not pkl_path.exists():
            return

        bench = self._read_benchmark()
        if bench < _MIN_MODEL_ACCURACY:
            return

        try:
            import joblib
            data = joblib.load(pkl_path)
            self.sklearn_model = data["model"]
            self.feature_names = data["feature_names"]
            self.scaler = data.get("scaler")
            self.model_type = "sklearn-deap"
            self._benchmark_accuracy = bench
        except Exception:
            pass

    def _load_model(self, model_path: str):
        """Load model from file (ONNX or sklearn pkl)."""
        self._benchmark_accuracy = self._read_benchmark()

        if self._benchmark_accuracy < _MIN_MODEL_ACCURACY:
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
                self.scaler = data.get("scaler")
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
        """Classify emotion from EEG signal.

        Args:
            eeg: 1D (single channel) or 2D (n_channels, n_samples) array.
            fs: Sampling frequency.
        """
        # Multichannel DEAP model (requires exactly 4 Muse 2 channels: AF7, AF8, TP9, TP10)
        if self.model_type == "sklearn-deap" and eeg.ndim == 2 and eeg.shape[0] >= 4:
            return self._predict_multichannel(eeg, fs)
        if self.onnx_session is not None and self._benchmark_accuracy >= _MIN_MODEL_ACCURACY:
            return self._predict_onnx(eeg if eeg.ndim == 1 else eeg[0], fs)
        if self.sklearn_model is not None and self._benchmark_accuracy >= _MIN_MODEL_ACCURACY:
            return self._predict_sklearn(eeg if eeg.ndim == 1 else eeg[0], fs)
        return self._predict_features(eeg if eeg.ndim == 1 else eeg[0], fs)

    # ────────────────────────────────────────────────────────────────
    # Multichannel DEAP-trained model (primary for Muse 2)
    # ────────────────────────────────────────────────────────────────

    def _predict_multichannel(self, channels: np.ndarray, fs: float) -> Dict:
        """Predict using the DEAP-trained multichannel model."""
        from training.train_deap_muse import extract_multichannel_features

        features = extract_multichannel_features(channels, fs)
        feat_vec = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)

        if self.scaler is not None:
            feat_vec = self.scaler.transform(feat_vec)

        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

        probs = self.sklearn_model.predict_proba(feat_vec)[0]
        emotion_idx = int(np.argmax(probs))

        # EMA smoothing
        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            self._ema_probs = self._ema_alpha * probs + (1 - self._ema_alpha) * self._ema_probs

        smoothed = self._ema_probs / (self._ema_probs.sum() + 1e-10)
        emotion_idx = int(np.argmax(smoothed))

        # Extract band powers from first channel for extra metrics
        processed = preprocess(channels[0], fs)
        bands = extract_band_powers(processed, fs)
        de = differential_entropy(processed, fs)

        alpha_raw = bands.get("alpha", 0)
        beta_raw  = bands.get("beta",  0)
        theta_raw = bands.get("theta", 0)
        gamma_raw = bands.get("gamma", 0)
        delta_raw = bands.get("delta", 0)

        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha = alpha_raw / total_power
        beta  = beta_raw  / total_power
        theta = theta_raw / total_power
        gamma = gamma_raw / total_power

        alpha_beta_ratio = alpha / max(beta, 1e-10)
        beta_alpha_ratio = beta / max(alpha, 1e-10)
        theta_beta_ratio = theta / max(beta, 1e-10)

        valence = float(np.tanh((alpha_beta_ratio - 1.0) * 1.5) * 0.5 + (gamma - 0.15) * 2)
        valence = float(np.clip(valence, -1, 1))
        arousal = float(np.clip(beta / max(beta + alpha, 1e-10) + gamma * 0.5, 0, 1))

        stress_index = float(np.clip(
            0.50 * min(1, beta_alpha_ratio * 0.3) + 0.30 * max(0, 1 - alpha * 2.5) + 0.20 * min(1, gamma * 2), 0, 1))
        focus_index = float(np.clip(
            0.40 * min(1, beta * 3.0) + 0.35 * max(0, 1 - theta_beta_ratio * 0.35) + 0.25 * min(1, gamma * 2), 0, 1))
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5) + 0.30 * max(0, 1 - beta_alpha_ratio * 0.3) + 0.20 * min(1, theta * 1.5), 0, 1))
        anger_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5) + 0.35 * min(1, gamma * 4) + 0.30 * max(0, 1 - alpha * 5), 0, 1))

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
            "anger_index": anger_index,
            "band_powers": bands,
            "differential_entropy": de,
        }

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

        alpha_raw = bands.get("alpha", 0)
        beta_raw  = bands.get("beta",  0)
        theta_raw = bands.get("theta", 0)
        gamma_raw = bands.get("gamma", 0)
        delta_raw = bands.get("delta", 0)

        # Normalize to relative fractions that sum to 1.
        # Without this the absolute power values from BrainFlow (sum often 0.4-0.8)
        # make every formula threshold fire at different levels, producing flat probs.
        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha = alpha_raw / total_power
        beta  = beta_raw  / total_power
        theta = theta_raw / total_power
        gamma = gamma_raw / total_power
        delta = delta_raw / total_power

        # Store snapshot for temporal analysis
        self._history.append(bands)

        # ── Valence (pleasantness) ──────────────────────────────
        # Higher alpha relative to beta → positive valence.
        # Reference ratio 0.7: eyes-open resting naturally has beta > alpha (ratio ~0.5-0.8)
        # so we treat that as neutral, not negative.
        # Theta is NOT a valence signal — high theta means drowsy/meditative (neutral),
        # not sad. Removing theta from valence prevents "drowsy = fearful" misclassification.
        alpha_beta_ratio = alpha / max(beta, 1e-10)
        valence_raw = (
            0.65 * np.tanh((alpha_beta_ratio - 0.7) * 2.0)  # 0.7 = neutral eyes-open resting
            + 0.35 * np.tanh((alpha - 0.15) * 4)            # absolute alpha boost
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

        # Sad: clearly negative valence + low arousal.
        # Theta only contributes when VERY dominant (ratio > 2.0) AND valence is negative
        # — prevents drowsy/meditative theta from being mis-classified as sad.
        probs[1] = (
            0.50 * max(0, -valence - 0.1)                  # needs clear negative valence
            + 0.30 * max(0, 0.35 - arousal)                 # must be truly low arousal
            + 0.20 * max(0, theta_beta_ratio - 2.0) * 0.4  # only VERY high theta/beta
        )

        # Angry: negative valence + elevated arousal + beta dominance + gamma spike.
        # Key differentiator from fearful: anger is HIGH-ENERGY (more gamma, more beta action),
        # while fear is more freeze-response. Lower thresholds so at least two terms fire together.
        probs[2] = (
            0.30 * max(0, -valence - 0.1)                   # negative valence (lowered from -0.2)
            + 0.25 * max(0, arousal - 0.55)                  # elevated arousal (lowered from 0.65)
            + 0.25 * max(0, beta_alpha_ratio - 1.5)          # beta dominance (removed *0.5 penalty, threshold 2.0→1.5)
            + 0.20 * min(1, gamma * 3)                        # gamma is the primary anger differentiator
        )

        # Fearful/Anxious: requires EXTREME conditions to avoid false positives.
        # Normal eyes-open (beta > alpha) should NOT be fearful.
        # Needs: very high beta/alpha ratio (>2.5) AND negative valence AND elevated arousal.
        probs[3] = (
            0.30 * max(0, -valence - 0.3)                    # needs STRONGLY negative valence
            + 0.30 * max(0, arousal - 0.6)                   # needs high arousal (>0.6)
            + 0.25 * max(0, beta_alpha_ratio - 2.5) * 0.5   # needs extreme beta/alpha (>2.5)
            + 0.15 * max(0, 1 - alpha * 6)                   # very low alpha required
        )

        # Relaxed: low arousal + dominant alpha OR high theta (meditative/drowsy).
        # Theta at 30-55% of power = meditative/drowsy waking state → relaxed, not fearful.
        probs[4] = (
            0.10 * max(0, valence * 0.5)                  # positive valence is a soft bonus
            + 0.20 * max(0, 0.6 - arousal)                # low arousal
            + 0.35 * min(1, alpha * 2.5)                   # dominant alpha (primary relaxation signal)
            + 0.20 * min(1, theta * 1.5)                   # high theta = meditative/drowsy → relaxed
            + 0.15 * max(0, 1 - beta_alpha_ratio * 0.5)   # low beta relative to alpha
        )

        # Focused: moderate-high arousal, strong beta, low theta/beta ratio.
        # Valence can be slightly positive (engaged) or neutral — penalizing any valence
        # deviation was wrong since focused people can feel any mild emotion.
        probs[5] = (
            0.10 * max(0, 0.5 + valence * 0.5)               # soft neutral-to-positive valence bonus
            + 0.25 * min(1, max(0, arousal - 0.35) * 2.5)    # moderate-high arousal
            + 0.35 * min(1, beta * 3.0)                       # strong beta (slightly stronger weight)
            + 0.30 * max(0, 1 - theta_beta_ratio * 0.35)      # low theta/beta (less strict than 0.5)
        )

        # Softmax-like normalization (temperature=2.5 for clear sharpness).
        # Subtract max before exp to prevent overflow (numerically stable softmax).
        temp = 2.5
        scaled = probs * temp
        scaled -= scaled.max()  # shift so max is 0 → exp values in (0, 1]
        probs_exp = np.exp(scaled)
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
            0.40 * min(1, beta * 3.0)
            + 0.35 * max(0, 1 - theta_beta_ratio * 0.35)
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

        # Anger: high beta/alpha ratio, gamma spike, suppressed alpha
        anger_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5)
            + 0.35 * min(1, gamma * 4)
            + 0.30 * max(0, 1 - alpha * 5),
            0, 1
        ))

        # If max confidence is below threshold the EEG doesn't clearly match any
        # of the 6 trained emotions — label as "neutral" instead of forcing a wrong label.
        # With 6 classes, random baseline = 0.167. Threshold 0.20 means the top emotion
        # is at least 20% more likely than chance — appropriate for consumer EEG devices.
        _CONFIDENCE_THRESHOLD = 0.19
        top_conf = float(smoothed[emotion_idx])
        emotion_label = EMOTIONS[emotion_idx] if top_conf >= _CONFIDENCE_THRESHOLD else "neutral"

        return {
            "emotion": emotion_label,
            "emotion_index": emotion_idx,
            "confidence": top_conf,
            "probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(smoothed)},
            "valence": valence,
            "arousal": arousal,
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "anger_index": anger_index,
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

        alpha_raw = bands.get("alpha", 0)
        beta_raw  = bands.get("beta",  0)
        theta_raw = bands.get("theta", 0)
        gamma_raw = bands.get("gamma", 0)
        delta_raw = bands.get("delta", 0)

        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha = alpha_raw / total_power
        beta  = beta_raw  / total_power
        theta = theta_raw / total_power
        gamma = gamma_raw / total_power

        beta_alpha_ratio = beta / max(alpha, 1e-10)
        theta_beta_ratio = theta / max(beta, 1e-10)

        valence = float(np.tanh((alpha / max(beta, 1e-10) - 0.7) * 2.0))
        arousal = float(np.clip(beta / max(beta + alpha, 1e-10) + gamma * 0.5, 0, 1))

        stress_index = float(np.clip(
            0.50 * min(1, beta_alpha_ratio * 0.3) + 0.30 * max(0, 1 - alpha * 2.5) + 0.20 * min(1, gamma * 2), 0, 1))
        focus_index = float(np.clip(
            0.40 * min(1, beta * 3.0) + 0.35 * max(0, 1 - theta_beta_ratio * 0.35) + 0.25 * min(1, gamma * 2), 0, 1))
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5) + 0.30 * max(0, 1 - beta_alpha_ratio * 0.3) + 0.20 * min(1, theta * 1.5), 0, 1))
        anger_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5) + 0.35 * min(1, gamma * 4) + 0.30 * max(0, 1 - alpha * 5), 0, 1))

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
            "anger_index": anger_index,
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

        try:
            feature_vector = np.array([features[k] for k in self.feature_names]).reshape(1, -1)
        except KeyError:
            return self._predict_features(eeg, fs)
        probs = self.sklearn_model.predict_proba(feature_vector)[0]
        emotion_idx = int(np.argmax(probs))

        alpha_raw = bands.get("alpha", 0)
        beta_raw  = bands.get("beta",  0)
        theta_raw = bands.get("theta", 0)
        gamma_raw = bands.get("gamma", 0)
        delta_raw = bands.get("delta", 0)

        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha = alpha_raw / total_power
        beta  = beta_raw  / total_power
        theta = theta_raw / total_power
        gamma = gamma_raw / total_power

        beta_alpha_ratio = beta / max(alpha, 1e-10)
        theta_beta_ratio = theta / max(beta, 1e-10)

        valence = float(np.tanh((alpha / max(beta, 1e-10) - 0.7) * 2.0))
        arousal = float(np.clip(beta / max(beta + alpha, 1e-10) + gamma * 0.5, 0, 1))

        stress_index = float(np.clip(
            0.50 * min(1, beta_alpha_ratio * 0.3) + 0.30 * max(0, 1 - alpha * 2.5) + 0.20 * min(1, gamma * 2), 0, 1))
        focus_index = float(np.clip(
            0.40 * min(1, beta * 3.0) + 0.35 * max(0, 1 - theta_beta_ratio * 0.35) + 0.25 * min(1, gamma * 2), 0, 1))
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5) + 0.30 * max(0, 1 - beta_alpha_ratio * 0.3) + 0.20 * min(1, theta * 1.5), 0, 1))
        anger_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5) + 0.35 * min(1, gamma * 4) + 0.30 * max(0, 1 - alpha * 5), 0, 1))

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
            "anger_index": anger_index,
            "band_powers": bands,
            "differential_entropy": de,
        }
