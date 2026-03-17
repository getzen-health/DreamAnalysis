"""EEGNet-based Sleep Staging Model.

Classifies EEG epochs into 5 sleep stages:
- Wake (W)
- N1 (light sleep)
- N2 (light sleep with spindles)
- N3 (deep/slow-wave sleep)
- REM (rapid eye movement)

Architecture: Compact CNN (EEGNet variant) optimized for ONNX export.
Supports three inference paths: ONNX > sklearn > feature-based fallback.
"""

import numpy as np
from typing import Dict, List, Optional
from processing.eeg_processor import extract_features, extract_band_powers, preprocess, detect_sleep_spindles

SLEEP_STAGES = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


class SleepStagingModel:
    """Sleep staging classifier with ONNX/sklearn/feature-based inference.

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

    def predict(self, eeg_epoch: np.ndarray, fs: float = 256.0) -> Dict:
        """Predict sleep stage from a single EEG epoch.

        Args:
            eeg_epoch: 1D numpy array of EEG samples (typically 30s * fs samples)
            fs: Sampling frequency

        Returns:
            Dict with 'stage', 'stage_index', 'confidence', 'probabilities'
        """
        if self.onnx_session is not None:
            return self._predict_onnx(eeg_epoch, fs)
        if self.sklearn_model is not None:
            try:
                return self._predict_sklearn(eeg_epoch, fs)
            except Exception:
                pass  # fall through to feature-based
        return self._predict_features(eeg_epoch, fs)

    def _predict_sklearn(self, eeg_epoch: np.ndarray, fs: float) -> Dict:
        """Sklearn model inference using extracted features."""
        # Reject epoch if amplitude exceeds 100 µV (blink/movement artifact)
        if np.any(np.abs(eeg_epoch) > 100):
            return {
                "stage": "artifact",
                "stage_index": -1,
                "confidence": 0.0,
                "probabilities": {s: 0.0 for s in STAGE_MAP.values()},
                "artifact_rejected": True,
            }

        # Use multichannel features when available — delta asymmetry between
        # hemispheres improves N3 vs REM discrimination.
        if eeg_epoch.ndim == 2 and eeg_epoch.shape[0] >= 2:
            try:
                from processing.eeg_processor import extract_features_multichannel
                features = extract_features_multichannel(eeg_epoch, fs)
                processed = preprocess(eeg_epoch[0], fs)  # AF7 channel for spindle detection
            except Exception:
                processed = preprocess(eeg_epoch[0], fs)
                features = extract_features(processed, fs)
        else:
            processed = preprocess(eeg_epoch, fs)
            features = extract_features(processed, fs)
        feature_vector = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)

        probs = self.sklearn_model.predict_proba(feature_vector)[0]

        # Boost N2 probability when sleep spindles are detected
        try:
            spindles = detect_sleep_spindles(processed, fs)
            if spindles.get("spindles_detected", False):
                probs[2] = min(1.0, probs[2] + 0.15)
                probs = probs / probs.sum()
        except Exception:
            pass  # spindle detection failure must not break staging

        stage_idx = int(np.argmax(probs))

        return {
            "stage": STAGE_MAP[stage_idx],
            "stage_index": stage_idx,
            "confidence": float(probs[stage_idx]),
            "probabilities": {STAGE_MAP[i]: float(p) for i, p in enumerate(probs)},
        }

    def _predict_features(self, eeg_epoch: np.ndarray, fs: float) -> Dict:
        """Feature-based classification using physiological rules."""
        # Reject epoch if amplitude exceeds 100 µV (blink/movement artifact)
        if np.any(np.abs(eeg_epoch) > 100):
            return {
                "stage": "artifact",
                "stage_index": -1,
                "confidence": 0.0,
                "probabilities": {s: 0.0 for s in STAGE_MAP.values()},
                "artifact_rejected": True,
            }

        # Extract single channel for feature-based analysis
        signal = eeg_epoch[0] if eeg_epoch.ndim == 2 else eeg_epoch
        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)
        features = extract_features(processed, fs)

        delta = bands.get("delta", 0)
        theta = bands.get("theta", 0)
        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        gamma = bands.get("gamma", 0)

        # Compute probabilities based on physiological characteristics
        probs = np.zeros(5)

        # Wake: high alpha + beta, low delta (no gamma — EMG contamination at AF7/AF8)
        probs[0] = alpha * 0.4 + beta * 0.4 + (1 - delta) * 0.2

        # N1: theta rising while alpha dropping — the transition zone.
        # Key differentiator from Wake (alpha high) and N2 (spindles/K-complexes).
        # Alpha dropout: alpha is falling but not gone; theta/alpha ratio increases.
        theta_alpha_ratio = theta / max(alpha + theta, 0.01)
        alpha_dropout = float(np.clip(alpha * (1.0 - theta_alpha_ratio), 0, 1))
        probs[1] = (theta * 0.35
                     + alpha_dropout * 0.25       # alpha still present but fading
                     + (1 - beta) * 0.20          # low beta (not awake-alert)
                     + (1 - delta) * 0.10         # not yet deep sleep
                     + theta_alpha_ratio * 0.10)  # theta starting to dominate

        # N2: theta + sleep spindles (sigma 12-14Hz within beta)
        sigma_component = min(beta * 0.3, 0.15)
        probs[2] = theta * 0.35 + sigma_component + delta * 0.25 + (1 - alpha) * 0.15

        # N3: high delta (slow-wave sleep)
        probs[3] = delta * 0.7 + theta * 0.15 + (1 - beta) * 0.1 + (1 - alpha) * 0.05

        # REM: mixed frequency, theta + beta, low delta (no gamma — EMG contamination)
        probs[4] = theta * 0.3 + beta * 0.3 + (1 - delta) * 0.2 + (1 - alpha) * 0.1 + (1 - alpha) * 0.1

        # Add noise for realism
        probs += np.random.uniform(0, 0.05, 5)

        # Normalize
        probs = probs / probs.sum()
        stage_idx = int(np.argmax(probs))

        return {
            "stage": STAGE_MAP[stage_idx],
            "stage_index": stage_idx,
            "confidence": float(probs[stage_idx]),
            "probabilities": {STAGE_MAP[i]: float(p) for i, p in enumerate(probs)},
        }

    def _predict_onnx(self, eeg_epoch: np.ndarray, fs: float) -> Dict:
        """ONNX model inference."""
        input_data = eeg_epoch.reshape(1, 1, -1).astype(np.float32)
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: input_data})
        probs = outputs[0][0]
        if probs.shape[0] != 5:
            probs = np.zeros(5)
            probs[0] = 1.0
        stage_idx = int(np.argmax(probs))
        return {
            "stage": STAGE_MAP[stage_idx],
            "stage_index": stage_idx,
            "confidence": float(probs[stage_idx]),
            "probabilities": {STAGE_MAP[i]: float(p) for i, p in enumerate(probs)},
        }

    def predict_sequence(self, epochs: List[np.ndarray], fs: float = 256.0) -> List[Dict]:
        """Predict sleep stages for a sequence of epochs (full night)."""
        results = []
        for epoch in epochs:
            result = self.predict(epoch, fs)
            results.append(result)
        return self._smooth_predictions(results)

    # Biologically invalid transition penalties (Markov prior).
    # Stage indices: Wake=0, N1=1, N2=2, N3=3, REM=4.
    # Tuple format: (from_stage, to_stage) → multiplier for the destination probability.
    _TRANSITION_PENALTIES = {
        (0, 4): 0.1,  # Wake → REM direct: essentially impossible
        (3, 1): 0.2,  # N3 → N1 direct: must surface through N2 first
        (4, 3): 0.2,  # REM → N3 direct: very rare, REM usually lightens to N1/N2
    }

    def _smooth_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Apply temporal smoothing with Markov transition priors.

        1. Penalise biologically invalid stage jumps by scaling down the
           destination-stage probability before argmax.
        2. Fall back to the previous stage if confidence is low and the
           jump is large (legacy behaviour retained).
        """
        if len(predictions) <= 1:
            return predictions

        smoothed = [predictions[0]]
        for i in range(1, len(predictions)):
            current = predictions[i]
            prev_stage = smoothed[i - 1]["stage_index"]
            curr_stage = current["stage_index"]

            # --- Markov transition prior ---
            penalty = self._TRANSITION_PENALTIES.get((prev_stage, curr_stage))
            if penalty is not None:
                # Re-weight probabilities: scale down the invalid destination,
                # then renormalise so all probs still sum to 1.
                raw_probs = np.array(
                    [current["probabilities"][STAGE_MAP[j]] for j in range(5)]
                )
                raw_probs[curr_stage] *= penalty
                raw_probs = raw_probs / raw_probs.sum()
                new_stage_idx = int(np.argmax(raw_probs))
                current = {
                    **current,
                    "stage": STAGE_MAP[new_stage_idx],
                    "stage_index": new_stage_idx,
                    "confidence": float(raw_probs[new_stage_idx]),
                    "probabilities": {STAGE_MAP[j]: float(raw_probs[j]) for j in range(5)},
                }
                curr_stage = new_stage_idx

            # --- Legacy large-jump guard (unchanged) ---
            if abs(curr_stage - prev_stage) > 2 and current["confidence"] < 0.6:
                current = {**current, "stage": STAGE_MAP[prev_stage], "stage_index": prev_stage}

            smoothed.append(current)

        return smoothed
