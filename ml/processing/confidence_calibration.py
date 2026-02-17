"""Confidence Calibration — Make model confidence scores actually meaningful.

Problem: A model that says "80% confident" should be right 80% of the time.
Our heuristic-based models often output overconfident scores (says 90% but
is only right 60% of the time). This is dangerous — users trust confident
predictions, so wrong-and-confident is worse than wrong-and-uncertain.

Solution:
1. Platt scaling: fit a sigmoid to map raw scores → calibrated probabilities
2. Temperature scaling: divide logits by a learned temperature parameter
3. Uncertainty detection: output "uncertain" when calibrated confidence is low
4. Per-model calibration: each model gets its own calibration curve

Without real labeled data, we use conservative heuristic calibration that
deliberately underestimates confidence. Better to say "I'm 60% sure" when
you're actually 70% sure than the reverse.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


CALIBRATION_DIR = Path(__file__).parent.parent / "data" / "confidence_calibrations"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)


class ConfidenceCalibrator:
    """Calibrates model confidence scores to be more honest.

    Without labeled validation data, we apply conservative transformations:
    - Compress scores toward 50% (reduce overconfidence)
    - Apply per-model priors based on expected accuracy
    - Output "uncertain" when confidence is below threshold
    """

    # Expected accuracy of each model (conservative estimates).
    # These determine how much we compress confidence scores.
    MODEL_PRIORS = {
        "sleep_staging": {
            "expected_accuracy": 0.65,  # Sleep staging is well-studied
            "compression": 0.6,         # Moderate compression
            "uncertain_threshold": 0.40,
        },
        "emotion": {
            "expected_accuracy": 0.50,  # Emotion from EEG is hard
            "compression": 0.5,         # More compression (less certain)
            "uncertain_threshold": 0.35,
        },
        "dream_detection": {
            "expected_accuracy": 0.55,
            "compression": 0.5,
            "uncertain_threshold": 0.45,
        },
        "flow_state": {
            "expected_accuracy": 0.55,
            "compression": 0.55,
            "uncertain_threshold": 0.40,
        },
        "creativity": {
            "expected_accuracy": 0.45,  # Most speculative model
            "compression": 0.45,
            "uncertain_threshold": 0.35,
        },
        "memory_encoding": {
            "expected_accuracy": 0.50,
            "compression": 0.5,
            "uncertain_threshold": 0.40,
        },
    }

    def __init__(self):
        self.calibration_params = {}
        self._load_calibrations()

    def calibrate(self, model_name: str, raw_confidence: float,
                  raw_probs: Optional[np.ndarray] = None) -> Dict:
        """Calibrate a model's confidence score.

        Args:
            model_name: Name of the model (e.g., "sleep_staging", "emotion").
            raw_confidence: Raw confidence score from model (0-1).
            raw_probs: Optional full probability distribution.

        Returns:
            Dict with calibrated_confidence, is_uncertain, and details.
        """
        prior = self.MODEL_PRIORS.get(model_name, {
            "expected_accuracy": 0.5,
            "compression": 0.5,
            "uncertain_threshold": 0.40,
        })

        # Check if we have learned Platt scaling params
        if model_name in self.calibration_params:
            calibrated = self._platt_scale(
                raw_confidence,
                self.calibration_params[model_name]
            )
        else:
            # Conservative heuristic calibration
            calibrated = self._conservative_calibrate(
                raw_confidence, prior["compression"]
            )

        # Calibrate full distribution if available
        calibrated_probs = None
        if raw_probs is not None:
            # Handle dict-style probabilities
            if isinstance(raw_probs, dict):
                raw_probs = np.array(list(raw_probs.values()), dtype=float)
            elif isinstance(raw_probs, list):
                raw_probs = np.array(raw_probs, dtype=float)
            calibrated_probs = self._calibrate_distribution(
                raw_probs, prior["compression"]
            )

        is_uncertain = calibrated < prior["uncertain_threshold"]

        return {
            "calibrated_confidence": round(calibrated, 3),
            "raw_confidence": round(raw_confidence, 3),
            "is_uncertain": is_uncertain,
            "uncertainty_note": (
                f"Low confidence ({calibrated:.0%}) — treat as suggestion, not fact"
                if is_uncertain else None
            ),
            "calibrated_probs": (
                [round(float(p), 3) for p in calibrated_probs]
                if calibrated_probs is not None else None
            ),
        }

    def calibrate_prediction(self, model_name: str, prediction: Dict) -> Dict:
        """Calibrate a full prediction dict in-place.

        Takes the raw prediction from any model and adds calibrated scores.
        """
        # Extract confidence from various model output formats
        raw_conf = (
            prediction.get("confidence")
            or prediction.get("flow_score")
            or prediction.get("creativity_score")
            or prediction.get("encoding_score")
            or prediction.get("probability")
            or 0.5
        )

        raw_probs = prediction.get("probabilities")

        # Handle dict-style probabilities (e.g., {"Wake": 0.5, "N1": 0.2})
        if isinstance(raw_probs, dict):
            raw_probs = np.array(list(raw_probs.values()), dtype=float)
        elif isinstance(raw_probs, list):
            raw_probs = np.array(raw_probs, dtype=float)

        cal = self.calibrate(model_name, raw_conf, raw_probs)

        # Add calibration info to prediction
        prediction["calibrated_confidence"] = cal["calibrated_confidence"]
        prediction["is_uncertain"] = cal["is_uncertain"]
        if cal["uncertainty_note"]:
            prediction["uncertainty_note"] = cal["uncertainty_note"]
        if cal["calibrated_probs"] is not None:
            prediction["calibrated_probs"] = cal["calibrated_probs"]

        return prediction

    def fit_platt_scaling(self, model_name: str,
                          raw_scores: List[float],
                          true_labels: List[int]):
        """Fit Platt scaling parameters from validation data.

        Call this when you have actual labeled data to calibrate against.

        Args:
            model_name: Model name.
            raw_scores: List of raw confidence scores from model.
            true_labels: List of 0/1 labels (1 = model was correct).
        """
        if len(raw_scores) < 20:
            return  # Not enough data

        scores = np.array(raw_scores)
        labels = np.array(true_labels, dtype=float)

        # Fit logistic regression: P(correct) = sigmoid(a * score + b)
        # Simple gradient descent
        a, b = 1.0, 0.0
        lr = 0.01

        for _ in range(1000):
            z = a * scores + b
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-7, 1 - 1e-7)

            # Cross-entropy gradient
            grad_a = np.mean((p - labels) * scores)
            grad_b = np.mean(p - labels)

            a -= lr * grad_a
            b -= lr * grad_b

        self.calibration_params[model_name] = {"a": float(a), "b": float(b)}
        self._save_calibrations()

    def _platt_scale(self, raw_score: float, params: Dict) -> float:
        """Apply Platt scaling: calibrated = sigmoid(a * raw + b)."""
        z = params["a"] * raw_score + params["b"]
        return float(1.0 / (1.0 + np.exp(-z)))

    def _conservative_calibrate(self, raw_conf: float, compression: float) -> float:
        """Conservative calibration without labeled data.

        Compresses scores toward the uniform prior (1/n_classes).
        compression=0 → always returns 0.5 (maximum uncertainty)
        compression=1 → returns raw score unchanged
        """
        # Compress toward 0.5 (maximum entropy)
        calibrated = 0.5 + compression * (raw_conf - 0.5)

        # Additional penalty for very high confidence (likely overconfident)
        if raw_conf > 0.9:
            calibrated *= 0.85  # 15% penalty for extreme confidence

        return np.clip(calibrated, 0.0, 1.0)

    def _calibrate_distribution(self, probs: np.ndarray,
                                compression: float) -> np.ndarray:
        """Calibrate a full probability distribution (temperature scaling).

        Softens the distribution by raising temperature (less peaked).
        """
        probs = np.array(probs, dtype=float)
        probs = np.maximum(probs, 1e-10)

        # Temperature scaling: higher temp = softer distribution
        temperature = 1.0 / max(compression, 0.1)

        # Convert to log space, scale, convert back
        log_probs = np.log(probs)
        scaled = log_probs / temperature
        scaled -= np.max(scaled)  # Numerical stability
        calibrated = np.exp(scaled)
        calibrated /= calibrated.sum()

        return calibrated

    def get_model_reliability(self, model_name: str) -> Dict:
        """Get reliability assessment for a model."""
        prior = self.MODEL_PRIORS.get(model_name, {})

        has_calibration = model_name in self.calibration_params

        return {
            "model": model_name,
            "expected_accuracy": prior.get("expected_accuracy", 0.5),
            "is_calibrated": has_calibration,
            "calibration_method": "platt_scaling" if has_calibration else "conservative_heuristic",
            "reliability_tier": (
                "high" if prior.get("expected_accuracy", 0) >= 0.65
                else "medium" if prior.get("expected_accuracy", 0) >= 0.50
                else "low"
            ),
            "note": (
                "Calibrated on real data" if has_calibration
                else "Using conservative estimates — confidence scores may be pessimistic"
            ),
        }

    def get_all_reliability(self) -> Dict:
        """Get reliability for all models."""
        return {
            name: self.get_model_reliability(name)
            for name in self.MODEL_PRIORS
        }

    def _save_calibrations(self):
        """Save learned calibration params to disk."""
        filepath = CALIBRATION_DIR / "platt_params.json"
        filepath.write_text(json.dumps(self.calibration_params, indent=2))

    def _load_calibrations(self):
        """Load learned calibration params from disk."""
        filepath = CALIBRATION_DIR / "platt_params.json"
        if filepath.exists():
            self.calibration_params = json.loads(filepath.read_text())


def add_uncertainty_labels(predictions: Dict, calibrator: ConfidenceCalibrator) -> Dict:
    """Add calibrated confidence and uncertainty labels to all predictions.

    Usage:
        calibrator = ConfidenceCalibrator()
        raw_predictions = run_all_models(eeg_signal)
        labeled = add_uncertainty_labels(raw_predictions, calibrator)
        # Now each prediction has calibrated_confidence and is_uncertain
    """
    model_key_map = {
        "sleep_staging": "sleep_staging",
        "emotions": "emotion",
        "dream_detection": "dream_detection",
        "flow_state": "flow_state",
        "creativity": "creativity",
        "memory_encoding": "memory_encoding",
    }

    for pred_key, model_name in model_key_map.items():
        if pred_key in predictions:
            calibrator.calibrate_prediction(model_name, predictions[pred_key])

    # Add overall confidence summary
    confidences = []
    for pred_key in model_key_map:
        if pred_key in predictions:
            c = predictions[pred_key].get("calibrated_confidence", 0.5)
            confidences.append(c)

    if confidences:
        predictions["_confidence_summary"] = {
            "mean_confidence": round(float(np.mean(confidences)), 3),
            "min_confidence": round(float(np.min(confidences)), 3),
            "n_uncertain": sum(
                1 for k in model_key_map
                if k in predictions and predictions[k].get("is_uncertain", False)
            ),
            "overall_reliability": (
                "good" if np.mean(confidences) >= 0.5
                else "fair" if np.mean(confidences) >= 0.35
                else "poor"
            ),
        }

    return predictions
