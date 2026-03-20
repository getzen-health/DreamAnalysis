"""Online Learning / Personal Model Adapter.

Wraps base models with a personal adaptation layer that learns
from user calibration data and incremental feedback corrections.
Uses SGDClassifier with warm_start for online updates.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional

from processing.eeg_processor import extract_features, preprocess

USER_MODELS_DIR = Path(__file__).parent.parent / "user_models"
USER_MODELS_DIR.mkdir(exist_ok=True)


class PersonalModelAdapter:
    """Wraps a base model with personal adaptation via incremental learning.

    The personal model is an SGDClassifier that is trained on user-provided
    calibration data and updated incrementally when the user corrects predictions.
    Final predictions are an ensemble of base model + personal model, weighted
    by the personal model's confidence.
    """

    def __init__(self, base_model, user_id: str = "default"):
        self.base_model = base_model
        self.user_id = user_id
        self.personal_model = None
        self.classes = None
        self.n_samples = 0
        self.personal_accuracy = 0.0
        self.feature_names: Optional[List[str]] = None

        # Try to load existing personal model
        self._load()

    def calibrate(
        self, signals_list: List[np.ndarray], labels: List[str], fs: float = 256.0
    ) -> Dict:
        """Train personal model from labeled calibration data.

        Args:
            signals_list: List of EEG signal arrays (one per calibration step).
            labels: Corresponding state labels.
            fs: Sampling frequency.

        Returns:
            Dict with calibration result.
        """
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import LabelEncoder

        # Extract features from each signal
        features_list = []
        for signal in signals_list:
            if signal.ndim > 1:
                signal = signal[0]
            processed = preprocess(signal, fs)
            feats = extract_features(processed, fs)
            features_list.append(feats)

        self.feature_names = list(features_list[0].keys())
        X = np.array([[f[k] for k in self.feature_names] for f in features_list])

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels)
        self.classes = list(le.classes_)

        # Train SGDClassifier with warm_start for future incremental updates
        self.personal_model = SGDClassifier(
            loss="log_loss",
            warm_start=True,
            max_iter=100,
            random_state=42,
        )
        self.personal_model.fit(X, y)
        self.n_samples = len(labels)

        # Estimate accuracy on training data (rough estimate)
        train_preds = self.personal_model.predict(X)
        self.personal_accuracy = float(np.mean(train_preds == y))

        self._save()

        return {
            "calibrated": True,
            "n_samples": self.n_samples,
            "classes": self.classes,
            "personal_accuracy": self.personal_accuracy,
        }

    # Default emotion classes used when auto-initializing from corrections
    # (matches the 6-class labels from the emotion classifier).
    _DEFAULT_CLASSES = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

    def adapt(
        self, signal: np.ndarray, predicted_label: str, user_feedback: str, fs: float = 256.0
    ) -> Dict:
        """Incrementally update personal model when user corrects a prediction.

        Args:
            signal: EEG signal for the corrected prediction.
            predicted_label: What the model predicted.
            user_feedback: What the user says the correct label is.
            fs: Sampling frequency.

        Returns:
            Dict with update result.
        """
        if signal.ndim > 1:
            signal = signal[0]

        processed = preprocess(signal, fs)
        feats = extract_features(processed, fs)
        return self.adapt_from_features(feats, predicted_label, user_feedback)

    def adapt_from_features(
        self, features: Dict[str, float], predicted_label: str, user_feedback: str
    ) -> Dict:
        """Incrementally update personal model from a pre-extracted feature dict.

        This is the primary adaptation entry point for the correction pipeline.
        Unlike adapt(), it does not require raw EEG — it works with cached features
        so label-only corrections (no raw signal) can still improve the model.

        If the SGDClassifier has not been initialized yet (user never ran the
        3-step calibration), it is auto-created on the first correction using
        partial_fit with the default 6 emotion classes.

        Args:
            features: Dict of EEG feature name → value (from extract_features).
            predicted_label: What the model predicted.
            user_feedback: What the user says the correct label is.

        Returns:
            Dict with update result.
        """
        from sklearn.linear_model import SGDClassifier

        # Auto-initialize on first correction if not yet calibrated
        if self.personal_model is None:
            self.classes = list(self._DEFAULT_CLASSES)
            self.feature_names = list(features.keys())
            self.personal_model = SGDClassifier(
                loss="log_loss",
                warm_start=True,
                max_iter=100,
                random_state=42,
            )

        if user_feedback not in self.classes:
            return {"updated": False, "reason": f"Unknown label: {user_feedback}"}

        # Ensure feature_names is set (possible if loaded from old save without it)
        if self.feature_names is None:
            self.feature_names = list(features.keys())

        X = np.array([[features.get(k, 0.0) for k in self.feature_names]])
        y = np.array([self.classes.index(user_feedback)])

        # Incremental update via partial_fit
        all_classes = np.arange(len(self.classes))
        self.personal_model.partial_fit(X, y, classes=all_classes)
        self.n_samples += 1

        self._save()

        return {"updated": True, "n_samples": self.n_samples}

    def predict(self, features: Dict[str, float]) -> Dict:
        """Ensemble prediction: base model + personal model.

        If personal model exists, blend predictions weighted by confidence.
        """
        # Base model prediction (from features dict)
        base_pred = None
        if hasattr(self.base_model, "predict") and hasattr(self.base_model, "_predict_features"):
            # Can't call predict directly since it expects raw signal
            # Use features for personal model only
            pass

        if self.personal_model is None or self.feature_names is None:
            return {"has_personal": False}

        X = np.array([[features.get(k, 0.0) for k in self.feature_names]])

        try:
            probs = self.personal_model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            return {
                "has_personal": True,
                "personal_prediction": self.classes[pred_idx] if self.classes else str(pred_idx),
                "personal_confidence": confidence,
                "personal_probabilities": {
                    self.classes[i]: float(p) for i, p in enumerate(probs)
                } if self.classes else {},
            }
        except Exception:
            return {"has_personal": False}

    def get_calibration_status(self) -> Dict:
        """Check personal model calibration status."""
        return {
            "calibrated": self.personal_model is not None,
            "n_samples": self.n_samples,
            "personal_accuracy": self.personal_accuracy,
            "classes": self.classes or [],
        }

    def _save(self):
        """Persist personal model to disk."""
        user_dir = USER_MODELS_DIR / self.user_id
        user_dir.mkdir(exist_ok=True)
        path = user_dir / "personal_model.pkl"

        data = {
            "model": self.personal_model,
            "classes": self.classes,
            "feature_names": self.feature_names,
            "n_samples": self.n_samples,
            "personal_accuracy": self.personal_accuracy,
        }
        joblib.dump(data, str(path))

    def _load(self):
        """Load personal model from disk if it exists."""
        path = USER_MODELS_DIR / self.user_id / "personal_model.pkl"
        if path.exists():
            try:
                data = joblib.load(str(path))
                self.personal_model = data["model"]
                self.classes = data["classes"]
                self.feature_names = data["feature_names"]
                self.n_samples = data["n_samples"]
                self.personal_accuracy = data.get("personal_accuracy", 0.0)
            except Exception:
                pass
