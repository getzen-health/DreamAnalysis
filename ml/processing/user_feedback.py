"""User Feedback & Online Learning — Personalize models from user corrections.

Problem: Our models work on population averages, but every brain is different.
A prediction that's right for 70% of people might be wrong for YOU specifically.

Solution: Let users correct predictions and use those corrections to fine-tune.

Feedback Types:
1. State correction: "I wasn't in flow, I was just relaxed"
2. Binary feedback: "This prediction was wrong / right"
3. Self-report: "Right now I feel: focused / tired / creative"
4. Session rating: "This session was: productive / mediocre / bad"

Online Learning:
- Collect (features, user_label) pairs from corrections
- After N corrections, fit a lightweight per-user model
- Blend per-user model with global model predictions
- The more feedback, the higher weight on personal model
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FEEDBACK_DIR = Path(__file__).parent.parent / "data" / "user_feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

# Minimum feedback samples before we start using per-user model
MIN_SAMPLES_TO_PERSONALIZE = 15

# Weight of personal model vs global model (grows with more data)
# At MIN_SAMPLES: blend = 0.3 personal + 0.7 global
# At 100 samples:  blend ≈ 0.6 personal + 0.4 global
# At 500 samples:  blend ≈ 0.8 personal + 0.2 global


class FeedbackCollector:
    """Collects and stores user feedback on model predictions."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.feedback_file = FEEDBACK_DIR / f"{user_id}_feedback.jsonl"
        self._feedback_cache: List[Dict] = []
        self._load_feedback()

    def record_state_correction(self, model_name: str,
                                predicted_state: str,
                                corrected_state: str,
                                features: Optional[np.ndarray] = None,
                                context: Optional[Dict] = None):
        """Record when user corrects a model's prediction.

        Args:
            model_name: Which model was wrong (e.g., "flow_state").
            predicted_state: What the model said.
            corrected_state: What the user says is correct.
            features: The 17-dim feature vector at time of prediction.
            context: Additional context (time of day, session duration, etc.).
        """
        entry = {
            "type": "state_correction",
            "timestamp": time.time(),
            "model": model_name,
            "predicted": predicted_state,
            "corrected": corrected_state,
            "was_correct": predicted_state == corrected_state,
            "features": features.tolist() if features is not None else None,
            "context": context or {},
        }
        self._append_feedback(entry)

    def record_binary_feedback(self, model_name: str,
                               predicted_state: str,
                               was_correct: bool,
                               features: Optional[np.ndarray] = None):
        """Record simple right/wrong feedback on a prediction."""
        entry = {
            "type": "binary_feedback",
            "timestamp": time.time(),
            "model": model_name,
            "predicted": predicted_state,
            "was_correct": was_correct,
            "features": features.tolist() if features is not None else None,
        }
        self._append_feedback(entry)

    def record_self_report(self, reported_state: str,
                           model_name: str = "general",
                           features: Optional[np.ndarray] = None,
                           band_powers: Optional[Dict] = None):
        """Record user's self-reported current state.

        This is valuable even without a model prediction — it creates
        supervised training data for personalization.
        """
        entry = {
            "type": "self_report",
            "timestamp": time.time(),
            "model": model_name,
            "reported_state": reported_state,
            "features": features.tolist() if features is not None else None,
            "band_powers": band_powers,
        }
        self._append_feedback(entry)

    def record_session_rating(self, session_id: str, rating: str,
                              notes: Optional[str] = None):
        """Record overall session rating.

        Args:
            rating: "productive", "mediocre", "bad", "relaxing", "stressful"
        """
        entry = {
            "type": "session_rating",
            "timestamp": time.time(),
            "session_id": session_id,
            "rating": rating,
            "notes": notes,
        }
        self._append_feedback(entry)

    def get_feedback_stats(self) -> Dict:
        """Get summary statistics of collected feedback."""
        if not self._feedback_cache:
            return {"total_entries": 0, "models": {}}

        model_stats = {}
        for entry in self._feedback_cache:
            model = entry.get("model", "unknown")
            if model not in model_stats:
                model_stats[model] = {
                    "total": 0, "corrections": 0, "correct": 0, "reports": 0
                }

            model_stats[model]["total"] += 1

            if entry["type"] == "state_correction":
                model_stats[model]["corrections"] += 1
                if entry.get("was_correct"):
                    model_stats[model]["correct"] += 1
            elif entry["type"] == "binary_feedback":
                if entry.get("was_correct"):
                    model_stats[model]["correct"] += 1
            elif entry["type"] == "self_report":
                model_stats[model]["reports"] += 1

        # Compute per-model accuracy where we have feedback.
        # feedback_total = all entries that carry a was_correct signal
        # (state_correction + binary_feedback, i.e. total minus self_reports).
        for model, stats in model_stats.items():
            feedback_total = stats["total"] - stats["reports"]
            if feedback_total > 0:
                stats["user_perceived_accuracy"] = round(
                    stats["correct"] / feedback_total, 3
                )

        return {
            "total_entries": len(self._feedback_cache),
            "models": model_stats,
            "can_personalize": {
                model: stats["total"] >= MIN_SAMPLES_TO_PERSONALIZE
                for model, stats in model_stats.items()
            },
        }

    def get_training_data(self, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract training data for a specific model from feedback.

        Returns:
            (X, y) where X is feature vectors and y is correct labels.
        """
        X = []
        y = []

        for entry in self._feedback_cache:
            if entry.get("model") != model_name:
                continue
            if entry.get("features") is None:
                continue

            features = np.array(entry["features"])

            if entry["type"] == "state_correction":
                X.append(features)
                y.append(entry["corrected"])
            elif entry["type"] == "self_report":
                X.append(features)
                y.append(entry["reported_state"])

        if not X:
            return np.array([]), np.array([])

        return np.array(X), np.array(y)

    def _append_feedback(self, entry: Dict):
        """Append feedback entry to file and cache."""
        self._feedback_cache.append(entry)
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_feedback(self):
        """Load existing feedback from disk."""
        if self.feedback_file.exists():
            with open(self.feedback_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._feedback_cache.append(json.loads(line))


class PersonalizedModel:
    """Lightweight per-user model that learns from feedback.

    Uses nearest-neighbor classification on user's corrected examples.
    Simple, interpretable, and works well with small datasets (15-100 samples).
    """

    def __init__(self, user_id: str, model_name: str):
        self.user_id = user_id
        self.model_name = model_name
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.is_fitted = False
        self.n_samples = 0
        self.classes = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the personal model on user feedback data."""
        if len(X) < MIN_SAMPLES_TO_PERSONALIZE:
            self.is_fitted = False
            return

        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        self.n_samples = len(X)
        self.classes = sorted(set(y))
        self.is_fitted = True

        # Precompute feature statistics for normalization
        self._mean = np.mean(self.X_train, axis=0)
        self._std = np.std(self.X_train, axis=0)
        self._std = np.maximum(self._std, 1e-6)

    def predict(self, features: np.ndarray) -> Optional[Dict]:
        """Predict using personal model (k-NN with k=5).

        Returns None if model isn't fitted yet.
        """
        if not self.is_fitted:
            return None

        features = np.array(features, dtype=float).reshape(1, -1)

        # Normalize
        X_norm = (self.X_train - self._mean) / self._std
        f_norm = (features - self._mean) / self._std

        # Compute distances
        distances = np.sqrt(np.sum((X_norm - f_norm) ** 2, axis=1))

        # K nearest neighbors
        k = min(5, self.n_samples)
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = self.y_train[nearest_idx]
        nearest_distances = distances[nearest_idx]

        # Distance-weighted voting
        weights = 1.0 / (nearest_distances + 1e-6)
        class_scores = {}
        for label, weight in zip(nearest_labels, weights):
            class_scores[label] = class_scores.get(label, 0) + weight

        total_weight = sum(class_scores.values())
        class_probs = {
            c: w / total_weight for c, w in class_scores.items()
        }

        predicted = max(class_probs, key=class_probs.get)
        confidence = class_probs[predicted]

        return {
            "state": predicted,
            "confidence": round(confidence, 3),
            "class_probabilities": {
                c: round(p, 3) for c, p in class_probs.items()
            },
            "n_training_samples": self.n_samples,
        }

    def get_blend_weight(self) -> float:
        """How much to weight personal model vs global model.

        Returns 0.0-0.8 (never fully replaces global model).
        """
        if not self.is_fitted:
            return 0.0

        # Logarithmic growth: more data → more trust, but diminishing returns
        weight = 0.3 + 0.5 * (1 - np.exp(-self.n_samples / 100))
        return min(0.8, weight)


class PersonalizedPipeline:
    """Blends global model predictions with per-user personalized predictions.

    Usage:
        pipeline = PersonalizedPipeline("user123")
        pipeline.update_from_feedback()  # Load latest feedback

        # Get blended prediction
        global_pred = {"state": "flow", "confidence": 0.7}
        features = extract_features(eeg_signal, fs)
        blended = pipeline.blend("flow_state", global_pred, features)
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.feedback = FeedbackCollector(user_id)
        self.personal_models: Dict[str, PersonalizedModel] = {}

    def update_from_feedback(self):
        """Retrain all personal models from latest feedback data."""
        stats = self.feedback.get_feedback_stats()

        for model_name, can_train in stats.get("can_personalize", {}).items():
            if can_train:
                X, y = self.feedback.get_training_data(model_name)
                if len(X) >= MIN_SAMPLES_TO_PERSONALIZE:
                    pm = PersonalizedModel(self.user_id, model_name)
                    pm.fit(X, y)
                    self.personal_models[model_name] = pm

    def blend(self, model_name: str, global_prediction: Dict,
              features: Optional[np.ndarray] = None) -> Dict:
        """Blend global model prediction with personal model.

        Args:
            model_name: Model name (e.g., "flow_state").
            global_prediction: Raw prediction from global model.
            features: Feature vector for personal model prediction.

        Returns:
            Blended prediction dict.
        """
        pm = self.personal_models.get(model_name)

        if pm is None or not pm.is_fitted or features is None:
            global_prediction["personalization"] = "none"
            return global_prediction

        personal_pred = pm.predict(features)
        if personal_pred is None:
            global_prediction["personalization"] = "none"
            return global_prediction

        blend_weight = pm.get_blend_weight()
        global_weight = 1.0 - blend_weight

        # If both agree, boost confidence
        global_state = (
            global_prediction.get("state")
            or global_prediction.get("stage")
            or global_prediction.get("emotion")
        )
        personal_state = personal_pred["state"]

        if global_state == personal_state:
            # Agreement → higher confidence
            global_conf = global_prediction.get("confidence", 0.5)
            personal_conf = personal_pred["confidence"]
            blended_conf = global_weight * global_conf + blend_weight * personal_conf
            blended_conf = min(1.0, blended_conf * 1.1)  # 10% boost for agreement

            result = dict(global_prediction)
            result["confidence"] = round(blended_conf, 3)
            result["personalization"] = "agreement_boost"
            result["personal_model_weight"] = round(blend_weight, 3)
            return result

        # Disagreement → use weighted blend
        global_conf = global_prediction.get("confidence", 0.5)
        personal_conf = personal_pred["confidence"]

        global_score = global_conf * global_weight
        personal_score = personal_conf * blend_weight

        if personal_score > global_score:
            # Personal model wins
            result = dict(global_prediction)
            # Override the state
            for key in ("state", "stage", "emotion"):
                if key in result:
                    result[key] = personal_state
            result["confidence"] = round(personal_conf * blend_weight, 3)
            result["personalization"] = "personal_override"
            result["personal_model_weight"] = round(blend_weight, 3)
            result["global_prediction"] = global_state
            return result
        else:
            # Global model wins but note disagreement
            result = dict(global_prediction)
            result["confidence"] = round(global_conf * global_weight, 3)
            result["personalization"] = "global_preferred"
            result["personal_model_weight"] = round(blend_weight, 3)
            result["personal_suggestion"] = personal_state
            return result

    def get_personalization_status(self) -> Dict:
        """Get status of personalization for all models."""
        stats = self.feedback.get_feedback_stats()

        model_status = {}
        for model_name in ["sleep_staging", "emotion", "flow_state",
                           "creativity", "memory_encoding", "dream_detection"]:
            pm = self.personal_models.get(model_name)
            model_feedback = stats.get("models", {}).get(model_name, {})

            model_status[model_name] = {
                "feedback_count": model_feedback.get("total", 0),
                "is_personalized": pm is not None and pm.is_fitted,
                "personal_model_weight": pm.get_blend_weight() if pm else 0.0,
                "samples_until_personalization": max(
                    0,
                    MIN_SAMPLES_TO_PERSONALIZE - model_feedback.get("total", 0)
                ),
                "user_perceived_accuracy": model_feedback.get("user_perceived_accuracy"),
            }

        return {
            "user_id": self.user_id,
            "total_feedback": stats["total_entries"],
            "models": model_status,
        }
