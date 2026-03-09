"""Prototypical few-shot EEG emotion personalization.

Per-user emotion recognition with only 5 labeled samples per class.
Computes per-emotion prototypes (mean feature vectors) and classifies
new signals by nearest-prototype distance. No model retraining needed.

Expected improvement: +15-25 points over population-average heuristics.

Calibration protocol:
1. Show user ~30 short emotion-inducing clips (5 per class, ~10 sec each)
2. Record EEG during each clip, extract features
3. Compute per-emotion prototype (mean feature vector)
4. Classify new EEG by nearest cosine similarity to prototype

References:
    Frontiers in Human Neuroscience 2024 — Few-shot EEG survey
    SDA-FSL 2024 — Prototypical Networks with instance-attention for cross-subject EEG
"""
from typing import Dict, List, Optional

import numpy as np

EMOTIONS_6 = ["happy", "sad", "angry", "fear", "surprise", "neutral"]


class FewShotPersonalizer:
    """Prototypical Network-based per-user emotion personalization.

    Works with pre-extracted feature vectors (any dimensionality).
    Stores support examples per emotion, computes prototypes, and
    classifies new samples by nearest cosine similarity.
    """

    def __init__(self, distance_metric: str = "cosine"):
        self._support: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self._prototypes: Dict[str, Dict[str, np.ndarray]] = {}
        self._distance_metric = distance_metric

    def add_support(
        self,
        features: np.ndarray,
        emotion: str,
        user_id: str = "default",
    ) -> Dict:
        """Add a labeled support example for a user.

        Args:
            features: 1D feature vector.
            emotion: Emotion label.
            user_id: User identifier.

        Returns:
            Dict with n_shots per class and adaptation status.
        """
        features = np.asarray(features, dtype=float).ravel()

        if user_id not in self._support:
            self._support[user_id] = {}
        if emotion not in self._support[user_id]:
            self._support[user_id][emotion] = []
        self._support[user_id][emotion].append(features)

        # Recompute prototype
        self._update_prototypes(user_id)

        shots = {e: len(self._support[user_id].get(e, [])) for e in EMOTIONS_6}
        min_shots = min(shots.get(e, 0) for e in EMOTIONS_6 if shots.get(e, 0) > 0) if any(shots.values()) else 0

        return {
            "user_id": user_id,
            "emotion": emotion,
            "shots_per_class": shots,
            "total_support": sum(shots.values()),
            "is_adapted": self.is_adapted(user_id),
            "min_shots": min_shots,
        }

    def classify(
        self,
        features: np.ndarray,
        user_id: str = "default",
    ) -> Dict:
        """Classify a new sample using prototypical matching.

        Args:
            features: 1D feature vector (same dim as support examples).
            user_id: User identifier.

        Returns:
            Dict with predicted emotion, probabilities, confidence,
            and whether personalization is active.
        """
        features = np.asarray(features, dtype=float).ravel()

        prototypes = self._prototypes.get(user_id, {})
        if not prototypes:
            return {
                "emotion": "neutral",
                "probabilities": {e: round(1 / 6, 4) for e in EMOTIONS_6},
                "confidence": "none",
                "personalized": False,
            }

        # Compute distances to each prototype
        distances = {}
        for emotion, proto in prototypes.items():
            if self._distance_metric == "cosine":
                distances[emotion] = self._cosine_similarity(features, proto)
            else:
                distances[emotion] = -float(np.linalg.norm(features - proto))

        # Softmax over similarities
        max_dist = max(distances.values())
        exp_dists = {e: np.exp((d - max_dist) * 5) for e, d in distances.items()}
        total = sum(exp_dists.values()) + 1e-10
        probabilities = {e: float(exp_dists[e] / total) for e in distances}

        best_emotion = max(distances, key=distances.get)
        best_prob = probabilities[best_emotion]

        if best_prob > 0.5:
            confidence = "high"
        elif best_prob > 0.3:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "emotion": best_emotion,
            "probabilities": {e: round(p, 4) for e, p in probabilities.items()},
            "confidence": confidence,
            "personalized": True,
            "best_similarity": round(float(max(distances.values())), 4),
        }

    def is_adapted(self, user_id: str = "default") -> bool:
        """Check if user has enough support examples for personalization.

        Requires at least 2 classes with at least 1 example each.
        """
        protos = self._prototypes.get(user_id, {})
        return len(protos) >= 2

    def get_status(self, user_id: str = "default") -> Dict:
        """Get personalization status for a user."""
        support = self._support.get(user_id, {})
        shots = {e: len(support.get(e, [])) for e in EMOTIONS_6}
        return {
            "is_adapted": self.is_adapted(user_id),
            "n_classes": len(self._prototypes.get(user_id, {})),
            "shots_per_class": shots,
            "total_support": sum(shots.values()),
            "adapted_emotions": list(self._prototypes.get(user_id, {}).keys()),
        }

    def reset(self, user_id: str = "default"):
        """Clear all support data and prototypes for a user."""
        self._support.pop(user_id, None)
        self._prototypes.pop(user_id, None)

    def get_prototypes(self, user_id: str = "default") -> Dict[str, List[float]]:
        """Get computed prototypes (for visualization/export)."""
        protos = self._prototypes.get(user_id, {})
        return {e: proto.tolist() for e, proto in protos.items()}

    # ── Private helpers ──────────────────────────────────────────

    def _update_prototypes(self, user_id: str):
        """Recompute prototypes as mean of support examples."""
        support = self._support.get(user_id, {})
        if user_id not in self._prototypes:
            self._prototypes[user_id] = {}

        for emotion, examples in support.items():
            if examples:
                self._prototypes[user_id][emotion] = np.mean(examples, axis=0)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
