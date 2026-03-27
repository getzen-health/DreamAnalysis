"""
Few-shot personalization for EEG emotion classification.
Adapts a global model to individual users using 5-10 labeled samples.

Methods:
1. Prototypical adaptation: compute per-class prototypes from user's labeled samples,
   use nearest-prototype classification with global model as fallback
2. Feature normalization: z-score user features against their personal baseline
3. Confidence-weighted blending: blend personal predictions with global model
   based on how many personal samples exist (more samples = more personal weight)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]


@dataclass
class LabeledSample:
    features: np.ndarray  # extracted EEG feature vector
    label: str            # emotion label
    timestamp: str        # ISO timestamp


@dataclass
class PersonalModel:
    user_id: str
    prototypes: Dict[str, np.ndarray]  # emotion -> mean feature vector
    prototype_counts: Dict[str, int]   # emotion -> sample count
    feature_mean: Optional[np.ndarray] = None  # personal baseline mean
    feature_std: Optional[np.ndarray] = None   # personal baseline std
    total_samples: int = 0

    def confidence(self) -> float:
        """0-1 confidence based on sample count. Reaches 0.8 at 30 samples."""
        return min(0.8, self.total_samples / 37.5)


class FewShotPersonalizer:
    """Adapts global model predictions using few labeled user samples."""

    def __init__(self, user_id: str, blend_threshold: int = 5):
        self.user_id = user_id
        self.blend_threshold = blend_threshold  # min samples before blending
        self.model = PersonalModel(
            user_id=user_id, prototypes={}, prototype_counts={}
        )
        self.samples: List[LabeledSample] = []

    def add_sample(
        self, features: np.ndarray, label: str, timestamp: str = ""
    ) -> None:
        """Add a labeled sample and update prototypes."""
        self.samples.append(
            LabeledSample(features=features, label=label, timestamp=timestamp)
        )
        # Update prototype for this label (running mean)
        if label not in self.model.prototypes:
            self.model.prototypes[label] = features.copy()
            self.model.prototype_counts[label] = 1
        else:
            n = self.model.prototype_counts[label]
            self.model.prototypes[label] = (
                self.model.prototypes[label] * n + features
            ) / (n + 1)
            self.model.prototype_counts[label] = n + 1
        self.model.total_samples += 1
        # Update personal baseline stats
        all_features = np.stack([s.features for s in self.samples])
        self.model.feature_mean = all_features.mean(axis=0)
        self.model.feature_std = all_features.std(axis=0) + 1e-8

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalize features against personal baseline."""
        if self.model.feature_mean is None:
            return features
        return (features - self.model.feature_mean) / self.model.feature_std

    def predict_personal(
        self, features: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """Nearest-prototype classification. Returns probs dict or None if
        not enough data."""
        if self.model.total_samples < self.blend_threshold:
            return None
        if not self.model.prototypes:
            return None
        # Compute distances to each prototype
        distances = {}
        for label, proto in self.model.prototypes.items():
            distances[label] = float(np.linalg.norm(features - proto))
        # Convert distances to probabilities (softmax of negative distances)
        labels = list(distances.keys())
        dists = np.array([distances[l] for l in labels])
        exp_neg = np.exp(-dists)
        probs_arr = exp_neg / (exp_neg.sum() + 1e-10)
        probs = {l: 0.0 for l in EMOTIONS}
        for l, p in zip(labels, probs_arr):
            if l in probs:
                probs[l] = float(p)
        return probs

    def blend(
        self, global_probs: Dict[str, float], features: np.ndarray
    ) -> Dict[str, float]:
        """Blend global model predictions with personal predictions."""
        personal = self.predict_personal(features)
        if personal is None:
            return global_probs  # not enough personal data
        alpha = self.model.confidence()  # 0-0.8 based on sample count
        blended = {}
        for e in EMOTIONS:
            g = global_probs.get(e, 1 / 6)
            p = personal.get(e, 1 / 6)
            blended[e] = (1 - alpha) * g + alpha * p
        # Renormalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended

    def get_status(self) -> Dict:
        """Return personalization status for API."""
        return {
            "user_id": self.user_id,
            "total_samples": self.model.total_samples,
            "classes_seen": list(self.model.prototypes.keys()),
            "samples_per_class": dict(self.model.prototype_counts),
            "confidence": self.model.confidence(),
            "is_active": self.model.total_samples >= self.blend_threshold,
        }

    def save(self, path: str) -> None:
        """Save personal model to numpy archive."""
        np.savez(
            path,
            user_id=self.user_id,
            prototypes={k: v for k, v in self.model.prototypes.items()},
            prototype_counts=dict(self.model.prototype_counts),
            feature_mean=self.model.feature_mean,
            feature_std=self.model.feature_std,
            total_samples=self.model.total_samples,
        )
