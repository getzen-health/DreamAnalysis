"""Emotional granularity: map VAD (valence-arousal-dominance) to nuanced labels.

Expands EEG emotion output from 6 basic emotions to 27 nuanced affect states
by mapping continuous VAD (Valence-Arousal-Dominance) coordinates to a
comprehensive emotion vocabulary.

Scientific basis:
- IEEE (2025): EEGEmotions-27 dataset -- 62.24% accuracy on 27 categories
- MDPI Mathematics (2025): EEG-RegNet, 95% DEAP via VAD regression
- Dominance (sense of control/agency) adds crucial third dimension
- VAD coordinates from Russell's Circumplex + dimensional emotion theory

Dominance (0-1) reflects perceived control / agency:
    High dominance = in control, empowered (e.g. determined, angry, focused)
    Low dominance  = overwhelmed, helpless  (e.g. fearful, overwhelmed, anxious)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# VAD coordinates for 27 emotion categories
# Format: (valence, arousal, dominance) -- all in [-1,1] or [0,1] depending on dimension
# Valence: -1 (very negative) to +1 (very positive)
# Arousal: 0 (very calm) to 1 (very energetic)
# Dominance: 0 (submissive/overwhelmed) to 1 (dominant/in-control)
EMOTION_VAD_MAP: Dict[str, Tuple[float, float, float]] = {
    # Positive high-arousal
    "excited":      (0.80,  0.90, 0.65),
    "happy":        (0.85,  0.65, 0.70),
    "elated":       (0.90,  0.85, 0.75),
    "enthusiastic": (0.75,  0.80, 0.65),
    "awe":          (0.60,  0.75, 0.30),
    "surprised":    (0.30,  0.80, 0.35),

    # Positive low-arousal
    "content":      (0.70,  0.20, 0.60),
    "calm":         (0.50,  0.10, 0.55),
    "serene":       (0.65,  0.08, 0.50),
    "nostalgic":    (0.30,  0.20, 0.35),
    "grateful":     (0.75,  0.30, 0.55),

    # Neutral
    "neutral":      (0.00,  0.20, 0.50),
    "bored":        (-0.20, 0.10, 0.30),
    "pensive":      (0.10,  0.15, 0.40),

    # Negative high-arousal
    "angry":        (-0.70, 0.85, 0.75),
    "fearful":      (-0.75, 0.85, 0.10),
    "anxious":      (-0.55, 0.80, 0.15),
    "frustrated":   (-0.60, 0.70, 0.50),
    "stressed":     (-0.50, 0.75, 0.25),
    "disgusted":    (-0.65, 0.55, 0.60),

    # Negative low-arousal
    "sad":          (-0.75, 0.15, 0.20),
    "melancholy":   (-0.40, 0.20, 0.35),
    "lonely":       (-0.50, 0.15, 0.15),
    "ashamed":      (-0.60, 0.35, 0.10),
    "hopeless":     (-0.80, 0.10, 0.05),

    # High dominance special
    "proud":        (0.70,  0.50, 0.90),
    "contemptuous": (-0.40, 0.45, 0.80),
}


# ── Standalone helpers (used by emotion_classifier.py and tests) ──────


def _euclidean_distance(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    """3D Euclidean distance between two VAD coordinates."""
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def _distance_to_similarity(distance: float, max_distance: float = 2.5) -> float:
    """Convert Euclidean distance to a 0-1 similarity score.

    max_distance is the theoretical maximum distance in the VAD space
    (valence spans 2, arousal spans 1, dominance spans 1 => sqrt(4+1+1) ~ 2.45).
    """
    return max(0.0, min(1.0, 1.0 - distance / max_distance))


def map_vad_to_granular_emotions(
    valence: float,
    arousal: float,
    dominance: float,
    top_k: int = 3,
) -> List[Dict[str, object]]:
    """Find the k nearest emotions in VAD space by Euclidean distance.

    Args:
        valence:   Emotional valence (-1 to 1, negative to positive).
        arousal:   Activation level (0 to 1, calm to energetic).
        dominance: Sense of control (0 to 1, overwhelmed to in-control).
        top_k:     Number of nearest emotions to return.

    Returns:
        List of dicts sorted by descending similarity:
        [{"emotion": "calm", "similarity": 0.89, "distance": 0.27}, ...]
    """
    if top_k < 1:
        top_k = 1

    point = (float(valence), float(arousal), float(dominance))

    scored: List[Tuple[float, str]] = []
    for label, coord in EMOTION_VAD_MAP.items():
        dist = _euclidean_distance(point, coord)
        scored.append((dist, label))

    scored.sort(key=lambda x: x[0])

    results: List[Dict[str, object]] = []
    for dist, label in scored[:top_k]:
        results.append({
            "emotion": label,
            "similarity": round(_distance_to_similarity(dist), 4),
            "distance": round(dist, 4),
        })

    return results


# ── Dominance estimation from band powers ─────────────────────────────


def estimate_dominance(band_powers: Dict[str, float]) -> float:
    """Estimate dominance (sense of control/agency) from frontal band powers.

    High beta + low theta at frontal = high dominance (in control, assertive).
    High theta + low beta = low dominance (overwhelmed, submissive).

    Args:
        band_powers: Dict from extract_band_powers() -- delta, theta, alpha, beta, etc.

    Returns:
        dominance in [0, 1]
    """
    alpha = max(band_powers.get("alpha", 0.2), 1e-10)
    beta = max(band_powers.get("beta", 0.15), 1e-10)
    theta = max(band_powers.get("theta", 0.15), 1e-10)
    high_beta = max(band_powers.get("high_beta", 0.05), 1e-10)

    eps = 1e-10
    theta_beta_ratio = theta / (beta + eps)

    # Dominance from beta/alpha ratio (high beta = assertive engagement)
    beta_alpha_component = float(np.tanh((beta / (alpha + eps) - 0.5) * 2.0)) * 0.5 + 0.5

    # Dominance from theta/beta (inverse -- high theta = overwhelmed)
    theta_beta_component = float(np.clip(1.0 - theta_beta_ratio * 0.8, 0, 1))

    # High-beta penalty for anxiety (reduces perceived dominance)
    anxiety_penalty = float(np.clip(high_beta / (beta + eps) - 0.3, 0, 0.4))

    dominance = float(np.clip(
        0.50 * beta_alpha_component
        + 0.50 * theta_beta_component
        - anxiety_penalty,
        0.0, 1.0,
    ))
    return dominance


# ── Weighted-distance mapper (class interface) ────────────────────────


def _vad_distance(
    valence: float, arousal: float, dominance: float,
    ev: float, ea: float, ed: float,
    weights: Tuple[float, float, float] = (0.45, 0.35, 0.20),
) -> float:
    """Weighted Euclidean distance in VAD space."""
    wv, wa, wd = weights
    return float(np.sqrt(
        wv * (valence - ev) ** 2
        + wa * (arousal - ea) ** 2
        + wd * (dominance - ed) ** 2
    ))


class EmotionGranularityMapper:
    """Maps VAD coordinates to nuanced emotion vocabulary.

    Finds top-N closest emotions in VAD space and returns them as
    a nuanced affect description.
    """

    def __init__(self, vad_map: Optional[Dict] = None):
        self._map = vad_map or EMOTION_VAD_MAP

    def map(
        self,
        valence: float,
        arousal: float,
        dominance: float = 0.5,
        top_n: int = 3,
    ) -> Dict:
        """Map VAD coordinates to nuanced emotion labels.

        Args:
            valence: -1 (very negative) to +1 (very positive)
            arousal: 0 (calm) to 1 (energetic)
            dominance: 0 (overwhelmed) to 1 (in control)
            top_n: number of top emotions to return

        Returns:
            dict with primary_emotion, nuance_emotions, narrative, vad, all_scores
        """
        # Compute distances to all VAD points
        scores = {}
        for emotion, (ev, ea, ed) in self._map.items():
            dist = _vad_distance(valence, arousal, dominance, ev, ea, ed)
            scores[emotion] = float(1.0 / (dist + 0.1))  # inverse distance = similarity

        # Sort by similarity
        sorted_emotions = sorted(scores.items(), key=lambda x: -x[1])
        top_emotions = sorted_emotions[:top_n]

        primary = top_emotions[0][0]
        nuances = [e for e, _ in top_emotions[1:]]

        # Normalize scores to probabilities
        total = sum(s for _, s in top_emotions)
        top_probs = {e: round(s / total, 3) for e, s in top_emotions}

        # Build narrative
        if len(nuances) >= 2:
            narrative = f"Mostly {primary}, with hints of {nuances[0]} and {nuances[1]}."
        elif len(nuances) == 1:
            narrative = f"Primarily {primary}, with undertones of {nuances[0]}."
        else:
            narrative = f"Predominantly {primary}."

        # VAD zone label
        zone = _get_vad_zone(valence, arousal, dominance)

        return {
            "primary_emotion": primary,
            "nuance_emotions": nuances,
            "narrative": narrative,
            "affect_zone": zone,
            "top_emotions": top_probs,
            "valence": round(float(valence), 3),
            "arousal": round(float(arousal), 3),
            "dominance": round(float(dominance), 3),
            "model_type": "vad_granularity_27",
        }

    def map_from_basic(
        self,
        basic_emotion: str,
        valence: float,
        arousal: float,
        band_powers: Optional[Dict] = None,
        top_n: int = 3,
    ) -> Dict:
        """Convenience: map from existing 6-class output + computed VAD.

        Args:
            basic_emotion: one of happy/sad/angry/fear/surprise/neutral
            valence: from emotion classifier
            arousal: from emotion classifier
            band_powers: optional, used to estimate dominance
            top_n: number of top emotions
        """
        dominance = 0.5
        if band_powers:
            dominance = estimate_dominance(band_powers)
        return self.map(valence, arousal, dominance, top_n)


def _get_vad_zone(valence: float, arousal: float, dominance: float) -> str:
    """Label the VAD octant / quadrant for easy interpretation."""
    v = "positive" if valence >= 0 else "negative"
    a = "high-energy" if arousal >= 0.5 else "low-energy"
    d = "in-control" if dominance >= 0.5 else "overwhelmed"
    return f"{v}, {a}, {d}"


_mapper_instance: Optional[EmotionGranularityMapper] = None


def get_granularity_mapper() -> EmotionGranularityMapper:
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = EmotionGranularityMapper()
    return _mapper_instance
