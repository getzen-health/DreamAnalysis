"""Ambient Sound Environment Profiler.

Classifies ambient sound environments from pre-computed audio features and
correlates sound environments with emotional states tracked by the EEG pipeline.

This model does NOT process raw audio. It works with pre-computed features:
  - spectral_centroid: centre of mass of the spectrum (Hz)
  - spectral_energy: total signal energy (RMS-like)
  - zero_crossing_rate: rate of sign changes in the time-domain signal
  - mfcc: mel-frequency cepstral coefficients (13 floats)
  - spectral_bandwidth: spread of the spectrum around the centroid
  - spectral_rolloff: frequency below which 85% of energy is concentrated

Sound environment categories:
  silence      -- near-zero energy, minimal spectral content
  nature       -- birds, wind, rain: moderate centroid, low ZCR, organic variation
  urban        -- traffic, construction: high energy, broadband, high ZCR
  social       -- speech, crowd: mid-range centroid, moderate ZCR, speech-band energy
  music        -- tonal structure: high spectral rolloff, low ZCR relative to energy
  indoor       -- HVAC, appliances: low centroid, steady energy, narrow bandwidth
  white_noise  -- flat spectrum: high bandwidth, centroid near Nyquist/2

Uses numpy/scipy only -- no external audio ML libraries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOUND_CATEGORIES = [
    "silence",
    "nature",
    "urban",
    "social",
    "music",
    "indoor",
    "white_noise",
]

# Feature-based profile templates for each category.
# Each entry: (spectral_centroid_norm, energy_norm, zcr_norm, bandwidth_norm, rolloff_norm)
# Values in [0, 1] — normalised relative to typical ranges.
_CATEGORY_PROFILES: Dict[str, Dict[str, float]] = {
    "silence": {
        "centroid": 0.05,
        "energy": 0.02,
        "zcr": 0.05,
        "bandwidth": 0.05,
        "rolloff": 0.05,
    },
    "nature": {
        "centroid": 0.35,
        "energy": 0.30,
        "zcr": 0.20,
        "bandwidth": 0.40,
        "rolloff": 0.45,
    },
    "urban": {
        "centroid": 0.55,
        "energy": 0.75,
        "zcr": 0.70,
        "bandwidth": 0.65,
        "rolloff": 0.60,
    },
    "social": {
        "centroid": 0.45,
        "energy": 0.50,
        "zcr": 0.45,
        "bandwidth": 0.35,
        "rolloff": 0.50,
    },
    "music": {
        "centroid": 0.50,
        "energy": 0.55,
        "zcr": 0.25,
        "bandwidth": 0.50,
        "rolloff": 0.70,
    },
    "indoor": {
        "centroid": 0.20,
        "energy": 0.25,
        "zcr": 0.15,
        "bandwidth": 0.20,
        "rolloff": 0.25,
    },
    "white_noise": {
        "centroid": 0.50,
        "energy": 0.60,
        "zcr": 0.65,
        "bandwidth": 0.90,
        "rolloff": 0.85,
    },
}

# Default emotion labels for correlation tracking
EMOTION_LABELS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_sound_features(
    spectral_centroid: float,
    spectral_energy: float,
    zero_crossing_rate: float,
    mfcc: Optional[List[float]] = None,
    spectral_bandwidth: Optional[float] = None,
    spectral_rolloff: Optional[float] = None,
    sample_rate: float = 22050.0,
) -> Dict[str, Any]:
    """Normalise and bundle raw audio features into a feature dict.

    All inputs are scalar pre-computed values (not raw audio).

    Args:
        spectral_centroid: Centre of spectral mass in Hz.
        spectral_energy: RMS energy of the signal.
        zero_crossing_rate: Fraction of sign-change frames.
        mfcc: 13-element MFCC vector (optional).
        spectral_bandwidth: Spread of spectrum in Hz (optional).
        spectral_rolloff: Roll-off frequency in Hz (optional).
        sample_rate: Audio sample rate used during feature extraction.

    Returns:
        Dict with normalised features ready for classification.
    """
    nyquist = sample_rate / 2.0

    # Normalise to [0, 1] relative to plausible ranges
    centroid_norm = float(np.clip(spectral_centroid / nyquist, 0.0, 1.0))
    energy_norm = float(np.clip(spectral_energy / 1.0, 0.0, 1.0))  # assume max RMS ~1.0
    zcr_norm = float(np.clip(zero_crossing_rate / 0.5, 0.0, 1.0))  # max ZCR ~0.5

    bw_norm = 0.5  # default when not provided
    if spectral_bandwidth is not None:
        bw_norm = float(np.clip(spectral_bandwidth / nyquist, 0.0, 1.0))

    rolloff_norm = 0.5
    if spectral_rolloff is not None:
        rolloff_norm = float(np.clip(spectral_rolloff / nyquist, 0.0, 1.0))

    mfcc_vec = list(mfcc) if mfcc is not None else [0.0] * 13

    return {
        "centroid": centroid_norm,
        "energy": energy_norm,
        "zcr": zcr_norm,
        "bandwidth": bw_norm,
        "rolloff": rolloff_norm,
        "mfcc": mfcc_vec,
        "raw": {
            "spectral_centroid": float(spectral_centroid),
            "spectral_energy": float(spectral_energy),
            "zero_crossing_rate": float(zero_crossing_rate),
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "sample_rate": sample_rate,
        },
    }


def classify_sound_environment(
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify the ambient sound environment from normalised features.

    Uses cosine-distance matching against category profile templates.

    Args:
        features: Dict from ``compute_sound_features`` (must have centroid,
                  energy, zcr, bandwidth, rolloff keys).

    Returns:
        Dict with ``category``, ``confidence``, ``scores`` (per-category),
        and ``feature_summary``.
    """
    feat_vec = np.array([
        features["centroid"],
        features["energy"],
        features["zcr"],
        features["bandwidth"],
        features["rolloff"],
    ], dtype=float)

    # Silence short-circuit: if energy is near zero, classify as silence
    if features["energy"] < 0.03:
        scores = {cat: 0.0 for cat in SOUND_CATEGORIES}
        scores["silence"] = 1.0
        return {
            "category": "silence",
            "confidence": 1.0,
            "scores": scores,
            "feature_summary": _feature_summary(features),
        }

    # Compute similarity to each profile using negative Euclidean distance
    raw_scores: Dict[str, float] = {}
    for cat, profile in _CATEGORY_PROFILES.items():
        profile_vec = np.array([
            profile["centroid"],
            profile["energy"],
            profile["zcr"],
            profile["bandwidth"],
            profile["rolloff"],
        ], dtype=float)
        dist = float(np.linalg.norm(feat_vec - profile_vec))
        # Convert distance to similarity: closer = higher score
        raw_scores[cat] = float(np.exp(-dist * 3.0))

    # Softmax normalisation
    total = sum(raw_scores.values())
    if total < 1e-12:
        total = 1e-12
    scores = {cat: raw_scores[cat] / total for cat in SOUND_CATEGORIES}

    best_cat = max(scores, key=scores.get)  # type: ignore[arg-type]
    confidence = scores[best_cat]

    return {
        "category": best_cat,
        "confidence": float(confidence),
        "scores": scores,
        "feature_summary": _feature_summary(features),
    }


def correlate_sound_emotion(
    sound_records: List[Dict[str, Any]],
    emotion_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute correlation between sound environments and emotion states.

    Each sound record must have ``category`` (str) and ``timestamp`` (float).
    Each emotion record must have ``emotion`` (str), ``valence`` (float),
    ``arousal`` (float), and ``timestamp`` (float).

    Pairs are matched by nearest timestamp within a 30-second window.

    Args:
        sound_records: List of sound classification results with timestamps.
        emotion_records: List of emotion results with timestamps.

    Returns:
        Dict with per-category emotion statistics and overall correlation matrix.
    """
    if not sound_records or not emotion_records:
        return {
            "paired_count": 0,
            "per_category": {},
            "valence_by_category": {},
            "arousal_by_category": {},
        }

    # Build sorted arrays for matching
    s_times = np.array([r["timestamp"] for r in sound_records])
    e_times = np.array([r["timestamp"] for r in emotion_records])

    # Pair each sound record with the nearest emotion record (within 30s)
    paired: List[Tuple[Dict, Dict]] = []
    for i, sr in enumerate(sound_records):
        diffs = np.abs(e_times - s_times[i])
        nearest_idx = int(np.argmin(diffs))
        if diffs[nearest_idx] <= 30.0:
            paired.append((sr, emotion_records[nearest_idx]))

    if not paired:
        return {
            "paired_count": 0,
            "per_category": {},
            "valence_by_category": {},
            "arousal_by_category": {},
        }

    # Aggregate per category
    per_cat: Dict[str, Dict[str, List[float]]] = {}
    for sr, er in paired:
        cat = sr["category"]
        if cat not in per_cat:
            per_cat[cat] = {"valence": [], "arousal": [], "emotions": []}
        per_cat[cat]["valence"].append(float(er.get("valence", 0.0)))
        per_cat[cat]["arousal"].append(float(er.get("arousal", 0.0)))
        per_cat[cat]["emotions"].append(er.get("emotion", "neutral"))

    valence_by_cat: Dict[str, Dict[str, float]] = {}
    arousal_by_cat: Dict[str, Dict[str, float]] = {}
    per_category_summary: Dict[str, Dict[str, Any]] = {}

    for cat, data in per_cat.items():
        v = np.array(data["valence"])
        a = np.array(data["arousal"])
        valence_by_cat[cat] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }
        arousal_by_cat[cat] = {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }
        # Dominant emotion in this category
        emotion_counts: Dict[str, int] = {}
        for e in data["emotions"]:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        dominant = max(emotion_counts, key=emotion_counts.get)  # type: ignore[arg-type]
        per_category_summary[cat] = {
            "n_samples": len(data["valence"]),
            "dominant_emotion": dominant,
            "emotion_distribution": emotion_counts,
            "mean_valence": float(np.mean(v)),
            "mean_arousal": float(np.mean(a)),
        }

    return {
        "paired_count": len(paired),
        "per_category": per_category_summary,
        "valence_by_category": valence_by_cat,
        "arousal_by_category": arousal_by_cat,
    }


def generate_sound_insights(
    correlation_data: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate actionable insights from sound-emotion correlation data.

    Args:
        correlation_data: Output of ``correlate_sound_emotion``.
        history: Optional list of sound classification records over time
                 (for transition analysis).

    Returns:
        Dict with best_environments, worst_environments, transition_effects,
        and recommendations.
    """
    per_cat = correlation_data.get("per_category", {})

    if not per_cat:
        return {
            "best_environments": [],
            "worst_environments": [],
            "transition_effects": [],
            "recommendations": ["Collect more paired sound-emotion data for insights."],
        }

    # Rank environments by mean valence (higher = better mood)
    ranked = sorted(
        per_cat.items(),
        key=lambda x: x[1].get("mean_valence", 0.0),
        reverse=True,
    )

    best = [
        {"category": cat, "mean_valence": info["mean_valence"], "mean_arousal": info["mean_arousal"]}
        for cat, info in ranked
        if info.get("mean_valence", 0.0) > 0.0
    ]
    worst = [
        {"category": cat, "mean_valence": info["mean_valence"], "mean_arousal": info["mean_arousal"]}
        for cat, info in reversed(ranked)
        if info.get("mean_valence", 0.0) <= 0.0
    ]

    # Transition analysis (if history available)
    transitions: List[Dict[str, Any]] = []
    if history and len(history) >= 2:
        for i in range(1, len(history)):
            prev_cat = history[i - 1].get("category", "unknown")
            curr_cat = history[i].get("category", "unknown")
            if prev_cat != curr_cat:
                transitions.append({
                    "from": prev_cat,
                    "to": curr_cat,
                    "index": i,
                })

    # Aggregate transition counts
    transition_effects: List[Dict[str, Any]] = []
    if transitions:
        trans_counts: Dict[str, int] = {}
        for t in transitions:
            key = f"{t['from']} -> {t['to']}"
            trans_counts[key] = trans_counts.get(key, 0) + 1
        transition_effects = [
            {"transition": k, "count": v}
            for k, v in sorted(trans_counts.items(), key=lambda x: -x[1])
        ]

    # Recommendations
    recommendations: List[str] = []
    if best:
        top = best[0]["category"]
        recommendations.append(
            f"Your best environment is '{top}' — seek it out for focus and positive mood."
        )
    if worst:
        bottom = worst[0]["category"]
        recommendations.append(
            f"The '{bottom}' environment correlates with lower mood — consider noise-cancelling headphones."
        )
    if not best and not worst:
        recommendations.append("Not enough data to determine environment preferences yet.")

    return {
        "best_environments": best,
        "worst_environments": worst,
        "transition_effects": transition_effects,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _feature_summary(features: Dict[str, Any]) -> Dict[str, float]:
    """Extract a concise summary of the normalised features."""
    return {
        "centroid": features.get("centroid", 0.0),
        "energy": features.get("energy", 0.0),
        "zcr": features.get("zcr", 0.0),
        "bandwidth": features.get("bandwidth", 0.0),
        "rolloff": features.get("rolloff", 0.0),
    }
