"""Longitudinal emotional genome -- lifetime identity-level map of emotional architecture.

Computes a user's core emotional traits (Big Five-like but for emotions),
generates a unique emotional DNA fingerprint, detects life chapters from
longitudinal data, and tracks emotional evolution across years.

Trait dimensions:
    - emotional_intensity:  How strongly emotions are felt (amplitude)
    - emotional_stability:  How much emotion varies day-to-day (inverse variance)
    - recovery_speed:       How quickly emotions return to baseline after perturbation
    - social_sensitivity:   How much social context modulates emotional state
    - novelty_reactivity:   How much novel/unfamiliar contexts shift emotion
    - threat_sensitivity:   How strongly negative/threat contexts drive emotion
    - positive_bias:        Baseline skew toward positive vs negative emotion

Issue #444.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIT_NAMES: Tuple[str, ...] = (
    "emotional_intensity",
    "emotional_stability",
    "recovery_speed",
    "social_sensitivity",
    "novelty_reactivity",
    "threat_sensitivity",
    "positive_bias",
)

# Minimum number of data points required for meaningful analysis.
_MIN_SAMPLES = 5
# Minimum samples for chapter detection (need enough to form segments).
_MIN_CHAPTER_SAMPLES = 10
# Default number of chapters to attempt detection for.
_MAX_CHAPTERS = 5
# Fingerprint vector dimensionality.
_FINGERPRINT_DIM = 16


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmotionSample:
    """A single longitudinal emotion measurement."""

    timestamp: float           # Unix epoch seconds
    valence: float = 0.0      # -1 .. +1
    arousal: float = 0.5      # 0 .. 1
    stress: float = 0.3       # 0 .. 1
    energy: float = 0.5       # 0 .. 1
    social_context: float = 0.0   # -1 (isolated) .. +1 (social)
    novelty_context: float = 0.0  # 0 (familiar) .. 1 (novel)
    threat_context: float = 0.0   # 0 (safe) .. 1 (threatening)


@dataclass
class EmotionalTraits:
    """Core emotional trait profile."""

    emotional_intensity: float = 0.5
    emotional_stability: float = 0.5
    recovery_speed: float = 0.5
    social_sensitivity: float = 0.5
    novelty_reactivity: float = 0.5
    threat_sensitivity: float = 0.5
    positive_bias: float = 0.5

    def to_array(self) -> np.ndarray:
        return np.array([
            self.emotional_intensity,
            self.emotional_stability,
            self.recovery_speed,
            self.social_sensitivity,
            self.novelty_reactivity,
            self.threat_sensitivity,
            self.positive_bias,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "EmotionalTraits":
        arr_clipped = np.clip(arr, 0.0, 1.0)
        return cls(
            emotional_intensity=float(arr_clipped[0]),
            emotional_stability=float(arr_clipped[1]),
            recovery_speed=float(arr_clipped[2]),
            social_sensitivity=float(arr_clipped[3]),
            novelty_reactivity=float(arr_clipped[4]),
            threat_sensitivity=float(arr_clipped[5]),
            positive_bias=float(arr_clipped[6]),
        )


@dataclass
class LifeChapter:
    """A detected emotional era within the longitudinal data."""

    chapter_index: int
    start_timestamp: float
    end_timestamp: float
    dominant_valence: float
    mean_arousal: float
    mean_stress: float
    mean_energy: float
    sample_count: int
    label: str = ""


@dataclass
class EvolutionSnapshot:
    """Trait values at a specific point in time."""

    timestamp: float
    traits: EmotionalTraits


@dataclass
class GenomeProfile:
    """Complete emotional genome result."""

    user_id: str
    traits: EmotionalTraits
    fingerprint: List[float]
    chapters: List[LifeChapter]
    evolution: List[EvolutionSnapshot]
    computed_at: float = field(default_factory=time.time)
    sample_count: int = 0


# ---------------------------------------------------------------------------
# Trait computation
# ---------------------------------------------------------------------------

def compute_emotional_traits(samples: List[EmotionSample]) -> EmotionalTraits:
    """Estimate core emotional traits from longitudinal emotion data.

    Uses statistical properties of valence/arousal/stress/energy over time,
    plus contextual sensitivity measures.

    Args:
        samples: Chronologically ordered emotion samples.

    Returns:
        EmotionalTraits with each dimension in [0, 1].
    """
    if len(samples) < _MIN_SAMPLES:
        logger.warning(
            "compute_emotional_traits requires >= %d samples; returning defaults.",
            _MIN_SAMPLES,
        )
        return EmotionalTraits()

    valences = np.array([s.valence for s in samples], dtype=np.float64)
    arousals = np.array([s.arousal for s in samples], dtype=np.float64)
    stresses = np.array([s.stress for s in samples], dtype=np.float64)
    energies = np.array([s.energy for s in samples], dtype=np.float64)

    # -- emotional_intensity: mean absolute deviation of valence + mean arousal
    val_mad = float(np.mean(np.abs(valences - np.mean(valences))))
    mean_arousal = float(np.mean(arousals))
    emotional_intensity = float(np.clip(0.5 * val_mad / 0.5 + 0.5 * mean_arousal, 0, 1))

    # -- emotional_stability: inverse of valence variance (scaled)
    val_var = float(np.var(valences))
    # Map variance [0, 0.5] -> stability [1, 0]
    emotional_stability = float(np.clip(1.0 - val_var / 0.5, 0, 1))

    # -- recovery_speed: measured by autocorrelation decay of valence deviations
    deviations = valences - np.mean(valences)
    if len(deviations) > 2 and np.std(deviations) > 1e-9:
        # Lag-1 autocorrelation; high autocorr = slow recovery
        autocorr_1 = float(np.corrcoef(deviations[:-1], deviations[1:])[0, 1])
        if np.isnan(autocorr_1):
            autocorr_1 = 0.0
        # Map autocorr [-1, 1] -> recovery speed [1, 0]
        recovery_speed = float(np.clip(0.5 * (1.0 - autocorr_1), 0, 1))
    else:
        recovery_speed = 0.5

    # -- social_sensitivity: correlation between social_context and valence shift
    social_ctxs = np.array([s.social_context for s in samples], dtype=np.float64)
    social_sensitivity = _compute_sensitivity(social_ctxs, valences)

    # -- novelty_reactivity: correlation between novelty_context and arousal shift
    novelty_ctxs = np.array([s.novelty_context for s in samples], dtype=np.float64)
    novelty_reactivity = _compute_sensitivity(novelty_ctxs, arousals)

    # -- threat_sensitivity: correlation between threat_context and stress response
    threat_ctxs = np.array([s.threat_context for s in samples], dtype=np.float64)
    threat_sensitivity = _compute_sensitivity(threat_ctxs, stresses)

    # -- positive_bias: fraction of time valence is positive, scaled
    positive_frac = float(np.mean(valences > 0))
    positive_bias = float(np.clip(positive_frac, 0, 1))

    return EmotionalTraits(
        emotional_intensity=round(emotional_intensity, 4),
        emotional_stability=round(emotional_stability, 4),
        recovery_speed=round(recovery_speed, 4),
        social_sensitivity=round(social_sensitivity, 4),
        novelty_reactivity=round(novelty_reactivity, 4),
        threat_sensitivity=round(threat_sensitivity, 4),
        positive_bias=round(positive_bias, 4),
    )


def _compute_sensitivity(context: np.ndarray, response: np.ndarray) -> float:
    """Compute sensitivity as absolute correlation between context and response.

    Returns a value in [0, 1].
    """
    if len(context) < 3:
        return 0.5
    std_ctx = float(np.std(context))
    std_resp = float(np.std(response))
    if std_ctx < 1e-9 or std_resp < 1e-9:
        return 0.5
    corr = float(np.corrcoef(context, response)[0, 1])
    if np.isnan(corr):
        return 0.5
    return float(np.clip(abs(corr), 0, 1))


# ---------------------------------------------------------------------------
# Fingerprint generation
# ---------------------------------------------------------------------------

def generate_emotional_fingerprint(
    traits: EmotionalTraits,
    samples: List[EmotionSample],
) -> List[float]:
    """Generate a unique emotional DNA fingerprint vector.

    Combines trait values with distributional statistics to create a
    signature that distinguishes this user from the population.

    Args:
        traits: Pre-computed emotional traits.
        samples: Longitudinal emotion samples.

    Returns:
        A list of floats of length _FINGERPRINT_DIM.
    """
    trait_arr = traits.to_array()  # 7 dims

    if len(samples) < _MIN_SAMPLES:
        # Pad with zeros for the remaining dimensions.
        fp = list(trait_arr) + [0.0] * (_FINGERPRINT_DIM - len(trait_arr))
        return [round(v, 6) for v in fp[:_FINGERPRINT_DIM]]

    valences = np.array([s.valence for s in samples], dtype=np.float64)
    arousals = np.array([s.arousal for s in samples], dtype=np.float64)
    stresses = np.array([s.stress for s in samples], dtype=np.float64)
    energies = np.array([s.energy for s in samples], dtype=np.float64)

    # Statistical shape features
    val_skew = _safe_skewness(valences)
    val_kurt = _safe_kurtosis(valences)
    arousal_range = float(np.ptp(arousals))
    stress_trend = _linear_trend(stresses)
    energy_trend = _linear_trend(energies)

    # Valence-arousal covariance (how arousal and valence co-vary)
    if np.std(valences) > 1e-9 and np.std(arousals) > 1e-9:
        va_corr = float(np.corrcoef(valences, arousals)[0, 1])
        if np.isnan(va_corr):
            va_corr = 0.0
    else:
        va_corr = 0.0

    # Stress-energy anti-correlation
    if np.std(stresses) > 1e-9 and np.std(energies) > 1e-9:
        se_corr = float(np.corrcoef(stresses, energies)[0, 1])
        if np.isnan(se_corr):
            se_corr = 0.0
    else:
        se_corr = 0.0

    extra = [val_skew, val_kurt, arousal_range, stress_trend,
             energy_trend, va_corr, se_corr]

    # Combine: 7 traits + 7 distributional + pad/truncate to _FINGERPRINT_DIM
    combined = list(trait_arr) + extra
    # Ensure fixed length
    if len(combined) < _FINGERPRINT_DIM:
        combined += [0.0] * (_FINGERPRINT_DIM - len(combined))
    combined = combined[:_FINGERPRINT_DIM]

    # L2-normalise for distance comparisons
    norm = float(np.linalg.norm(combined))
    if norm > 1e-9:
        combined = [v / norm for v in combined]

    return [round(v, 6) for v in combined]


def _safe_skewness(arr: np.ndarray) -> float:
    """Compute skewness without scipy dependency."""
    n = len(arr)
    if n < 3:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std < 1e-9:
        return 0.0
    m3 = float(np.mean((arr - mean) ** 3))
    return m3 / (std ** 3)


def _safe_kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis without scipy dependency."""
    n = len(arr)
    if n < 4:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std < 1e-9:
        return 0.0
    m4 = float(np.mean((arr - mean) ** 4))
    return m4 / (std ** 4) - 3.0


def _linear_trend(arr: np.ndarray) -> float:
    """Compute normalised linear trend (slope) of a 1-D array.

    Returns slope scaled to roughly [-1, 1].
    """
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = np.mean(x)
    y_mean = np.mean(arr)
    denom = float(np.sum((x - x_mean) ** 2))
    if denom < 1e-9:
        return 0.0
    slope = float(np.sum((x - x_mean) * (arr - y_mean)) / denom)
    # Normalise by array length so slope is scale-independent.
    return float(np.clip(slope * n, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Life chapter detection
# ---------------------------------------------------------------------------

def detect_life_chapters(
    samples: List[EmotionSample],
    max_chapters: int = _MAX_CHAPTERS,
) -> List[LifeChapter]:
    """Detect distinct emotional eras from longitudinal data.

    Uses a simple change-point detection approach: divides the sorted
    timeline into segments where the mean valence/arousal shift
    significantly between adjacent windows.

    Args:
        samples: Chronologically ordered emotion samples.
        max_chapters: Maximum number of chapters to detect.

    Returns:
        List of LifeChapter objects.
    """
    if len(samples) < _MIN_CHAPTER_SAMPLES:
        if not samples:
            return []
        # Return a single chapter spanning all data.
        return [_build_chapter(samples, 0)]

    # Sort by timestamp (should already be sorted, but be safe).
    sorted_samples = sorted(samples, key=lambda s: s.timestamp)
    n = len(sorted_samples)

    # Compute running valence for change-point detection.
    valences = np.array([s.valence for s in sorted_samples], dtype=np.float64)

    # Find change points via cumulative sum method.
    change_points = _detect_change_points(valences, max_chapters - 1)

    # Build chapter boundaries.
    boundaries = [0] + sorted(change_points) + [n]
    chapters: List[LifeChapter] = []

    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        segment = sorted_samples[start_idx:end_idx]
        if segment:
            chapter = _build_chapter(segment, i)
            chapters.append(chapter)

    return chapters


def _detect_change_points(values: np.ndarray, max_points: int) -> List[int]:
    """Detect change points using binary segmentation on cumulative sum.

    Returns indices where significant mean shifts occur.
    """
    n = len(values)
    if n < 4 or max_points <= 0:
        return []

    mean_val = np.mean(values)
    cusum = np.cumsum(values - mean_val)

    # Find the point of maximum deviation from expected cumulative sum.
    max_idx = int(np.argmax(np.abs(cusum)))

    # Check significance: deviation must exceed threshold.
    deviation = abs(cusum[max_idx])
    threshold = 0.5 * np.std(values) * math.sqrt(n)

    if deviation < threshold or max_idx <= 1 or max_idx >= n - 2:
        return []

    points = [max_idx]

    # Recursively find change points in sub-segments.
    if max_points > 1:
        left_points = _detect_change_points(values[:max_idx], max_points // 2)
        right_points = _detect_change_points(values[max_idx:], max_points - max_points // 2)
        points.extend(left_points)
        points.extend([p + max_idx for p in right_points])

    # Deduplicate and limit.
    points = sorted(set(points))
    return points[:max_points]


def _build_chapter(
    segment: List[EmotionSample],
    index: int,
) -> LifeChapter:
    """Build a LifeChapter from a segment of samples."""
    valences = [s.valence for s in segment]
    arousals = [s.arousal for s in segment]
    stresses = [s.stress for s in segment]
    energies = [s.energy for s in segment]

    mean_val = float(np.mean(valences))
    mean_aro = float(np.mean(arousals))
    mean_str = float(np.mean(stresses))
    mean_eng = float(np.mean(energies))

    # Auto-generate label based on dominant emotional signature.
    if mean_val > 0.3 and mean_str < 0.4:
        label = "positive-stable"
    elif mean_val < -0.2 and mean_str > 0.5:
        label = "difficult-period"
    elif mean_aro > 0.6:
        label = "high-activation"
    elif mean_aro < 0.3:
        label = "low-energy"
    else:
        label = "neutral-baseline"

    return LifeChapter(
        chapter_index=index,
        start_timestamp=segment[0].timestamp,
        end_timestamp=segment[-1].timestamp,
        dominant_valence=round(mean_val, 4),
        mean_arousal=round(mean_aro, 4),
        mean_stress=round(mean_str, 4),
        mean_energy=round(mean_eng, 4),
        sample_count=len(segment),
        label=label,
    )


# ---------------------------------------------------------------------------
# Evolution tracking
# ---------------------------------------------------------------------------

def track_emotional_evolution(
    samples: List[EmotionSample],
    window_size: int = 20,
    step_size: int = 10,
) -> List[EvolutionSnapshot]:
    """Track how emotional traits change over time using a sliding window.

    Args:
        samples: Chronologically ordered emotion samples.
        window_size: Number of samples per window for trait estimation.
        step_size: Number of samples to advance between windows.

    Returns:
        List of EvolutionSnapshot objects tracking trait changes.
    """
    if len(samples) < _MIN_SAMPLES:
        return []

    sorted_samples = sorted(samples, key=lambda s: s.timestamp)
    snapshots: List[EvolutionSnapshot] = []

    # If we have fewer samples than window_size, compute a single snapshot.
    if len(sorted_samples) < window_size:
        traits = compute_emotional_traits(sorted_samples)
        mid_ts = sorted_samples[len(sorted_samples) // 2].timestamp
        snapshots.append(EvolutionSnapshot(timestamp=mid_ts, traits=traits))
        return snapshots

    idx = 0
    while idx + window_size <= len(sorted_samples):
        window = sorted_samples[idx:idx + window_size]
        traits = compute_emotional_traits(window)
        mid_ts = window[len(window) // 2].timestamp
        snapshots.append(EvolutionSnapshot(timestamp=mid_ts, traits=traits))
        idx += step_size

    return snapshots


# ---------------------------------------------------------------------------
# Full genome profile
# ---------------------------------------------------------------------------

def compute_genome_profile(
    user_id: str,
    samples: List[EmotionSample],
    max_chapters: int = _MAX_CHAPTERS,
    evolution_window: int = 20,
    evolution_step: int = 10,
) -> GenomeProfile:
    """Compute the complete emotional genome profile for a user.

    Orchestrates trait computation, fingerprint generation, chapter
    detection, and evolution tracking.

    Args:
        user_id: Unique user identifier.
        samples: Chronologically ordered emotion samples.
        max_chapters: Maximum life chapters to detect.
        evolution_window: Window size for evolution tracking.
        evolution_step: Step size for evolution tracking.

    Returns:
        Complete GenomeProfile.
    """
    traits = compute_emotional_traits(samples)
    fingerprint = generate_emotional_fingerprint(traits, samples)
    chapters = detect_life_chapters(samples, max_chapters)
    evolution = track_emotional_evolution(
        samples, window_size=evolution_window, step_size=evolution_step,
    )

    return GenomeProfile(
        user_id=user_id,
        traits=traits,
        fingerprint=fingerprint,
        chapters=chapters,
        evolution=evolution,
        sample_count=len(samples),
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def profile_to_dict(profile: GenomeProfile) -> Dict[str, Any]:
    """Serialise a GenomeProfile to a JSON-safe dictionary."""
    return {
        "user_id": profile.user_id,
        "traits": asdict(profile.traits),
        "fingerprint": profile.fingerprint,
        "fingerprint_dim": len(profile.fingerprint),
        "chapters": [asdict(ch) for ch in profile.chapters],
        "evolution": [
            {
                "timestamp": snap.timestamp,
                "traits": asdict(snap.traits),
            }
            for snap in profile.evolution
        ],
        "computed_at": profile.computed_at,
        "sample_count": profile.sample_count,
    }
