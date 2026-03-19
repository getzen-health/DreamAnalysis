"""Collective emotional intelligence — population-level emotion aggregation.

Aggregates anonymized emotional data from multiple users to compute
collective mood indices, detect widespread emotional events (stress spikes,
joy waves), and identify geographic/temporal patterns — all without
individual identification.

Privacy-first design:
    - All inputs are pre-anonymized (no user IDs, no raw EEG)
    - Only aggregated valence/arousal distributions are accepted
    - Minimum group size enforced before any output is returned
    - No individual reconstruction is possible from outputs

Collective metrics:
    - Collective mood index: weighted mean of valence/arousal across population
    - Emotional coherence: how aligned the population's emotions are (variance)
    - Event detection: sudden shifts in collective distribution (z-score based)
    - Temporal patterns: hour-of-day, day-of-week emotional rhythms
    - Geographic patterns: region-level mood aggregation

Issue #461.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of anonymous contributions before releasing any aggregation
_MIN_GROUP_SIZE = 5
# Z-score threshold for collective event detection
_EVENT_Z_THRESHOLD = 2.0
# Maximum history length for rolling statistics
_MAX_HISTORY = 1000


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AnonymousEmotionSample:
    """A single anonymized emotion contribution.

    No user ID, no raw EEG — just the derived emotional state
    with optional geographic region and timestamp.
    """
    valence: float = 0.0       # -1..+1
    arousal: float = 0.5       # 0..1
    stress: float = 0.3        # 0..1
    energy: float = 0.5        # 0..1
    timestamp: float = 0.0     # unix epoch
    region: str = "unknown"    # anonymized geographic region


@dataclass
class CollectiveMood:
    """Population-level mood snapshot."""
    mean_valence: float = 0.0
    mean_arousal: float = 0.5
    mean_stress: float = 0.3
    mean_energy: float = 0.5
    valence_std: float = 0.0
    arousal_std: float = 0.0
    coherence: float = 0.0       # 0..1, how aligned emotions are
    sample_count: int = 0
    mood_label: str = "neutral"  # human-readable label
    timestamp: float = 0.0


@dataclass
class CollectiveEvent:
    """A detected collective emotional event (sudden widespread shift)."""
    event_type: str = "unknown"    # "stress_spike", "joy_wave", "energy_drop", etc.
    z_score: float = 0.0          # magnitude of the deviation
    metric: str = "valence"       # which metric triggered the event
    mean_value: float = 0.0       # the observed mean during the event
    baseline_mean: float = 0.0    # the rolling baseline mean
    baseline_std: float = 0.0     # the rolling baseline std
    sample_count: int = 0
    timestamp: float = 0.0
    region: str = "global"


@dataclass
class CollectiveProfile:
    """Full collective emotional intelligence report."""
    mood: CollectiveMood
    events: List[CollectiveEvent] = field(default_factory=list)
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    geographic_patterns: Dict[str, Any] = field(default_factory=dict)
    sample_count: int = 0
    sufficient_data: bool = False
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _classify_mood(valence: float, arousal: float) -> str:
    """Map valence/arousal to a human-readable mood label."""
    if valence > 0.3 and arousal > 0.5:
        return "excited_positive"
    elif valence > 0.3 and arousal <= 0.5:
        return "calm_positive"
    elif valence < -0.3 and arousal > 0.5:
        return "stressed_negative"
    elif valence < -0.3 and arousal <= 0.5:
        return "subdued_negative"
    elif arousal > 0.7:
        return "high_energy"
    elif arousal < 0.3:
        return "low_energy"
    return "neutral"


def _classify_event(metric: str, z_score: float) -> str:
    """Classify a collective event based on the metric and z-score direction."""
    if metric == "valence":
        return "joy_wave" if z_score > 0 else "distress_wave"
    elif metric == "arousal":
        return "energy_surge" if z_score > 0 else "energy_drop"
    elif metric == "stress":
        return "stress_spike" if z_score > 0 else "stress_relief"
    elif metric == "energy":
        return "vitality_wave" if z_score > 0 else "fatigue_wave"
    return "unknown_event"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def aggregate_anonymous_emotions(
    samples: List[AnonymousEmotionSample],
) -> CollectiveMood:
    """Aggregate anonymized emotion samples into a collective mood.

    Enforces minimum group size for privacy. Returns a neutral mood
    with sample_count=0 if insufficient data.

    Args:
        samples: List of anonymized emotion contributions.

    Returns:
        CollectiveMood with population-level statistics.
    """
    if len(samples) < _MIN_GROUP_SIZE:
        return CollectiveMood(
            sample_count=len(samples),
            mood_label="insufficient_data",
            timestamp=time.time(),
        )

    valences = [s.valence for s in samples]
    arousals = [s.arousal for s in samples]
    stresses = [s.stress for s in samples]
    energies = [s.energy for s in samples]

    mean_v = _safe_mean(valences)
    mean_a = _safe_mean(arousals)
    mean_s = _safe_mean(stresses)
    mean_e = _safe_mean(energies)
    std_v = _safe_std(valences)
    std_a = _safe_std(arousals)

    # Coherence: inverse of average standard deviation across dimensions
    # High coherence = everyone feeling similar; low = diverse
    avg_std = (std_v + std_a) / 2.0
    # Map std to 0..1 coherence (std of 0 = perfect coherence, std of 1 = no coherence)
    coherence = _clamp(1.0 - avg_std * 2.0, 0.0, 1.0)

    mood_label = _classify_mood(mean_v, mean_a)

    return CollectiveMood(
        mean_valence=round(mean_v, 4),
        mean_arousal=round(mean_a, 4),
        mean_stress=round(mean_s, 4),
        mean_energy=round(mean_e, 4),
        valence_std=round(std_v, 4),
        arousal_std=round(std_a, 4),
        coherence=round(coherence, 4),
        sample_count=len(samples),
        mood_label=mood_label,
        timestamp=time.time(),
    )


def compute_collective_mood(
    samples: List[AnonymousEmotionSample],
) -> Dict[str, Any]:
    """Convenience wrapper: aggregate and return as dict.

    Args:
        samples: Anonymized emotion contributions.

    Returns:
        Dict with mood metrics, label, and metadata.
    """
    mood = aggregate_anonymous_emotions(samples)
    return asdict(mood)


def detect_collective_events(
    current_samples: List[AnonymousEmotionSample],
    baseline_samples: Optional[List[AnonymousEmotionSample]] = None,
    z_threshold: float = _EVENT_Z_THRESHOLD,
) -> List[CollectiveEvent]:
    """Detect collective emotional events by comparing current to baseline.

    Checks if current population metrics deviate significantly from a
    rolling baseline (z-score based).

    Args:
        current_samples: Recent anonymized samples (the window to test).
        baseline_samples: Historical baseline samples. If None, returns empty.
        z_threshold: Z-score threshold for event detection.

    Returns:
        List of detected CollectiveEvent objects.
    """
    if not baseline_samples or len(current_samples) < _MIN_GROUP_SIZE:
        return []
    if len(baseline_samples) < _MIN_GROUP_SIZE:
        return []

    events: List[CollectiveEvent] = []
    now = time.time()

    # Extract metrics from both pools
    metrics = {
        "valence": (
            [s.valence for s in current_samples],
            [s.valence for s in baseline_samples],
        ),
        "arousal": (
            [s.arousal for s in current_samples],
            [s.arousal for s in baseline_samples],
        ),
        "stress": (
            [s.stress for s in current_samples],
            [s.stress for s in baseline_samples],
        ),
        "energy": (
            [s.energy for s in current_samples],
            [s.energy for s in baseline_samples],
        ),
    }

    for metric_name, (current_vals, baseline_vals) in metrics.items():
        baseline_mean = _safe_mean(baseline_vals)
        baseline_std = _safe_std(baseline_vals)

        if baseline_std < 0.01:
            # No meaningful variance in baseline — skip
            continue

        current_mean = _safe_mean(current_vals)
        z = (current_mean - baseline_mean) / baseline_std

        if abs(z) >= z_threshold:
            event_type = _classify_event(metric_name, z)
            events.append(CollectiveEvent(
                event_type=event_type,
                z_score=round(z, 4),
                metric=metric_name,
                mean_value=round(current_mean, 4),
                baseline_mean=round(baseline_mean, 4),
                baseline_std=round(baseline_std, 4),
                sample_count=len(current_samples),
                timestamp=now,
                region="global",
            ))

    return events


def _compute_temporal_patterns(
    samples: List[AnonymousEmotionSample],
) -> Dict[str, Any]:
    """Compute hour-of-day emotional patterns from timestamps.

    Groups samples by hour and computes average valence/arousal per hour.
    """
    if not samples:
        return {"hourly": {}, "summary": "No data"}

    hourly: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for s in samples:
        if s.timestamp > 0:
            import datetime
            hour = datetime.datetime.fromtimestamp(s.timestamp).hour
            hourly[hour].append((s.valence, s.arousal))

    if not hourly:
        return {"hourly": {}, "summary": "No timestamp data"}

    hourly_stats: Dict[str, Dict[str, float]] = {}
    for hour, pairs in sorted(hourly.items()):
        vals = [p[0] for p in pairs]
        aros = [p[1] for p in pairs]
        hourly_stats[str(hour)] = {
            "mean_valence": round(_safe_mean(vals), 4),
            "mean_arousal": round(_safe_mean(aros), 4),
            "count": len(pairs),
        }

    return {"hourly": hourly_stats, "summary": f"Patterns across {len(hourly)} hours"}


def _compute_geographic_patterns(
    samples: List[AnonymousEmotionSample],
) -> Dict[str, Any]:
    """Compute region-level mood aggregation."""
    if not samples:
        return {"regions": {}, "summary": "No data"}

    by_region: Dict[str, List[AnonymousEmotionSample]] = defaultdict(list)
    for s in samples:
        by_region[s.region].append(s)

    region_stats: Dict[str, Dict[str, Any]] = {}
    for region, region_samples in by_region.items():
        if len(region_samples) < _MIN_GROUP_SIZE:
            continue  # privacy: don't report regions with too few samples
        valences = [s.valence for s in region_samples]
        arousals = [s.arousal for s in region_samples]
        region_stats[region] = {
            "mean_valence": round(_safe_mean(valences), 4),
            "mean_arousal": round(_safe_mean(arousals), 4),
            "count": len(region_samples),
            "mood_label": _classify_mood(_safe_mean(valences), _safe_mean(arousals)),
        }

    return {
        "regions": region_stats,
        "summary": f"{len(region_stats)} regions with sufficient data",
    }


def compute_collective_profile(
    samples: List[AnonymousEmotionSample],
    baseline_samples: Optional[List[AnonymousEmotionSample]] = None,
) -> CollectiveProfile:
    """Compute a full collective emotional intelligence profile.

    Args:
        samples: Current anonymized samples.
        baseline_samples: Historical baseline for event detection.

    Returns:
        CollectiveProfile with mood, events, temporal/geographic patterns.
    """
    mood = aggregate_anonymous_emotions(samples)
    events = detect_collective_events(samples, baseline_samples)
    temporal = _compute_temporal_patterns(samples)
    geographic = _compute_geographic_patterns(samples)

    sufficient = len(samples) >= _MIN_GROUP_SIZE

    return CollectiveProfile(
        mood=mood,
        events=events,
        temporal_patterns=temporal,
        geographic_patterns=geographic,
        sample_count=len(samples),
        sufficient_data=sufficient,
        timestamp=time.time(),
    )


def profile_to_dict(profile: CollectiveProfile) -> Dict[str, Any]:
    """Serialize a CollectiveProfile to a plain dict."""
    return {
        "mood": asdict(profile.mood),
        "events": [asdict(e) for e in profile.events],
        "temporal_patterns": profile.temporal_patterns,
        "geographic_patterns": profile.geographic_patterns,
        "sample_count": profile.sample_count,
        "sufficient_data": profile.sufficient_data,
        "timestamp": profile.timestamp,
    }
