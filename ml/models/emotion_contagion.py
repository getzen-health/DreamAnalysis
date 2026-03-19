"""Contextual emotion contagion graph — issue #412.

Builds a personal emotional influence graph where nodes represent entities
(people, places, activities, media) and edges represent influence scores
derived from temporal co-occurrence with emotional state transitions.

Algorithm:
1. Detect significant emotional state transitions (valence/arousal shifts
   sustained for >5 minutes).
2. For each transition, look backward 0-60 minutes for context events that
   preceded it.
3. Compute transition attribution scores using temporal precedence weighting
   (closer events get higher attribution).
4. Aggregate attributions into an influence graph with node and edge metadata.
5. Generate insights: top energizers, top drainers, latency patterns,
   recovery times.

References:
    Hatfield et al. (1993) — Emotional contagion
    Fowler & Christakis (2008) — Dynamic spread of happiness in social networks
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmotionSample:
    """A single timestamped emotion measurement."""
    timestamp: float  # unix epoch seconds
    valence: float    # -1.0 to 1.0 (negative to positive)
    arousal: float    # 0.0 to 1.0 (calm to energetic)


@dataclass
class ContextEvent:
    """An external event with known entity and duration."""
    timestamp: float       # unix epoch seconds — start of event
    entity_type: str       # "person", "place", "activity", "media"
    entity_id: str         # unique identifier for the entity
    duration_min: float    # duration in minutes


@dataclass
class StateTransition:
    """A detected emotional state change."""
    start_ts: float
    end_ts: float
    valence_delta: float
    arousal_delta: float
    magnitude: float       # Euclidean distance in VA space
    direction: str         # "positive", "negative", "arousal_up", "arousal_down", "mixed"


@dataclass
class TransitionAttribution:
    """Links a state transition to a context event with a score."""
    transition: StateTransition
    event: ContextEvent
    score: float           # 0.0 to 1.0 — attribution weight
    latency_min: float     # minutes between event start and transition start


@dataclass
class InfluenceEdge:
    """A directed edge from an entity to the user's emotional state."""
    entity_type: str
    entity_id: str
    avg_valence_delta: float
    avg_arousal_delta: float
    influence_score: float      # aggregated attribution score
    exposure_count: int
    avg_latency_min: float
    avg_recovery_min: float
    direction: str              # "energizer", "drainer", "calming", "neutral"


@dataclass
class InfluenceGraph:
    """The full influence graph: nodes are entities, edges carry influence data."""
    edges: List[InfluenceEdge] = field(default_factory=list)
    entity_types: Dict[str, int] = field(default_factory=dict)
    total_transitions: int = 0
    total_attributions: int = 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum shift in valence or arousal to count as a significant transition
_VALENCE_THRESHOLD = 0.25
_AROUSAL_THRESHOLD = 0.20

# Transition must be sustained for at least this many minutes
_SUSTAIN_MINUTES = 5.0

# Lookback window for context events (minutes before transition start)
_LOOKBACK_MIN = 60.0

# Temporal decay half-life (minutes) for attribution weighting
_DECAY_HALF_LIFE_MIN = 15.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def detect_state_transitions(
    samples: List[EmotionSample],
    valence_threshold: float = _VALENCE_THRESHOLD,
    arousal_threshold: float = _AROUSAL_THRESHOLD,
    sustain_minutes: float = _SUSTAIN_MINUTES,
) -> List[StateTransition]:
    """Detect significant emotional state transitions.

    A transition is a shift in valence or arousal (or both) that exceeds
    the given thresholds and is sustained for at least ``sustain_minutes``.

    Args:
        samples: Chronologically sorted emotion samples.
        valence_threshold: Minimum absolute valence change.
        arousal_threshold: Minimum absolute arousal change.
        sustain_minutes: Minimum duration the new state must be held.

    Returns:
        List of detected StateTransition objects.
    """
    if len(samples) < 2:
        return []

    transitions: List[StateTransition] = []
    sustain_sec = sustain_minutes * 60.0

    # Sliding comparison: for each sample, compare against future samples
    # that are at least sustain_sec later.
    i = 0
    while i < len(samples) - 1:
        anchor = samples[i]

        # Find the first sample that is sustain_sec later
        j = i + 1
        while j < len(samples) and (samples[j].timestamp - anchor.timestamp) < sustain_sec:
            j += 1

        if j >= len(samples):
            break

        # Average the samples in the sustained window [j, j+window] to
        # reduce noise — use up to 5 samples or whatever is available.
        window_end = min(j + 5, len(samples))
        sustained_vals = samples[j:window_end]
        avg_valence = sum(s.valence for s in sustained_vals) / len(sustained_vals)
        avg_arousal = sum(s.arousal for s in sustained_vals) / len(sustained_vals)

        v_delta = avg_valence - anchor.valence
        a_delta = avg_arousal - anchor.arousal
        magnitude = math.sqrt(v_delta ** 2 + a_delta ** 2)

        is_significant = (
            abs(v_delta) >= valence_threshold or
            abs(a_delta) >= arousal_threshold
        )

        if is_significant:
            direction = _classify_direction(v_delta, a_delta)
            transitions.append(StateTransition(
                start_ts=anchor.timestamp,
                end_ts=samples[j].timestamp,
                valence_delta=round(v_delta, 4),
                arousal_delta=round(a_delta, 4),
                magnitude=round(magnitude, 4),
                direction=direction,
            ))
            # Skip past this transition to avoid overlapping detections.
            i = j
        else:
            i += 1

    return transitions


def attribute_transitions(
    transitions: List[StateTransition],
    events: List[ContextEvent],
    lookback_min: float = _LOOKBACK_MIN,
    decay_half_life_min: float = _DECAY_HALF_LIFE_MIN,
) -> List[TransitionAttribution]:
    """Attribute each transition to preceding context events.

    For each transition, finds events whose start time falls within the
    lookback window before the transition start. Attribution score decays
    exponentially with temporal distance — closer events get higher scores.

    Args:
        transitions: Detected state transitions.
        events: Context events with timestamps.
        lookback_min: Maximum lookback window in minutes.
        decay_half_life_min: Half-life for exponential temporal decay.

    Returns:
        List of TransitionAttribution objects.
    """
    if not transitions or not events:
        return []

    lookback_sec = lookback_min * 60.0
    decay_lambda = math.log(2) / (decay_half_life_min * 60.0) if decay_half_life_min > 0 else 0.0

    attributions: List[TransitionAttribution] = []

    for trans in transitions:
        window_start = trans.start_ts - lookback_sec

        # Find events in the lookback window
        candidates: List[Tuple[ContextEvent, float]] = []
        for evt in events:
            if window_start <= evt.timestamp <= trans.start_ts:
                latency_sec = trans.start_ts - evt.timestamp
                latency_min = latency_sec / 60.0

                # Exponential temporal decay
                raw_score = math.exp(-decay_lambda * latency_sec)
                candidates.append((evt, raw_score, latency_min))

        if not candidates:
            continue

        # Normalize scores so they sum to 1 within this transition
        total = sum(c[1] for c in candidates)
        if total <= 0:
            continue

        for evt, raw, lat_min in candidates:
            norm_score = raw / total
            attributions.append(TransitionAttribution(
                transition=trans,
                event=evt,
                score=round(norm_score, 4),
                latency_min=round(lat_min, 2),
            ))

    return attributions


def build_influence_graph(
    attributions: List[TransitionAttribution],
    samples: Optional[List[EmotionSample]] = None,
) -> InfluenceGraph:
    """Aggregate attributions into an influence graph.

    Groups attributions by entity_id, computes aggregate metrics, and
    classifies each entity as energizer, drainer, calming, or neutral.

    Args:
        attributions: All transition-event attributions.
        samples: Optional emotion samples for recovery time estimation.

    Returns:
        InfluenceGraph with edges and metadata.
    """
    if not attributions:
        return InfluenceGraph()

    # Group by entity_id
    entity_data: Dict[str, Dict] = defaultdict(lambda: {
        "entity_type": "",
        "valence_deltas": [],
        "arousal_deltas": [],
        "scores": [],
        "latencies": [],
        "recovery_times": [],
    })

    for attr in attributions:
        key = attr.event.entity_id
        data = entity_data[key]
        data["entity_type"] = attr.event.entity_type
        data["valence_deltas"].append(attr.transition.valence_delta * attr.score)
        data["arousal_deltas"].append(attr.transition.arousal_delta * attr.score)
        data["scores"].append(attr.score)
        data["latencies"].append(attr.latency_min)

        # Estimate recovery time if we have samples
        if samples:
            recovery = _estimate_recovery(attr.transition, samples)
            if recovery is not None:
                data["recovery_times"].append(recovery)

    # Build edges
    edges: List[InfluenceEdge] = []
    type_counts: Dict[str, int] = defaultdict(int)

    for entity_id, data in entity_data.items():
        avg_v = sum(data["valence_deltas"]) / len(data["valence_deltas"])
        avg_a = sum(data["arousal_deltas"]) / len(data["arousal_deltas"])
        total_score = sum(data["scores"])
        avg_latency = sum(data["latencies"]) / len(data["latencies"])
        avg_recovery = (
            sum(data["recovery_times"]) / len(data["recovery_times"])
            if data["recovery_times"]
            else 0.0
        )
        count = len(data["scores"])
        direction = _classify_entity_direction(avg_v, avg_a)

        type_counts[data["entity_type"]] += 1

        edges.append(InfluenceEdge(
            entity_type=data["entity_type"],
            entity_id=entity_id,
            avg_valence_delta=round(avg_v, 4),
            avg_arousal_delta=round(avg_a, 4),
            influence_score=round(total_score, 4),
            exposure_count=count,
            avg_latency_min=round(avg_latency, 2),
            avg_recovery_min=round(avg_recovery, 2),
            direction=direction,
        ))

    # Unique transitions
    unique_transitions = set()
    for attr in attributions:
        unique_transitions.add((attr.transition.start_ts, attr.transition.end_ts))

    return InfluenceGraph(
        edges=sorted(edges, key=lambda e: e.influence_score, reverse=True),
        entity_types=dict(type_counts),
        total_transitions=len(unique_transitions),
        total_attributions=len(attributions),
    )


def compute_influence_insights(graph: InfluenceGraph) -> Dict:
    """Generate human-readable insights from the influence graph.

    Returns:
        Dict with keys: top_energizers, top_drainers, top_calming,
        latency_patterns, recovery_summary, entity_type_breakdown.
    """
    if not graph.edges:
        return {
            "top_energizers": [],
            "top_drainers": [],
            "top_calming": [],
            "latency_patterns": [],
            "recovery_summary": {"avg_recovery_min": 0.0, "entities_with_longest_recovery": []},
            "entity_type_breakdown": {},
            "summary": "Not enough data to generate insights.",
        }

    energizers = [e for e in graph.edges if e.direction == "energizer"]
    drainers = [e for e in graph.edges if e.direction == "drainer"]
    calming = [e for e in graph.edges if e.direction == "calming"]

    # Sort by influence score descending
    top_energizers = sorted(energizers, key=lambda e: e.influence_score, reverse=True)[:5]
    top_drainers = sorted(drainers, key=lambda e: e.influence_score, reverse=True)[:5]
    top_calming = sorted(calming, key=lambda e: e.influence_score, reverse=True)[:5]

    # Latency patterns — entities with consistently short or long lag
    latency_patterns = []
    for edge in graph.edges:
        if edge.exposure_count >= 2:
            latency_patterns.append({
                "entity_id": edge.entity_id,
                "entity_type": edge.entity_type,
                "avg_latency_min": edge.avg_latency_min,
                "exposure_count": edge.exposure_count,
                "category": "fast_acting" if edge.avg_latency_min < 10 else "slow_onset",
            })
    latency_patterns.sort(key=lambda p: p["avg_latency_min"])

    # Recovery summary
    edges_with_recovery = [e for e in graph.edges if e.avg_recovery_min > 0]
    avg_recovery = (
        sum(e.avg_recovery_min for e in edges_with_recovery) / len(edges_with_recovery)
        if edges_with_recovery
        else 0.0
    )
    longest_recovery = sorted(edges_with_recovery, key=lambda e: e.avg_recovery_min, reverse=True)[:3]

    # Entity type breakdown
    type_breakdown: Dict[str, Dict] = {}
    for etype, count in graph.entity_types.items():
        type_edges = [e for e in graph.edges if e.entity_type == etype]
        avg_influence = sum(e.influence_score for e in type_edges) / len(type_edges) if type_edges else 0
        type_breakdown[etype] = {
            "count": count,
            "avg_influence_score": round(avg_influence, 4),
        }

    # Summary sentence
    parts = []
    if top_energizers:
        parts.append(f"{top_energizers[0].entity_id} is your top energizer")
    if top_drainers:
        parts.append(f"{top_drainers[0].entity_id} is your top drainer")
    summary = "; ".join(parts) + "." if parts else "No clear patterns detected yet."

    return {
        "top_energizers": [_edge_to_dict(e) for e in top_energizers],
        "top_drainers": [_edge_to_dict(e) for e in top_drainers],
        "top_calming": [_edge_to_dict(e) for e in top_calming],
        "latency_patterns": latency_patterns,
        "recovery_summary": {
            "avg_recovery_min": round(avg_recovery, 2),
            "entities_with_longest_recovery": [_edge_to_dict(e) for e in longest_recovery],
        },
        "entity_type_breakdown": type_breakdown,
        "summary": summary,
    }


def graph_to_dict(graph: InfluenceGraph) -> Dict:
    """Serialize an InfluenceGraph to a plain dict for JSON responses."""
    return {
        "edges": [_edge_to_dict(e) for e in graph.edges],
        "entity_types": graph.entity_types,
        "total_transitions": graph.total_transitions,
        "total_attributions": graph.total_attributions,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_direction(v_delta: float, a_delta: float) -> str:
    """Classify a transition by its dominant axis."""
    if abs(v_delta) >= abs(a_delta):
        return "positive" if v_delta > 0 else "negative"
    else:
        return "arousal_up" if a_delta > 0 else "arousal_down"


def _classify_entity_direction(avg_v: float, avg_a: float) -> str:
    """Classify an entity's overall emotional influence direction."""
    if avg_v > 0.01 and avg_a > 0.01:
        return "energizer"
    elif avg_v < -0.01:
        return "drainer"
    elif avg_v > 0.01 and avg_a <= 0.01:
        return "calming"
    elif avg_v <= 0.01 and avg_a <= 0.01:
        if avg_a < -0.01:
            return "calming"
        return "neutral"
    else:
        return "neutral"


def _estimate_recovery(
    transition: StateTransition,
    samples: List[EmotionSample],
    threshold_fraction: float = 0.5,
) -> Optional[float]:
    """Estimate recovery time after a transition.

    Recovery = time until valence returns halfway to pre-transition level.
    Returns minutes, or None if recovery not observed in available data.
    """
    # Find samples after the transition end
    post_samples = [s for s in samples if s.timestamp > transition.end_ts]
    if not post_samples:
        return None

    # We define recovery as returning at least half of the valence delta
    target_return = abs(transition.valence_delta) * threshold_fraction

    for sample in post_samples:
        # How much valence has recovered toward the pre-transition level
        if transition.valence_delta > 0:
            # Positive transition — recovery means valence drops back
            recovered = transition.valence_delta - (sample.valence - (
                post_samples[0].valence - transition.valence_delta
            ))
        else:
            # Negative transition — recovery means valence rises back
            recovered = abs(transition.valence_delta) - abs(
                sample.valence - post_samples[0].valence
            )

        if recovered >= target_return:
            recovery_min = (sample.timestamp - transition.end_ts) / 60.0
            return round(recovery_min, 2)

    return None


def _edge_to_dict(edge: InfluenceEdge) -> Dict:
    """Serialize an InfluenceEdge to a plain dict."""
    return {
        "entity_type": edge.entity_type,
        "entity_id": edge.entity_id,
        "avg_valence_delta": edge.avg_valence_delta,
        "avg_arousal_delta": edge.avg_arousal_delta,
        "influence_score": edge.influence_score,
        "exposure_count": edge.exposure_count,
        "avg_latency_min": edge.avg_latency_min,
        "avg_recovery_min": edge.avg_recovery_min,
        "direction": edge.direction,
    }
