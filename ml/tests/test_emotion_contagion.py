"""Tests for emotion contagion graph — issue #412.

Covers:
  - State transition detection (thresholds, sustain, edge cases)
  - Transition attribution (temporal decay, normalization, lookback)
  - Influence graph building (aggregation, sorting, direction classification)
  - Insights generation (energizers, drainers, calming, latency, recovery)
  - Serialization (graph_to_dict)
  - Edge cases (empty input, single sample, no events, no transitions)
"""

import sys
import os
import math

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.emotion_contagion import (
    EmotionSample,
    ContextEvent,
    StateTransition,
    TransitionAttribution,
    InfluenceEdge,
    InfluenceGraph,
    detect_state_transitions,
    attribute_transitions,
    build_influence_graph,
    compute_influence_insights,
    graph_to_dict,
    _classify_direction,
    _classify_entity_direction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(minutes: float) -> float:
    """Convert minutes to a fake unix timestamp (base = 1_000_000)."""
    return 1_000_000 + minutes * 60.0


def _make_samples(points: list) -> list:
    """Build EmotionSample list from (minute, valence, arousal) tuples."""
    return [EmotionSample(timestamp=_ts(m), valence=v, arousal=a) for m, v, a in points]


def _make_events(entries: list) -> list:
    """Build ContextEvent list from (minute, type, id, duration) tuples."""
    return [
        ContextEvent(timestamp=_ts(m), entity_type=t, entity_id=eid, duration_min=d)
        for m, t, eid, d in entries
    ]


# ===========================================================================
# Test detect_state_transitions
# ===========================================================================

class TestDetectStateTransitions:

    def test_clear_positive_shift(self):
        """Large positive valence shift sustained for 6 min should be detected."""
        samples = _make_samples([
            (0, -0.5, 0.3),
            (1, -0.4, 0.3),
            (2, -0.3, 0.3),
            (6, 0.3, 0.3),   # shift of +0.8 sustained at t=6
            (7, 0.35, 0.3),
            (8, 0.3, 0.3),
        ])
        transitions = detect_state_transitions(samples, sustain_minutes=5.0)
        assert len(transitions) >= 1
        t = transitions[0]
        assert t.valence_delta > 0
        assert t.direction == "positive"
        assert t.magnitude > 0

    def test_clear_negative_shift(self):
        """Large negative valence shift should be detected as negative."""
        samples = _make_samples([
            (0, 0.5, 0.5),
            (1, 0.4, 0.5),
            (6, -0.3, 0.5),   # shift of -0.8
            (7, -0.35, 0.5),
            (8, -0.3, 0.5),
        ])
        transitions = detect_state_transitions(samples, sustain_minutes=5.0)
        assert len(transitions) >= 1
        assert transitions[0].direction == "negative"

    def test_arousal_shift_detected(self):
        """Large arousal change with small valence change: arousal_up."""
        samples = _make_samples([
            (0, 0.0, 0.1),
            (1, 0.0, 0.15),
            (6, 0.0, 0.6),   # arousal jumps from 0.1 to 0.6
            (7, 0.0, 0.58),
            (8, 0.0, 0.62),
        ])
        transitions = detect_state_transitions(
            samples, valence_threshold=0.25, arousal_threshold=0.20,
            sustain_minutes=5.0,
        )
        assert len(transitions) >= 1
        assert "arousal" in transitions[0].direction

    def test_no_transition_below_threshold(self):
        """Small changes below threshold should not produce transitions."""
        samples = _make_samples([
            (0, 0.0, 0.5),
            (1, 0.05, 0.52),
            (6, 0.1, 0.55),
            (7, 0.08, 0.53),
        ])
        transitions = detect_state_transitions(
            samples, valence_threshold=0.25, arousal_threshold=0.20,
            sustain_minutes=5.0,
        )
        assert len(transitions) == 0

    def test_not_sustained_enough(self):
        """Shift that occurs in < sustain_minutes should not be detected if
        the only sustained sample is too soon."""
        samples = _make_samples([
            (0, 0.0, 0.5),
            (2, 0.6, 0.5),   # shift at t=2, only 2 minutes later
            (3, 0.55, 0.5),
        ])
        # Sustain = 5 min, so t=2 is only 2 min from anchor — below threshold.
        transitions = detect_state_transitions(samples, sustain_minutes=5.0)
        assert len(transitions) == 0

    def test_empty_samples(self):
        """Empty sample list returns empty transitions."""
        assert detect_state_transitions([]) == []

    def test_single_sample(self):
        """Single sample cannot produce a transition."""
        samples = _make_samples([(0, 0.5, 0.5)])
        assert detect_state_transitions(samples) == []

    def test_magnitude_is_euclidean(self):
        """Magnitude should be Euclidean distance in VA space."""
        samples = _make_samples([
            (0, 0.0, 0.0),
            (6, 0.3, 0.4),
            (7, 0.3, 0.4),
            (8, 0.3, 0.4),
        ])
        transitions = detect_state_transitions(
            samples, valence_threshold=0.1, arousal_threshold=0.1,
            sustain_minutes=5.0,
        )
        assert len(transitions) >= 1
        t = transitions[0]
        expected_mag = math.sqrt(t.valence_delta ** 2 + t.arousal_delta ** 2)
        assert abs(t.magnitude - expected_mag) < 0.01


# ===========================================================================
# Test attribute_transitions
# ===========================================================================

class TestAttributeTransitions:

    def test_single_event_full_attribution(self):
        """Single event in lookback window gets score = 1.0."""
        trans = StateTransition(
            start_ts=_ts(10), end_ts=_ts(16),
            valence_delta=0.5, arousal_delta=0.1,
            magnitude=0.51, direction="positive",
        )
        events = _make_events([(5, "person", "alice", 30)])
        attrs = attribute_transitions([trans], events, lookback_min=60)
        assert len(attrs) == 1
        assert abs(attrs[0].score - 1.0) < 0.01
        assert attrs[0].latency_min == pytest.approx(5.0, abs=0.1)

    def test_closer_event_gets_higher_score(self):
        """Between two events, the one closer to the transition gets more attribution."""
        trans = StateTransition(
            start_ts=_ts(30), end_ts=_ts(36),
            valence_delta=-0.4, arousal_delta=0.0,
            magnitude=0.4, direction="negative",
        )
        events = _make_events([
            (5, "activity", "commute", 30),    # 25 min before
            (25, "person", "boss", 15),         # 5 min before
        ])
        attrs = attribute_transitions([trans], events, lookback_min=60)
        assert len(attrs) == 2

        scores = {a.event.entity_id: a.score for a in attrs}
        assert scores["boss"] > scores["commute"]

    def test_event_outside_lookback_excluded(self):
        """Events before the lookback window should not be attributed."""
        trans = StateTransition(
            start_ts=_ts(120), end_ts=_ts(126),
            valence_delta=0.3, arousal_delta=0.1,
            magnitude=0.32, direction="positive",
        )
        # Event at t=0, transition at t=120, lookback=60 min → window starts at t=60
        events = _make_events([(0, "place", "park", 60)])
        attrs = attribute_transitions([trans], events, lookback_min=60)
        assert len(attrs) == 0

    def test_scores_sum_to_one(self):
        """Attribution scores for one transition should sum to 1.0."""
        trans = StateTransition(
            start_ts=_ts(60), end_ts=_ts(66),
            valence_delta=0.5, arousal_delta=0.2,
            magnitude=0.54, direction="positive",
        )
        events = _make_events([
            (10, "person", "alice", 10),
            (30, "activity", "gym", 30),
            (55, "media", "podcast", 15),
        ])
        attrs = attribute_transitions([trans], events, lookback_min=60)
        total = sum(a.score for a in attrs)
        assert abs(total - 1.0) < 0.01

    def test_no_events_returns_empty(self):
        """No events → no attributions."""
        trans = StateTransition(
            start_ts=_ts(10), end_ts=_ts(16),
            valence_delta=0.5, arousal_delta=0.1,
            magnitude=0.51, direction="positive",
        )
        attrs = attribute_transitions([trans], [])
        assert attrs == []

    def test_no_transitions_returns_empty(self):
        """No transitions → no attributions."""
        events = _make_events([(5, "person", "alice", 30)])
        attrs = attribute_transitions([], events)
        assert attrs == []


# ===========================================================================
# Test build_influence_graph
# ===========================================================================

class TestBuildInfluenceGraph:

    def _make_attributions(self):
        """Helper: create attributions for two entities."""
        t1 = StateTransition(
            start_ts=_ts(10), end_ts=_ts(16),
            valence_delta=0.5, arousal_delta=0.2,
            magnitude=0.54, direction="positive",
        )
        t2 = StateTransition(
            start_ts=_ts(30), end_ts=_ts(36),
            valence_delta=-0.4, arousal_delta=-0.1,
            magnitude=0.41, direction="negative",
        )
        e1 = ContextEvent(timestamp=_ts(5), entity_type="person", entity_id="alice", duration_min=30)
        e2 = ContextEvent(timestamp=_ts(25), entity_type="activity", entity_id="commute", duration_min=30)

        return [
            TransitionAttribution(transition=t1, event=e1, score=0.7, latency_min=5.0),
            TransitionAttribution(transition=t1, event=e2, score=0.3, latency_min=5.0),
            TransitionAttribution(transition=t2, event=e2, score=1.0, latency_min=5.0),
        ]

    def test_graph_has_correct_edges(self):
        """Graph should have one edge per unique entity."""
        attrs = self._make_attributions()
        graph = build_influence_graph(attrs)
        assert len(graph.edges) == 2
        ids = {e.entity_id for e in graph.edges}
        assert ids == {"alice", "commute"}

    def test_edges_sorted_by_influence(self):
        """Edges should be sorted by influence_score descending."""
        attrs = self._make_attributions()
        graph = build_influence_graph(attrs)
        scores = [e.influence_score for e in graph.edges]
        assert scores == sorted(scores, reverse=True)

    def test_entity_types_counted(self):
        """entity_types dict should count entities per type."""
        attrs = self._make_attributions()
        graph = build_influence_graph(attrs)
        assert "person" in graph.entity_types
        assert "activity" in graph.entity_types

    def test_empty_attributions(self):
        """Empty attributions → empty graph."""
        graph = build_influence_graph([])
        assert graph.edges == []
        assert graph.total_transitions == 0

    def test_direction_classification(self):
        """alice has positive valence_delta → energizer; commute has mixed negative → drainer."""
        attrs = self._make_attributions()
        graph = build_influence_graph(attrs)
        edge_map = {e.entity_id: e for e in graph.edges}
        # alice: avg_v = 0.5*0.7 = 0.35 > 0, avg_a = 0.2*0.7 = 0.14 > 0 → energizer
        assert edge_map["alice"].direction == "energizer"


# ===========================================================================
# Test compute_influence_insights
# ===========================================================================

class TestComputeInsights:

    def test_insights_with_data(self):
        """Insights should have all expected keys with real data."""
        t1 = StateTransition(
            start_ts=_ts(10), end_ts=_ts(16),
            valence_delta=0.5, arousal_delta=0.2,
            magnitude=0.54, direction="positive",
        )
        e1 = ContextEvent(timestamp=_ts(5), entity_type="person", entity_id="alice", duration_min=30)
        attrs = [TransitionAttribution(transition=t1, event=e1, score=1.0, latency_min=5.0)]
        graph = build_influence_graph(attrs)
        insights = compute_influence_insights(graph)

        assert "top_energizers" in insights
        assert "top_drainers" in insights
        assert "top_calming" in insights
        assert "latency_patterns" in insights
        assert "recovery_summary" in insights
        assert "entity_type_breakdown" in insights
        assert "summary" in insights
        assert len(insights["top_energizers"]) >= 1

    def test_empty_graph_insights(self):
        """Empty graph should produce a safe insights dict."""
        graph = InfluenceGraph()
        insights = compute_influence_insights(graph)
        assert insights["top_energizers"] == []
        assert insights["top_drainers"] == []
        assert "Not enough data" in insights["summary"]

    def test_entity_type_breakdown(self):
        """Entity type breakdown should reflect the graph's entity_types."""
        edge = InfluenceEdge(
            entity_type="media", entity_id="podcast_x",
            avg_valence_delta=0.3, avg_arousal_delta=0.1,
            influence_score=2.0, exposure_count=3,
            avg_latency_min=8.0, avg_recovery_min=0.0,
            direction="energizer",
        )
        graph = InfluenceGraph(
            edges=[edge],
            entity_types={"media": 1},
            total_transitions=1,
            total_attributions=3,
        )
        insights = compute_influence_insights(graph)
        assert "media" in insights["entity_type_breakdown"]
        assert insights["entity_type_breakdown"]["media"]["count"] == 1


# ===========================================================================
# Test graph_to_dict
# ===========================================================================

class TestGraphToDict:

    def test_serialization_keys(self):
        """graph_to_dict should produce a dict with expected top-level keys."""
        graph = InfluenceGraph(
            edges=[],
            entity_types={"person": 1},
            total_transitions=2,
            total_attributions=5,
        )
        d = graph_to_dict(graph)
        assert d["edges"] == []
        assert d["entity_types"] == {"person": 1}
        assert d["total_transitions"] == 2
        assert d["total_attributions"] == 5

    def test_edge_serialization(self):
        """Each edge should serialize to a dict with all fields."""
        edge = InfluenceEdge(
            entity_type="person", entity_id="alice",
            avg_valence_delta=0.3, avg_arousal_delta=0.1,
            influence_score=1.5, exposure_count=4,
            avg_latency_min=10.0, avg_recovery_min=5.0,
            direction="energizer",
        )
        graph = InfluenceGraph(edges=[edge], entity_types={"person": 1})
        d = graph_to_dict(graph)
        assert len(d["edges"]) == 1
        e = d["edges"][0]
        assert e["entity_id"] == "alice"
        assert e["direction"] == "energizer"
        assert e["influence_score"] == 1.5


# ===========================================================================
# Test helper functions
# ===========================================================================

class TestHelpers:

    def test_classify_direction_positive(self):
        assert _classify_direction(0.5, 0.1) == "positive"

    def test_classify_direction_negative(self):
        assert _classify_direction(-0.5, 0.1) == "negative"

    def test_classify_direction_arousal_up(self):
        assert _classify_direction(0.1, 0.5) == "arousal_up"

    def test_classify_direction_arousal_down(self):
        assert _classify_direction(0.1, -0.5) == "arousal_down"

    def test_classify_entity_energizer(self):
        assert _classify_entity_direction(0.3, 0.2) == "energizer"

    def test_classify_entity_drainer(self):
        assert _classify_entity_direction(-0.3, 0.2) == "drainer"

    def test_classify_entity_calming(self):
        assert _classify_entity_direction(0.3, -0.1) == "calming"

    def test_classify_entity_neutral(self):
        assert _classify_entity_direction(0.0, 0.0) == "neutral"
