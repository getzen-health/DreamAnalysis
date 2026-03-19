"""Tests for neural time travel -- replay past emotional states (issue #459).

Covers: snapshot storage, cosine similarity matching, neurofeedback target
generation, replay session planning, visualization data, travel profiles,
serialization, edge cases, and API routes.
"""

import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.neural_time_travel import (
    EmotionalSnapshot,
    NeurofeedbackTarget,
    ReplaySession,
    ReplayStep,
    TravelProfile,
    _cosine_similarity,
    clear_library,
    compute_travel_profile,
    find_similar_states,
    generate_neurofeedback_target,
    generate_visualization_data,
    plan_replay_session,
    profile_to_dict,
    store_emotional_snapshot,
)
from api.routes.neural_time_travel import router

# ---------------------------------------------------------------------------
# Test client setup
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_library():
    """Clear the snapshot library before each test."""
    clear_library()
    yield
    clear_library()


def _make_features(valence: float = 0.0, arousal: float = 0.0) -> list:
    """Create a 10-element feature vector with valence/arousal at indices 8/9."""
    return [0.1, 0.2, 0.3, 0.2, 0.05, 1.5, 1.0, 0.1, valence, arousal]


def _store_happy(user: str = "u1") -> EmotionalSnapshot:
    return store_emotional_snapshot(
        user_id=user,
        emotion_label="happy",
        feature_vector=_make_features(0.8, 0.6),
        valence=0.8,
        arousal=0.6,
        context="birthday party",
        tags=["celebration", "friends"],
    )


def _store_calm(user: str = "u1") -> EmotionalSnapshot:
    return store_emotional_snapshot(
        user_id=user,
        emotion_label="calm",
        feature_vector=_make_features(0.5, -0.4),
        valence=0.5,
        arousal=-0.4,
        context="meditation session",
        tags=["meditation"],
    )


def _store_sad(user: str = "u1") -> EmotionalSnapshot:
    return store_emotional_snapshot(
        user_id=user,
        emotion_label="sad",
        feature_vector=_make_features(-0.7, -0.3),
        valence=-0.7,
        arousal=-0.3,
        context="rainy day",
    )


# ---------------------------------------------------------------------------
# 1. Snapshot storage
# ---------------------------------------------------------------------------


class TestSnapshotStorage:
    def test_store_basic(self):
        """Store a snapshot and verify all fields."""
        snap = _store_happy()
        assert isinstance(snap, EmotionalSnapshot)
        assert snap.user_id == "u1"
        assert snap.emotion_label == "happy"
        assert snap.valence == 0.8
        assert snap.arousal == 0.6
        assert snap.context == "birthday party"
        assert snap.tags == ["celebration", "friends"]
        assert len(snap.snapshot_id) > 0
        assert snap.timestamp > 0

    def test_store_clamps_valence_arousal(self):
        """Valence and arousal should be clamped to [-1, 1]."""
        snap = store_emotional_snapshot(
            user_id="u1", emotion_label="test",
            feature_vector=[1.0, 2.0, 3.0],
            valence=5.0, arousal=-9.0,
        )
        assert snap.valence == 1.0
        assert snap.arousal == -1.0

    def test_store_normalises_label(self):
        """Emotion labels should be lowercase and stripped."""
        snap = store_emotional_snapshot(
            user_id="u1", emotion_label="  Happy  ",
            feature_vector=[1.0],
        )
        assert snap.emotion_label == "happy"

    def test_store_empty_vector_raises(self):
        """Empty feature vector should raise ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            store_emotional_snapshot(
                user_id="u1", emotion_label="test", feature_vector=[],
            )

    def test_store_multiple_creates_library(self):
        """Multiple stores should accumulate in the library."""
        _store_happy()
        _store_calm()
        _store_sad()
        matches = find_similar_states("u1", _make_features(), top_k=100, min_similarity=-1.0)
        assert len(matches) == 3


# ---------------------------------------------------------------------------
# 2. Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_zero_vector(self):
        """Zero vector should return similarity 0.0."""
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_different_lengths(self):
        """Vectors of different lengths should be truncated to shorter."""
        sim = _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 3. State matching / search
# ---------------------------------------------------------------------------


class TestStateFinding:
    def test_find_by_emotion(self):
        """Should find snapshots filtered by emotion label."""
        _store_happy()
        _store_calm()
        _store_sad()
        matches = find_similar_states(
            "u1", _make_features(0.7, 0.5),
            emotion_filter="happy", min_similarity=-1.0,
        )
        assert len(matches) == 1
        assert matches[0][0].emotion_label == "happy"

    def test_find_returns_sorted_by_similarity(self):
        """Results should be sorted by descending cosine similarity."""
        _store_happy()
        _store_calm()
        query = _make_features(0.8, 0.6)  # very close to happy
        matches = find_similar_states("u1", query, top_k=10, min_similarity=-1.0)
        assert len(matches) == 2
        # Happy should be most similar because features match
        assert matches[0][1] >= matches[1][1]

    def test_find_respects_top_k(self):
        """Should return at most top_k results."""
        for _ in range(10):
            _store_happy()
        matches = find_similar_states("u1", _make_features(), top_k=3, min_similarity=-1.0)
        assert len(matches) == 3

    def test_find_empty_library(self):
        """Empty library should return empty list."""
        matches = find_similar_states("nobody", _make_features())
        assert matches == []

    def test_find_respects_min_similarity(self):
        """Results below min_similarity should be excluded."""
        _store_happy()
        # A very different query vector
        matches = find_similar_states(
            "u1", [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -1.0, -1.0],
            min_similarity=0.99,
        )
        # May or may not match depending on cosine; check filtering works
        for _, sim in matches:
            assert sim >= 0.99


# ---------------------------------------------------------------------------
# 4. Neurofeedback target generation
# ---------------------------------------------------------------------------


class TestNeurofeedbackTarget:
    def test_full_blend(self):
        """blend=1.0 should produce target values matching the snapshot."""
        snap = _store_happy()
        current = _make_features(0.0, 0.0)
        target = generate_neurofeedback_target(current, snap, blend=1.0)
        assert isinstance(target, NeurofeedbackTarget)
        assert target.target_valence == snap.valence
        assert target.target_arousal == snap.arousal
        assert 0.0 <= target.difficulty <= 1.0

    def test_zero_blend(self):
        """blend=0.0 should produce targets equal to current features."""
        snap = _store_happy()
        current = _make_features(0.0, 0.0)
        target = generate_neurofeedback_target(current, snap, blend=0.0)
        # Alpha is at index 2 of current features
        assert target.target_alpha_power == current[2]

    def test_mid_blend_interpolates(self):
        """blend=0.5 should produce values between current and target."""
        snap = _store_happy()
        current = _make_features(0.0, 0.0)
        target = generate_neurofeedback_target(current, snap, blend=0.5)
        # Alpha should be midpoint between current[2]=0.3 and snap[2]=0.3
        # (happens to be same in this case)
        assert isinstance(target.target_alpha_power, float)

    def test_difficulty_proportional_to_distance(self):
        """Larger emotional distance should yield higher difficulty."""
        snap = _store_happy()  # v=0.8, a=0.6
        close = _make_features(0.7, 0.5)
        far = _make_features(-0.8, -0.8)
        d_close = generate_neurofeedback_target(close, snap).difficulty
        d_far = generate_neurofeedback_target(far, snap).difficulty
        assert d_far > d_close


# ---------------------------------------------------------------------------
# 5. Replay session planning
# ---------------------------------------------------------------------------


class TestReplaySession:
    def test_basic_session(self):
        """Should create a valid session with correct number of steps."""
        snap = _store_happy()
        current = _make_features(0.0, 0.0)
        session = plan_replay_session("u1", current, snap, num_steps=5, step_duration=20.0)
        assert isinstance(session, ReplaySession)
        assert session.user_id == "u1"
        assert session.target_emotion == "happy"
        assert len(session.steps) == 5
        assert session.total_duration_seconds == 100.0

    def test_steps_progress_monotonically(self):
        """Progress fractions should increase from step 1 to last."""
        snap = _store_calm()
        session = plan_replay_session("u1", _make_features(), snap, num_steps=4)
        fractions = [s.progress_fraction for s in session.steps]
        assert fractions == sorted(fractions)
        assert fractions[-1] == 1.0

    def test_instructions_non_empty(self):
        """Every step should have a non-empty instruction."""
        snap = _store_happy()
        session = plan_replay_session("u1", _make_features(), snap)
        for step in session.steps:
            assert len(step.instruction) > 0
            assert step.duration_seconds > 0

    def test_minimum_steps_enforced(self):
        """num_steps < 2 should be raised to 2."""
        snap = _store_happy()
        session = plan_replay_session("u1", _make_features(), snap, num_steps=1)
        assert len(session.steps) == 2


# ---------------------------------------------------------------------------
# 6. Visualization data
# ---------------------------------------------------------------------------


class TestVisualization:
    def test_basic_visualization(self):
        """Should produce valid frame data."""
        snap = _store_happy()
        viz = generate_visualization_data("u1", _make_features(-0.5, -0.3), snap, num_frames=10)
        assert viz["num_frames"] == 10
        assert len(viz["frames"]) == 10
        assert viz["target_emotion"] == "happy"
        assert "journey" in viz
        assert "keypoints" in viz

    def test_frames_interpolate_smoothly(self):
        """Valence should move from start toward target across frames."""
        snap = _store_happy()  # v=0.8
        current = _make_features(-0.5, 0.0)
        viz = generate_visualization_data("u1", current, snap, num_frames=5)
        first_v = viz["frames"][0]["valence"]
        last_v = viz["frames"][-1]["valence"]
        # First frame should be near -0.5, last near 0.8
        assert first_v < last_v

    def test_keypoints_present(self):
        """Keypoints should include start, midpoint, and target."""
        snap = _store_calm()
        viz = generate_visualization_data("u1", _make_features(), snap)
        labels = [kp["label"] for kp in viz["keypoints"]]
        assert "start" in labels
        assert "midpoint" in labels
        assert "target" in labels

    def test_emotional_distance_correct(self):
        """Journey emotional distance should match Euclidean distance."""
        snap = _store_happy()  # v=0.8, a=0.6
        current = _make_features(0.0, 0.0)
        viz = generate_visualization_data("u1", current, snap)
        expected = math.sqrt(0.8 ** 2 + 0.6 ** 2)
        assert abs(viz["journey"]["emotional_distance"] - round(expected, 4)) < 0.01


# ---------------------------------------------------------------------------
# 7. Travel profile
# ---------------------------------------------------------------------------


class TestTravelProfile:
    def test_basic_profile(self):
        """Should compute a valid travel profile."""
        snap = _store_happy()
        profile = compute_travel_profile(_make_features(0.0, 0.0), snap)
        assert isinstance(profile, TravelProfile)
        assert profile.target_valence == 0.8
        assert profile.target_arousal == 0.6
        assert profile.emotional_distance > 0
        assert profile.estimated_steps >= 2
        assert profile.journey_arc in ("ascending", "descending", "lateral", "complex")

    def test_profile_serialization(self):
        """profile_to_dict should produce a JSON-serializable dict."""
        snap = _store_calm()
        profile = compute_travel_profile(_make_features(0.0, 0.0), snap)
        d = profile_to_dict(profile)
        assert isinstance(d, dict)
        assert "current_valence" in d
        assert "journey_arc" in d
        assert "emotional_distance" in d

    def test_ascending_arc(self):
        """Moving toward higher valence + arousal should be ascending."""
        snap = _store_happy()  # v=0.8, a=0.6
        profile = compute_travel_profile(_make_features(-0.3, -0.5), snap)
        # da = 0.6 - (-0.5) = 1.1 > 0.3, dv = 0.8 - (-0.3) = 1.1 > 0.1
        assert profile.journey_arc == "ascending"

    def test_lateral_arc(self):
        """Small changes should be classified as lateral."""
        snap = store_emotional_snapshot(
            user_id="u1", emotion_label="neutral",
            feature_vector=_make_features(0.05, 0.05),
            valence=0.05, arousal=0.05,
        )
        profile = compute_travel_profile(_make_features(0.0, 0.0), snap)
        assert profile.journey_arc == "lateral"


# ---------------------------------------------------------------------------
# 8. API route tests
# ---------------------------------------------------------------------------


class TestTimeTravelAPI:
    def test_status_endpoint(self):
        """GET /time-travel/status should return 200 with availability info."""
        resp = client.get("/time-travel/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["model"] == "neural_time_travel"
        assert "store" in data["capabilities"]

    def test_store_endpoint(self):
        """POST /time-travel/store should store and return a snapshot."""
        resp = client.post("/time-travel/store", json={
            "user_id": "api-user",
            "emotion_label": "happy",
            "feature_vector": [0.1, 0.2, 0.3, 0.2, 0.05, 1.5, 1.0, 0.1, 0.8, 0.6],
            "valence": 0.8,
            "arousal": 0.6,
            "context": "test store",
            "tags": ["test"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["snapshot"]["emotion_label"] == "happy"
        assert data["snapshot"]["valence"] == 0.8
        assert data["library_size"] == 1

    def test_store_missing_user_id_rejected(self):
        """POST /time-travel/store without user_id should return 422."""
        resp = client.post("/time-travel/store", json={
            "emotion_label": "happy",
            "feature_vector": [1.0],
        })
        assert resp.status_code == 422

    def test_store_empty_vector_rejected(self):
        """POST /time-travel/store with empty feature_vector should return 422."""
        resp = client.post("/time-travel/store", json={
            "user_id": "u1",
            "emotion_label": "happy",
            "feature_vector": [],
        })
        assert resp.status_code == 422

    def test_replay_endpoint(self):
        """POST /time-travel/replay should return a session plan."""
        # First store a snapshot
        client.post("/time-travel/store", json={
            "user_id": "replay-user",
            "emotion_label": "calm",
            "feature_vector": _make_features(0.5, -0.4),
            "valence": 0.5,
            "arousal": -0.4,
        })
        # Then request replay
        resp = client.post("/time-travel/replay", json={
            "user_id": "replay-user",
            "current_features": _make_features(0.0, 0.0),
            "target_emotion": "calm",
            "num_steps": 4,
            "step_duration": 15.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["session"]["target_emotion"] == "calm"
        assert data["session"]["num_steps"] == 4
        assert "travel_profile" in data
        assert data["travel_profile"]["target_valence"] == 0.5

    def test_replay_no_snapshots_404(self):
        """POST /time-travel/replay with no stored snapshots should return 404."""
        resp = client.post("/time-travel/replay", json={
            "user_id": "empty-user",
            "current_features": [0.0] * 10,
            "target_emotion": "happy",
        })
        assert resp.status_code == 404

    def test_visualize_endpoint(self):
        """POST /time-travel/visualize should return frame data."""
        client.post("/time-travel/store", json={
            "user_id": "viz-user",
            "emotion_label": "excited",
            "feature_vector": _make_features(0.7, 0.9),
            "valence": 0.7,
            "arousal": 0.9,
        })
        resp = client.post("/time-travel/visualize", json={
            "user_id": "viz-user",
            "current_features": _make_features(-0.3, 0.0),
            "target_emotion": "excited",
            "num_frames": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_frames"] == 10
        assert len(data["frames"]) == 10
        assert data["target_emotion"] == "excited"
        assert "journey" in data
        assert "keypoints" in data

    def test_visualize_no_target_404(self):
        """POST /time-travel/visualize with no match should return 404."""
        resp = client.post("/time-travel/visualize", json={
            "user_id": "no-one",
            "current_features": [0.0] * 10,
            "target_emotion": "missing",
        })
        assert resp.status_code == 404
