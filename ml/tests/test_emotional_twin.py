"""Tests for the emotional digital twin model and API routes (issue #419)."""

from __future__ import annotations

import math
import time
from dataclasses import asdict

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def twin():
    from models.emotional_twin import EmotionalTwin
    return EmotionalTwin(user_id="test-user")


@pytest.fixture
def sample_contexts():
    from models.emotional_twin import ContextVector
    return [
        ContextVector(stress_level=0.2, sleep_quality=0.8, exercise=0.5),
        ContextVector(stress_level=0.5, sleep_quality=0.6, exercise=0.3),
        ContextVector(stress_level=0.8, sleep_quality=0.3, exercise=0.1),
        ContextVector(stress_level=0.3, sleep_quality=0.7, exercise=0.6),
        ContextVector(stress_level=0.1, sleep_quality=0.9, exercise=0.7),
    ]


@pytest.fixture
def sample_states():
    from models.emotional_twin import EmotionalState
    return [
        EmotionalState(valence=0.5, arousal=0.4, stress=0.2, energy=0.7),
        EmotionalState(valence=0.3, arousal=0.5, stress=0.4, energy=0.5),
        EmotionalState(valence=-0.2, arousal=0.7, stress=0.7, energy=0.3),
        EmotionalState(valence=0.2, arousal=0.4, stress=0.3, energy=0.6),
        EmotionalState(valence=0.6, arousal=0.3, stress=0.1, energy=0.8),
    ]


# ---------------------------------------------------------------------------
# EmotionalState
# ---------------------------------------------------------------------------

class TestEmotionalState:
    def test_default_values(self):
        from models.emotional_twin import EmotionalState
        s = EmotionalState()
        assert s.valence == 0.0
        assert s.arousal == 0.5
        assert s.stress == 0.3
        assert s.energy == 0.5

    def test_to_array_shape(self):
        from models.emotional_twin import EmotionalState
        s = EmotionalState(valence=0.1, arousal=0.2, stress=0.3, energy=0.4)
        arr = s.to_array()
        assert arr.shape == (4,)
        np.testing.assert_allclose(arr, [0.1, 0.2, 0.3, 0.4])

    def test_from_array_clips_bounds(self):
        from models.emotional_twin import EmotionalState
        arr = np.array([2.0, -0.5, 1.5, -1.0])
        s = EmotionalState.from_array(arr)
        assert s.valence == 1.0
        assert s.arousal == 0.0
        assert s.stress == 1.0
        assert s.energy == 0.0

    def test_roundtrip_array(self):
        from models.emotional_twin import EmotionalState
        original = EmotionalState(valence=-0.3, arousal=0.8, stress=0.1, energy=0.9)
        restored = EmotionalState.from_array(original.to_array())
        assert restored.valence == pytest.approx(original.valence)
        assert restored.arousal == pytest.approx(original.arousal)


# ---------------------------------------------------------------------------
# ContextVector
# ---------------------------------------------------------------------------

class TestContextVector:
    def test_to_array_shape(self):
        from models.emotional_twin import ContextVector
        c = ContextVector()
        assert c.to_array().shape == (6,)

    def test_from_array(self):
        from models.emotional_twin import ContextVector
        arr = np.array([0.1, 0.2, -0.3, 0.4, 0.5, 0.6])
        c = ContextVector.from_array(arr)
        assert c.stress_level == pytest.approx(0.1)
        assert c.social_change == pytest.approx(-0.3)
        assert c.exercise == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# train_twin
# ---------------------------------------------------------------------------

class TestTrainTwin:
    def test_updates_weights(self, twin, sample_contexts, sample_states):
        from models.emotional_twin import train_twin, _POPULATION_WEIGHTS
        initial_weights = twin.weights.copy()
        train_twin(twin, sample_contexts, sample_states)
        # Weights should have changed from population priors
        assert not np.allclose(twin.weights, initial_weights)
        assert twin.n_updates > 0

    def test_records_prediction_errors(self, twin, sample_contexts, sample_states):
        from models.emotional_twin import train_twin
        train_twin(twin, sample_contexts, sample_states)
        assert len(twin.prediction_errors) == len(sample_contexts) - 1

    def test_updates_current_state(self, twin, sample_contexts, sample_states):
        from models.emotional_twin import train_twin
        train_twin(twin, sample_contexts, sample_states)
        # Current state should be set to last observed
        assert twin.current_state.valence == pytest.approx(sample_states[-1].valence)

    def test_too_few_samples_returns_unchanged(self, twin):
        from models.emotional_twin import ContextVector, EmotionalState, train_twin
        # Only 1 sample -- not enough for transition pairs
        train_twin(twin, [ContextVector()], [EmotionalState()])
        assert twin.n_updates == 0
        assert twin.prediction_errors == []

    def test_multiple_training_rounds_accumulate(self, twin, sample_contexts, sample_states):
        from models.emotional_twin import train_twin
        train_twin(twin, sample_contexts[:3], sample_states[:3])
        n1 = twin.n_updates
        train_twin(twin, sample_contexts[2:], sample_states[2:])
        assert twin.n_updates > n1

    def test_prediction_error_decreases_with_training(self):
        """Training twice on the same data should reduce recent error."""
        from models.emotional_twin import (
            ContextVector, EmotionalState, EmotionalTwin, train_twin,
        )
        # Create consistent training data with a clear pattern
        contexts = [
            ContextVector(stress_level=0.1, sleep_quality=0.9, exercise=0.8),
            ContextVector(stress_level=0.1, sleep_quality=0.9, exercise=0.8),
            ContextVector(stress_level=0.1, sleep_quality=0.9, exercise=0.8),
        ]
        states = [
            EmotionalState(valence=0.5, arousal=0.3, stress=0.1, energy=0.8),
            EmotionalState(valence=0.6, arousal=0.3, stress=0.1, energy=0.85),
            EmotionalState(valence=0.65, arousal=0.3, stress=0.1, energy=0.87),
        ]
        twin = EmotionalTwin(user_id="learner", learning_rate=0.1)
        # Train multiple passes on same data
        for _ in range(5):
            train_twin(twin, contexts, states)
        first_errors = twin.prediction_errors[:2]
        last_errors = twin.prediction_errors[-2:]
        assert np.mean(last_errors) <= np.mean(first_errors)


# ---------------------------------------------------------------------------
# simulate_scenario
# ---------------------------------------------------------------------------

class TestSimulateScenario:
    def test_returns_trajectory(self, twin, sample_contexts, sample_states):
        from models.emotional_twin import (
            ScenarioDefinition, simulate_scenario, train_twin,
        )
        train_twin(twin, sample_contexts, sample_states)
        scenario = ScenarioDefinition(name="high-stress", duration_steps=3,
                                      context_overrides={"stress_level": 0.9})
        result = simulate_scenario(twin, scenario)
        # trajectory has initial state + duration_steps entries
        assert len(result.trajectory) == 4
        assert result.scenario_name == "high-stress"
        assert result.duration_steps == 3

    def test_high_stress_reduces_valence(self):
        """High stress with population priors should decrease valence."""
        from models.emotional_twin import (
            EmotionalState, EmotionalTwin,
            ScenarioDefinition, simulate_scenario,
        )
        # Use a fresh twin with population priors (no training noise)
        # and a known positive starting state
        t = EmotionalTwin(user_id="stress-test")
        t.current_state = EmotionalState(valence=0.5, arousal=0.4, stress=0.2, energy=0.7)
        scenario = ScenarioDefinition(
            name="extreme-stress",
            duration_steps=10,
            context_overrides={"stress_level": 1.0, "sleep_quality": 0.0},
        )
        result = simulate_scenario(t, scenario)
        initial_v = result.initial_state["valence"]
        final_v = result.final_state["valence"]
        assert final_v < initial_v

    def test_default_context_used_when_none(self, twin):
        from models.emotional_twin import ScenarioDefinition, simulate_scenario
        scenario = ScenarioDefinition(name="baseline", duration_steps=2)
        result = simulate_scenario(twin, scenario)
        assert result.trajectory is not None
        assert len(result.trajectory) == 3

    def test_scenario_with_no_overrides(self, twin, sample_contexts, sample_states):
        from models.emotional_twin import (
            ScenarioDefinition, simulate_scenario, train_twin,
        )
        train_twin(twin, sample_contexts, sample_states)
        scenario = ScenarioDefinition(name="neutral", duration_steps=2)
        result = simulate_scenario(twin, scenario)
        assert "valence" in result.final_state


# ---------------------------------------------------------------------------
# predict_trajectory
# ---------------------------------------------------------------------------

class TestPredictTrajectory:
    def test_length_matches_contexts_plus_one(self, twin, sample_contexts):
        from models.emotional_twin import predict_trajectory
        traj = predict_trajectory(twin, sample_contexts)
        assert len(traj) == len(sample_contexts) + 1

    def test_first_entry_is_current_state(self, twin, sample_contexts):
        from models.emotional_twin import predict_trajectory
        traj = predict_trajectory(twin, sample_contexts)
        assert traj[0]["valence"] == pytest.approx(twin.current_state.valence)

    def test_empty_contexts_returns_just_current(self, twin):
        from models.emotional_twin import predict_trajectory
        traj = predict_trajectory(twin, [])
        assert len(traj) == 1


# ---------------------------------------------------------------------------
# compute_calibration_score
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_no_data_returns_zero(self, twin):
        from models.emotional_twin import compute_calibration_score
        cal = compute_calibration_score(twin)
        assert cal["calibration_score"] == 0.0
        assert cal["enough_data"] is False
        assert cal["mean_error"] is None

    def test_after_training_has_score(self, twin, sample_contexts, sample_states):
        from models.emotional_twin import compute_calibration_score, train_twin
        train_twin(twin, sample_contexts, sample_states)
        cal = compute_calibration_score(twin)
        assert 0.0 < cal["calibration_score"] <= 1.0
        assert cal["mean_error"] is not None
        assert cal["n_updates"] > 0

    def test_perfect_prediction_gives_high_score(self):
        from models.emotional_twin import EmotionalTwin, compute_calibration_score
        twin = EmotionalTwin(user_id="perfect")
        # Simulate near-zero errors
        twin.prediction_errors = [0.001] * 20
        twin.n_updates = 20
        cal = compute_calibration_score(twin)
        assert cal["calibration_score"] > 0.9
        assert cal["enough_data"] is True


# ---------------------------------------------------------------------------
# twin_to_dict
# ---------------------------------------------------------------------------

class TestTwinToDict:
    def test_returns_serializable_dict(self, twin, sample_contexts, sample_states):
        from models.emotional_twin import train_twin, twin_to_dict
        train_twin(twin, sample_contexts, sample_states)
        d = twin_to_dict(twin)
        assert d["user_id"] == "test-user"
        assert "current_state" in d
        assert "calibration" in d
        assert isinstance(d["weights_shape"], list)
        assert d["weights_shape"] == [4, 10]

    def test_untrained_twin_serialises(self, twin):
        from models.emotional_twin import twin_to_dict
        d = twin_to_dict(twin)
        assert d["n_updates"] == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_extreme_context_values(self, twin):
        from models.emotional_twin import (
            ContextVector, EmotionalState, train_twin,
        )
        contexts = [
            ContextVector(stress_level=1.0, novelty=1.0, social_change=1.0,
                          routine_disruption=1.0, sleep_quality=0.0, exercise=0.0),
            ContextVector(stress_level=0.0, novelty=0.0, social_change=-1.0,
                          routine_disruption=0.0, sleep_quality=1.0, exercise=1.0),
        ]
        states = [
            EmotionalState(valence=-1.0, arousal=1.0, stress=1.0, energy=0.0),
            EmotionalState(valence=1.0, arousal=0.0, stress=0.0, energy=1.0),
        ]
        # Should not raise
        train_twin(twin, contexts, states)
        assert twin.n_updates == 1

    def test_state_bounds_enforced_after_transition(self, twin):
        from models.emotional_twin import (
            ContextVector, predict_trajectory,
        )
        # Push context to extremes
        extreme = [ContextVector(stress_level=1.0, sleep_quality=0.0)] * 20
        traj = predict_trajectory(twin, extreme)
        for step in traj:
            assert -1.0 <= step["valence"] <= 1.0
            assert 0.0 <= step["arousal"] <= 1.0
            assert 0.0 <= step["stress"] <= 1.0
            assert 0.0 <= step["energy"] <= 1.0


# ---------------------------------------------------------------------------
# API route tests (FastAPI TestClient)
# ---------------------------------------------------------------------------

class TestAPIRoutes:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes.emotional_twin import router, _twins
        _twins.clear()
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_status_endpoint(self, client):
        resp = client.get("/emotional-twin/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "available"
        assert data["active_twins"] == 0

    def test_train_endpoint(self, client):
        payload = {
            "user_id": "api-user",
            "contexts": [
                {"stress_level": 0.2, "sleep_quality": 0.8},
                {"stress_level": 0.5, "sleep_quality": 0.5},
                {"stress_level": 0.7, "sleep_quality": 0.3},
            ],
            "observed_states": [
                {"valence": 0.5, "arousal": 0.4, "stress": 0.2, "energy": 0.7},
                {"valence": 0.3, "arousal": 0.5, "stress": 0.4, "energy": 0.5},
                {"valence": -0.1, "arousal": 0.6, "stress": 0.6, "energy": 0.3},
            ],
        }
        resp = client.post("/emotional-twin/train", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["twin"]["user_id"] == "api-user"
        assert data["twin"]["n_updates"] > 0

    def test_simulate_requires_trained_twin(self, client):
        payload = {
            "user_id": "nonexistent",
            "scenario": {"name": "test", "duration_steps": 3},
        }
        resp = client.post("/emotional-twin/simulate", json=payload)
        assert resp.status_code == 404

    def test_simulate_after_train(self, client):
        # Train first
        train_payload = {
            "user_id": "sim-user",
            "contexts": [
                {"stress_level": 0.2},
                {"stress_level": 0.5},
                {"stress_level": 0.8},
            ],
            "observed_states": [
                {"valence": 0.5, "arousal": 0.4, "stress": 0.2, "energy": 0.7},
                {"valence": 0.3, "arousal": 0.5, "stress": 0.4, "energy": 0.5},
                {"valence": -0.1, "arousal": 0.7, "stress": 0.7, "energy": 0.3},
            ],
        }
        resp = client.post("/emotional-twin/train", json=train_payload)
        assert resp.status_code == 200

        # Now simulate
        sim_payload = {
            "user_id": "sim-user",
            "scenario": {
                "name": "good-sleep",
                "context_overrides": {"sleep_quality": 1.0, "exercise": 0.8},
                "duration_steps": 5,
            },
        }
        resp = client.post("/emotional-twin/simulate", json=sim_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["result"]["trajectory"]) == 6  # initial + 5 steps

    def test_train_mismatched_lengths_returns_400(self, client):
        payload = {
            "user_id": "bad-user",
            "contexts": [
                {"stress_level": 0.2},
                {"stress_level": 0.3},
                {"stress_level": 0.4},
            ],
            "observed_states": [
                {"valence": 0.5},
                {"valence": 0.3},
            ],
        }
        resp = client.post("/emotional-twin/train", json=payload)
        assert resp.status_code == 400
