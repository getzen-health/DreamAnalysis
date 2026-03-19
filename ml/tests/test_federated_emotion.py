"""Tests for federated emotion model -- privacy-preserving cross-user EEG improvement."""

import math

import numpy as np
import pytest


# -- Helpers -------------------------------------------------------------------

def _make_features(n_samples: int = 20, n_features: int = 17) -> np.ndarray:
    return np.random.randn(n_samples, n_features).astype(np.float32)


def _make_labels(n_samples: int = 20, n_classes: int = 6) -> np.ndarray:
    return np.random.randint(0, n_classes, size=n_samples).astype(np.int32)


def _make_weights() -> dict:
    return {
        "W": np.random.randn(17, 6).astype(np.float32) * 0.1,
        "b": np.zeros(6, dtype=np.float32),
    }


def _make_delta() -> dict:
    return {
        "W": np.random.randn(17, 6).astype(np.float32) * 0.01,
        "b": np.random.randn(6).astype(np.float32) * 0.01,
    }


# -- Module reset fixture -----------------------------------------------------

@pytest.fixture(autouse=True)
def reset_federated_state():
    """Reset all module-level state between tests."""
    from models.federated_emotion import _reset_state
    _reset_state()
    yield
    _reset_state()


# -- train_local_model ---------------------------------------------------------

class TestTrainLocalModel:
    def test_basic_training(self):
        from models.federated_emotion import train_local_model
        features = _make_features(30)
        labels = _make_labels(30)
        result = train_local_model(features, labels)
        assert "W" in result
        assert "b" in result
        assert result["W"].shape == (17, 6)
        assert result["b"].shape == (6,)

    def test_training_with_initial_weights(self):
        from models.federated_emotion import train_local_model
        features = _make_features(20)
        labels = _make_labels(20)
        init = _make_weights()
        result = train_local_model(features, labels, initial_weights=init)
        # Should produce different weights than the initial
        assert not np.allclose(result["W"], init["W"])

    def test_too_few_samples_raises(self):
        from models.federated_emotion import train_local_model
        features = _make_features(5)
        labels = _make_labels(5)
        with pytest.raises(ValueError, match="at least"):
            train_local_model(features, labels)


# -- compute_weight_delta ------------------------------------------------------

class TestComputeWeightDelta:
    def test_delta_is_difference(self):
        from models.federated_emotion import compute_weight_delta
        trained = _make_weights()
        global_w = _make_weights()
        delta = compute_weight_delta(trained, global_w)
        np.testing.assert_allclose(
            delta["W"], trained["W"] - global_w["W"], atol=1e-6
        )
        np.testing.assert_allclose(
            delta["b"], trained["b"] - global_w["b"], atol=1e-6
        )

    def test_delta_zero_when_identical(self):
        from models.federated_emotion import compute_weight_delta
        w = _make_weights()
        delta = compute_weight_delta(w, w)
        assert np.allclose(delta["W"], 0.0)
        assert np.allclose(delta["b"], 0.0)


# -- apply_differential_privacy ------------------------------------------------

class TestApplyDifferentialPrivacy:
    def test_noise_added(self):
        from models.federated_emotion import apply_differential_privacy
        delta = _make_delta()
        noised = apply_differential_privacy(delta, epsilon=1.0, delta=1e-5)
        # Noised values should differ from originals
        assert not np.allclose(noised["W"], delta["W"])

    def test_higher_epsilon_less_noise(self):
        from models.federated_emotion import apply_differential_privacy
        np.random.seed(42)
        delta = _make_delta()
        np.random.seed(42)
        noised_low = apply_differential_privacy(
            {k: v.copy() for k, v in delta.items()},
            epsilon=0.1, delta=1e-5,
        )
        np.random.seed(42)
        noised_high = apply_differential_privacy(
            {k: v.copy() for k, v in delta.items()},
            epsilon=10.0, delta=1e-5,
        )
        # Deviation from original should be larger for low epsilon
        dev_low = np.abs(noised_low["W"] - delta["W"]).mean()
        dev_high = np.abs(noised_high["W"] - delta["W"]).mean()
        assert dev_low > dev_high

    def test_invalid_epsilon_raises(self):
        from models.federated_emotion import apply_differential_privacy
        delta = _make_delta()
        with pytest.raises(ValueError, match="[Ee]psilon"):
            apply_differential_privacy(delta, epsilon=-1.0)

    def test_invalid_delta_raises(self):
        from models.federated_emotion import apply_differential_privacy
        delta = _make_delta()
        with pytest.raises(ValueError, match="[Dd]elta"):
            apply_differential_privacy(delta, epsilon=1.0, delta=0.0)

    def test_gradient_clipping(self):
        from models.federated_emotion import _clip_weight_delta
        delta = {"W": np.ones((17, 6), dtype=np.float32) * 100.0}
        clipped = _clip_weight_delta(delta, max_norm=1.0)
        total_sq = np.sum(clipped["W"] ** 2)
        assert math.sqrt(float(total_sq)) <= 1.0 + 1e-6


# -- Privacy budget ------------------------------------------------------------

class TestPrivacyBudget:
    def test_new_user_full_budget(self):
        from models.federated_emotion import compute_privacy_budget
        budget = compute_privacy_budget("new_user")
        assert budget.epsilon_spent == 0.0
        assert budget.epsilon_remaining == 10.0
        assert not budget.is_exhausted

    def test_budget_decreases_after_spend(self):
        from models.federated_emotion import (
            compute_privacy_budget, _record_privacy_spend,
        )
        _record_privacy_spend("user_1", epsilon=2.0, delta=1e-5)
        budget = compute_privacy_budget("user_1")
        assert budget.epsilon_spent == 2.0
        assert budget.epsilon_remaining == 8.0
        assert budget.n_contributions == 1

    def test_budget_exhaustion(self):
        from models.federated_emotion import (
            compute_privacy_budget, _record_privacy_spend,
        )
        for _ in range(10):
            _record_privacy_spend("user_2", epsilon=1.0, delta=1e-5)
        budget = compute_privacy_budget("user_2")
        assert budget.is_exhausted
        assert budget.epsilon_remaining == 0.0


# -- submit_local_update -------------------------------------------------------

class TestSubmitLocalUpdate:
    def test_submit_accepted(self):
        from models.federated_emotion import submit_local_update
        delta = {k: v for k, v in _make_delta().items()}
        result = submit_local_update("user_a", delta, n_samples=50)
        assert result["accepted"] is True
        assert result["pending_count"] == 1

    def test_submit_rejected_when_budget_exhausted(self):
        from models.federated_emotion import (
            submit_local_update, _record_privacy_spend,
        )
        for _ in range(10):
            _record_privacy_spend("user_b", epsilon=1.0, delta=1e-5)
        delta = _make_delta()
        result = submit_local_update("user_b", delta, n_samples=50)
        assert result["accepted"] is False
        assert result["reason"] == "privacy_budget_exhausted"

    def test_submit_rejected_insufficient_budget(self):
        from models.federated_emotion import (
            submit_local_update, _record_privacy_spend,
        )
        _record_privacy_spend("user_c", epsilon=9.5, delta=1e-5)
        delta = _make_delta()
        result = submit_local_update("user_c", delta, n_samples=50, epsilon=1.0)
        assert result["accepted"] is False
        assert result["reason"] == "insufficient_privacy_budget"


# -- aggregate_deltas ----------------------------------------------------------

class TestAggregate:
    def test_aggregate_with_two_updates(self):
        from models.federated_emotion import (
            submit_local_update, aggregate_deltas, get_global_model_status,
        )
        for uid in ["u1", "u2"]:
            submit_local_update(uid, _make_delta(), n_samples=50)

        result = aggregate_deltas()
        assert result is not None
        status = get_global_model_status()
        assert status.current_round == 1
        assert status.total_participants == 2

    def test_aggregate_empty_returns_none(self):
        from models.federated_emotion import aggregate_deltas
        assert aggregate_deltas() is None

    def test_weighted_aggregation(self):
        from models.federated_emotion import aggregate_deltas, LocalUpdate
        d1 = {"W": np.ones((17, 6), dtype=np.float32)}
        d2 = {"W": np.ones((17, 6), dtype=np.float32) * 3.0}
        updates = [
            LocalUpdate("u1", d1, n_samples=100, quality_score=1.0),
            LocalUpdate("u2", d2, n_samples=100, quality_score=1.0),
        ]
        result = aggregate_deltas(updates)
        # Equal weights -> average = 2.0
        np.testing.assert_allclose(result["W"], 2.0, atol=1e-6)

    def test_quality_weighted_aggregation(self):
        from models.federated_emotion import aggregate_deltas, LocalUpdate
        d1 = {"W": np.zeros((17, 6), dtype=np.float32)}
        d2 = {"W": np.ones((17, 6), dtype=np.float32) * 10.0}
        updates = [
            LocalUpdate("u1", d1, n_samples=100, quality_score=0.0),
            LocalUpdate("u2", d2, n_samples=100, quality_score=1.0),
        ]
        result = aggregate_deltas(updates)
        # u1 has quality=0 -> weight=0, u2 dominates
        np.testing.assert_allclose(result["W"], 10.0, atol=1e-6)


# -- get_global_model_status ---------------------------------------------------

class TestGlobalModelStatus:
    def test_initial_status(self):
        from models.federated_emotion import get_global_model_status
        status = get_global_model_status()
        assert status.current_round == 0
        assert status.total_participants == 0
        assert status.pending_updates == 0
        assert not status.global_model_available

    def test_status_after_aggregation(self):
        from models.federated_emotion import (
            submit_local_update, aggregate_deltas, get_global_model_status,
        )
        submit_local_update("u1", _make_delta(), 30)
        submit_local_update("u2", _make_delta(), 40)
        aggregate_deltas()
        status = get_global_model_status()
        assert status.current_round == 1
        assert status.global_model_available is True
        assert status.total_contributions == 2


# -- federated_to_dict / privacy_budget_to_dict --------------------------------

class TestSerialization:
    def test_federated_to_dict(self):
        from models.federated_emotion import (
            get_global_model_status, federated_to_dict,
        )
        status = get_global_model_status()
        d = federated_to_dict(status)
        assert "current_round" in d
        assert "rounds" in d
        assert isinstance(d["rounds"], list)

    def test_privacy_budget_to_dict(self):
        from models.federated_emotion import (
            compute_privacy_budget, privacy_budget_to_dict,
        )
        budget = compute_privacy_budget("test_user")
        d = privacy_budget_to_dict(budget)
        assert d["user_id"] == "test_user"
        assert d["epsilon_remaining"] == 10.0
        assert d["is_exhausted"] is False


# -- Route integration tests ---------------------------------------------------

class TestRoutes:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from api.routes.federated_emotion import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_status_endpoint(self, client):
        resp = client.get("/federated-emotion/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_round" in data
        assert data["current_round"] == 0

    def test_privacy_budget_endpoint(self, client):
        resp = client.get("/federated-emotion/privacy-budget/new_user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "new_user"
        assert data["epsilon_remaining"] == 10.0

    def test_local_update_endpoint(self, client):
        features = _make_features(15).tolist()
        labels = _make_labels(15).tolist()
        resp = client.post("/federated-emotion/local-update", json={
            "user_id": "test_user",
            "features": features,
            "labels": labels,
            "quality_score": 0.9,
            "epsilon": 1.0,
            "delta": 1e-5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True

    def test_aggregate_endpoint_insufficient(self, client):
        resp = client.post("/federated-emotion/aggregate", json={
            "min_updates": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["reason"] == "insufficient_updates"

    def test_full_round_trip(self, client):
        """Submit two updates, aggregate, check status."""
        for uid in ["user_x", "user_y"]:
            features = _make_features(15).tolist()
            labels = _make_labels(15).tolist()
            resp = client.post("/federated-emotion/local-update", json={
                "user_id": uid,
                "features": features,
                "labels": labels,
            })
            assert resp.status_code == 200
            assert resp.json()["accepted"] is True

        # Aggregate
        resp = client.post("/federated-emotion/aggregate", json={
            "min_updates": 2,
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Check status
        resp = client.get("/federated-emotion/status")
        data = resp.json()
        assert data["current_round"] == 1
        assert data["global_model_available"] is True
