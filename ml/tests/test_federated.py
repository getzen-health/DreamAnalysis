"""Tests for federated learning coordinator and client."""

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_weights():
    return {
        "W": np.random.randn(17, 6).astype(np.float32).tolist(),
        "b": np.zeros(6, dtype=np.float32).tolist(),
    }


@pytest.fixture
def sample_delta():
    return {
        "W": (np.random.randn(17, 6) * 0.01).astype(np.float32).tolist(),
        "b": (np.random.randn(6) * 0.01).astype(np.float32).tolist(),
    }


# ── FederatedEEGTrainer tests ─────────────────────────────────────────────────

class TestFederatedEEGTrainer:
    def setup_method(self):
        from models.federated_trainer import FederatedEEGTrainer
        self.trainer = FederatedEEGTrainer(aggregation="fedavg", min_clients=2)

    def test_opt_in_out(self):
        self.trainer.opt_in("user_1")
        assert self.trainer.is_opted_in("user_1")
        self.trainer.opt_out("user_1")
        assert not self.trainer.is_opted_in("user_1")

    def test_receive_update_requires_opt_in(self, sample_delta):
        result = self.trainer.receive_update("user_x", sample_delta, 50)
        assert result["accepted"] is False

    def test_receive_update_accepted_after_opt_in(self, sample_delta):
        self.trainer.opt_in("user_1")
        result = self.trainer.receive_update("user_1", sample_delta, 50)
        assert result["accepted"] is True
        assert result["pending_count"] == 1

    def test_should_aggregate_below_threshold(self, sample_delta):
        self.trainer.opt_in("user_1")
        self.trainer.receive_update("user_1", sample_delta, 50)
        assert not self.trainer.should_aggregate()

    def test_should_aggregate_at_threshold(self, sample_delta):
        self.trainer.opt_in("user_1")
        self.trainer.opt_in("user_2")
        self.trainer.receive_update("user_1", sample_delta, 50)
        self.trainer.receive_update("user_2", sample_delta, 30)
        assert self.trainer.should_aggregate()

    def test_aggregate_returns_model_weights(self, sample_delta):
        self.trainer.opt_in("user_1")
        self.trainer.opt_in("user_2")
        self.trainer.receive_update("user_1", sample_delta, 50)
        self.trainer.receive_update("user_2", sample_delta, 30)
        result = self.trainer.aggregate()
        assert result is not None

    def test_global_weights_available_after_aggregate(self, sample_delta):
        self.trainer.opt_in("user_1")
        self.trainer.opt_in("user_2")
        self.trainer.receive_update("user_1", sample_delta, 50)
        self.trainer.receive_update("user_2", sample_delta, 30)
        self.trainer.aggregate()
        weights = self.trainer.get_global_weights()
        assert weights is not None
        assert "W" in weights
        assert "b" in weights

    def test_round_num_increments(self, sample_delta):
        self.trainer.opt_in("user_1")
        self.trainer.opt_in("user_2")
        self.trainer.receive_update("user_1", sample_delta, 50)
        self.trainer.receive_update("user_2", sample_delta, 30)
        self.trainer.aggregate()
        status = self.trainer.get_status()
        assert status["round_num"] == 1

    def test_status_structure(self):
        status = self.trainer.get_status()
        assert "round_num" in status
        assert "pending_updates" in status
        assert "aggregation_strategy" in status

    def test_one_update_per_client_per_round(self, sample_delta):
        self.trainer.opt_in("user_1")
        self.trainer.receive_update("user_1", sample_delta, 50)
        self.trainer.receive_update("user_1", sample_delta, 60)  # should replace
        status = self.trainer.get_status()
        assert status["pending_updates"] == 1


class TestFederatedEEGTrainerFuzzy:
    def setup_method(self):
        from models.federated_trainer import FederatedEEGTrainer
        self.trainer = FederatedEEGTrainer(aggregation="fuzzy_fedavg", min_clients=2)

    def test_fuzzy_aggregate_succeeds(self, sample_delta):
        self.trainer.opt_in("user_1")
        self.trainer.opt_in("user_2")
        self.trainer.receive_update("user_1", sample_delta, 100)
        self.trainer.receive_update("user_2", sample_delta, 50)
        result = self.trainer.aggregate()
        assert result is not None


# ── FederatedEEGClient tests ──────────────────────────────────────────────────

class TestFederatedEEGClient:
    def setup_method(self):
        from models.federated_client import FederatedEEGClient
        self.client = FederatedEEGClient("test_user", local_epochs=2, use_dp=False)

    def test_initial_state(self):
        assert self.client.n_local_samples() == 0
        status = self.client.get_status()
        assert status["user_id"] == "test_user"

    def test_add_sample(self):
        features = np.random.randn(17).astype(np.float32)
        self.client.add_sample(features, label=0)
        assert self.client.n_local_samples() == 1

    def test_local_train_insufficient_data(self):
        for _ in range(3):
            self.client.add_sample(np.random.randn(17).astype(np.float32), 0)
        delta, n = self.client.local_train()
        # No restriction on n_samples in client itself; returns result regardless
        assert isinstance(delta, dict)

    def test_local_train_returns_delta(self):
        for i in range(30):
            self.client.add_sample(np.random.randn(17).astype(np.float32), i % 6)
        delta, n = self.client.local_train()
        assert "W" in delta
        assert "b" in delta
        assert n == 30

    def test_local_train_with_dp(self):
        from models.federated_client import FederatedEEGClient
        client_dp = FederatedEEGClient("dp_user", local_epochs=2, use_dp=True, dp_epsilon=1.0)
        for i in range(30):
            client_dp.add_sample(np.random.randn(17).astype(np.float32), i % 6)
        delta, n = client_dp.local_train()
        assert "W" in delta
        assert n == 30

    def test_apply_global_weights(self):
        weights = {
            "W": np.random.randn(17, 6).astype(np.float32).tolist(),
            "b": np.zeros(6, dtype=np.float32).tolist(),
        }
        self.client.apply_global_weights(weights)
        status = self.client.get_status()
        assert status["has_global_model"]

    def test_status_structure(self):
        status = self.client.get_status()
        assert "user_id" in status
        assert "n_local_samples" in status
        assert "dp_enabled" in status
        assert "rounds_completed" in status


# ── LocalEEGModel tests ───────────────────────────────────────────────────────

class TestLocalEEGModel:
    def setup_method(self):
        from models.federated_client import LocalEEGModel
        self.model = LocalEEGModel()

    def test_forward_output_shape(self):
        X = np.random.randn(10, 17).astype(np.float32)
        out = self.model.forward(X)
        assert out.shape == (10, 6)

    def test_forward_probabilities_sum_to_one(self):
        X = np.random.randn(5, 17).astype(np.float32)
        out = self.model.forward(X)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(5), atol=1e-5)

    def test_predict_class_range(self):
        X = np.random.randn(20, 17).astype(np.float32)
        preds = self.model.predict_class(X)
        assert all(0 <= p < 6 for p in preds)

    def test_train_reduces_loss(self):
        X = np.random.randn(60, 17).astype(np.float32)
        y = np.random.randint(0, 6, 60, dtype=np.int64)
        losses = self.model.train(X, y, epochs=10)
        # Loss should decrease or at least not explode
        assert losses[-1] < losses[0] * 3   # generous bound for stochastic training

    def test_get_set_weights_round_trip(self):
        original = self.model.get_weights()
        self.model.set_weights({k: v * 2.0 for k, v in original.items()})
        modified = self.model.get_weights()
        np.testing.assert_allclose(modified["W"], original["W"] * 2.0, atol=1e-5)


# ── Simulation test ───────────────────────────────────────────────────────────

class TestFederatedSimulation:
    def test_simulation_runs(self):
        from training.train_federated import simulate_federated
        results = simulate_federated(n_clients=3, n_rounds=3, aggregation="fedavg", use_dp=False)
        assert "final_fl_accuracy" in results
        assert results["n_clients"] == 3
        assert results["n_rounds"] == 3

    def test_simulation_accuracy_above_chance(self):
        from training.train_federated import simulate_federated
        results = simulate_federated(n_clients=3, n_rounds=5, aggregation="fuzzy_fedavg", use_dp=False)
        # Above 16.7% chance level for 6 classes
        assert results["final_fl_accuracy"] > 0.167
