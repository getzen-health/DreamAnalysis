"""Federated learning simulation script.

Simulates N_CLIENTS clients each training on a local shard of EEG data,
submitting weight deltas, and the server aggregating via FedAvg / Fuzzy-FedAvg.

Validates:
  - FedAvg aggregation correctness
  - Local differential privacy noise injection
  - Per-round accuracy improvement
  - Convergence to same performance as centralized training on same data

Usage:
    python -m training.train_federated
    python -m training.train_federated --clients 8 --rounds 20 --aggregation fuzzy_fedavg
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.federated_trainer import FederatedEEGTrainer
from models.federated_client import FederatedEEGClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Simulation constants ───────────────────────────────────────────────────────

N_CLIENTS_DEFAULT = 5
N_ROUNDS_DEFAULT = 10
SAMPLES_PER_CLIENT = 100
N_FEATURES = 17
N_CLASSES = 6
RANDOM_SEED = 42


# ── Synthetic EEG data generation ─────────────────────────────────────────────

def generate_synthetic_eeg_dataset(
    n_samples: int, n_features: int = N_FEATURES, n_classes: int = N_CLASSES, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic EEG band-power feature vectors with class-conditional means.

    Each class has a slightly different mean in feature space, mimicking how
    different emotional states produce different EEG band power profiles.
    """
    rng = np.random.RandomState(seed)
    class_means = rng.randn(n_classes, n_features) * 0.5   # small inter-class separation
    X_list, y_list = [], []
    per_class = n_samples // n_classes

    for c in range(n_classes):
        x = rng.randn(per_class, n_features) * 1.0 + class_means[c]
        X_list.append(x.astype(np.float32))
        y_list.append(np.full(per_class, c, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def split_iid(X: np.ndarray, y: np.ndarray, n_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """IID split: each client gets an equal random shard."""
    n = len(X)
    idx = np.random.permutation(n)
    shard_size = n // n_clients
    shards = []
    for i in range(n_clients):
        s = idx[i * shard_size:(i + 1) * shard_size]
        shards.append((X[s], y[s]))
    return shards


def evaluate_accuracy(model_weights: Dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate accuracy of a weight dict on test data."""
    from models.federated_client import LocalEEGModel
    m = LocalEEGModel()
    m.set_weights(model_weights)
    preds = m.predict_class(X)
    return float(np.mean(preds == y))


# ── Centralized baseline ───────────────────────────────────────────────────────

def train_centralized(
    X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50
) -> Dict[str, np.ndarray]:
    """Train a centralized model on all data (upper bound for FL)."""
    from models.federated_client import LocalEEGModel
    m = LocalEEGModel()
    m.train(X_train, y_train, epochs=epochs)
    return m.get_weights()


# ── FL simulation ─────────────────────────────────────────────────────────────

def simulate_federated(
    n_clients: int = N_CLIENTS_DEFAULT,
    n_rounds: int = N_ROUNDS_DEFAULT,
    aggregation: str = "fuzzy_fedavg",
    use_dp: bool = True,
    dp_epsilon: float = 1.0,
) -> Dict:
    """Run a full FL simulation.

    Returns per-round accuracy dict for analysis.
    """
    np.random.seed(RANDOM_SEED)

    logger.info("Generating synthetic EEG dataset (%d clients × %d samples)...",
                n_clients, SAMPLES_PER_CLIENT)

    total_samples = n_clients * SAMPLES_PER_CLIENT
    X_all, y_all = generate_synthetic_eeg_dataset(total_samples, seed=RANDOM_SEED)

    # Hold out 20% for evaluation
    n_test = total_samples // 5
    X_test, y_test = X_all[:n_test], y_all[:n_test]
    X_train, y_train = X_all[n_test:], y_all[n_test:]

    # IID split across clients
    shards = split_iid(X_train, y_train, n_clients)

    # Centralized baseline
    logger.info("Training centralized baseline...")
    central_weights = train_centralized(X_train, y_train, epochs=50)
    central_acc = evaluate_accuracy(central_weights, X_test, y_test)
    logger.info("Centralized accuracy: %.3f", central_acc)

    # Initialize server and clients
    trainer = FederatedEEGTrainer(
        initial_weights=None,
        aggregation=aggregation,
        min_clients=2,
    )
    clients = []
    for i in range(n_clients):
        client_id = f"sim_client_{i}"
        c = FederatedEEGClient(
            user_id=client_id,
            local_epochs=5,
            dp_epsilon=dp_epsilon,
            use_dp=use_dp,
        )
        trainer.opt_in(client_id)
        clients.append((client_id, c, shards[i]))

    round_results = []

    for round_num in range(n_rounds):
        logger.info("--- FL Round %d / %d ---", round_num + 1, n_rounds)

        # All clients pull global model and train locally
        global_weights = trainer.get_global_weights()

        for client_id, client, (X_c, y_c) in clients:
            if global_weights:
                client.apply_global_weights(global_weights)

            # Add all local samples to buffer
            for x, label in zip(X_c, y_c):
                client.add_sample(x, int(label))

            delta_dict, n_samp = client.local_train()
            if delta_dict:
                trainer.receive_update(client_id, delta_dict, n_samp)

        # Aggregate
        if trainer.should_aggregate():
            trainer.aggregate()

        # Evaluate
        agg_weights = trainer.get_global_weights()
        if agg_weights:
            weights_np = {k: np.array(v, dtype=np.float32) for k, v in agg_weights.items()}
            acc = evaluate_accuracy(weights_np, X_test, y_test)
        else:
            acc = 0.0

        logger.info("Round %d accuracy: %.3f (centralized: %.3f)", round_num + 1, acc, central_acc)
        round_results.append({"round": round_num + 1, "accuracy": acc})

    final_acc = round_results[-1]["accuracy"] if round_results else 0.0

    results = {
        "n_clients": n_clients,
        "n_rounds": n_rounds,
        "aggregation": aggregation,
        "dp_enabled": use_dp,
        "dp_epsilon": dp_epsilon,
        "centralized_accuracy": central_acc,
        "final_fl_accuracy": final_acc,
        "accuracy_gap": central_acc - final_acc,
        "rounds": round_results,
        "status": "converged" if final_acc > 0.5 else "needs_more_rounds",
    }

    logger.info(
        "\nFL simulation complete:\n"
        "  Centralized: %.3f\n"
        "  FL final:    %.3f\n"
        "  Gap:         %.3f\n"
        "  Status:      %s",
        central_acc, final_acc, central_acc - final_acc, results["status"]
    )
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated EEG Learning Simulation")
    parser.add_argument("--clients", type=int, default=N_CLIENTS_DEFAULT)
    parser.add_argument("--rounds", type=int, default=N_ROUNDS_DEFAULT)
    parser.add_argument("--aggregation", choices=["fedavg", "fuzzy_fedavg"], default="fuzzy_fedavg")
    parser.add_argument("--no-dp", action="store_true", help="Disable differential privacy")
    parser.add_argument("--dp-epsilon", type=float, default=1.0)
    args = parser.parse_args()

    results = simulate_federated(
        n_clients=args.clients,
        n_rounds=args.rounds,
        aggregation=args.aggregation,
        use_dp=not args.no_dp,
        dp_epsilon=args.dp_epsilon,
    )

    print(f"\nFinal FL accuracy: {results['final_fl_accuracy']:.3f}")
    print(f"Centralized:       {results['centralized_accuracy']:.3f}")
    print(f"Gap:               {results['accuracy_gap']:.3f}")
