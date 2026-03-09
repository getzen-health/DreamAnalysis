"""Federated learning server-side coordinator for privacy-preserving EEG model training.

Based on:
  - FedAvg (McMahan et al., 2017) — weighted average of client model updates
  - Fuzzy ensemble FL (J. King Saud Univ., 2025) — Gompertz-based fuzzy rank aggregation
  - Privacy-preserving EEG FL (SECRYPT 2025) — gradient-only sharing, no raw EEG

Architecture:
    Client (user device) → trains locally on EEG data (never leaves device)
                        → computes weight delta = new_weights - global_weights
                        → POSTs delta to /federated/submit-update
    Server              → aggregates deltas from N clients via FedAvg
                        → broadcasts updated global model
                        → raw EEG never stored on server
"""

import copy
import logging
import math
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MIN_CLIENTS_TO_AGGREGATE = 2   # need at least 2 updates before FedAvg runs
MAX_ROUNDS_STORED = 100        # cap history to avoid unbounded growth
AGGREGATION_TIMEOUT_SECONDS = 3600  # 1 hour — force aggregate if too much time passes


# ── Global model representation ───────────────────────────────────────────────

class ModelWeights:
    """Lightweight numpy-based model weight container.

    Instead of shipping full PyTorch/sklearn model objects over HTTP,
    we work with flat dicts of numpy arrays (identical to what ONNX/sklearn
    expose via get_params / state_dict).
    """

    def __init__(self, weights: Dict[str, np.ndarray]):
        self.weights = weights
        self.created_at = time.time()

    @staticmethod
    def zero_like(other: "ModelWeights") -> "ModelWeights":
        return ModelWeights({k: np.zeros_like(v) for k, v in other.weights.items()})

    def add_(self, other: "ModelWeights", scale: float = 1.0) -> None:
        for k in self.weights:
            if k in other.weights:
                self.weights[k] = self.weights[k] + scale * other.weights[k]

    def scale_(self, factor: float) -> None:
        for k in self.weights:
            self.weights[k] = self.weights[k] * factor

    def to_dict(self) -> Dict[str, list]:
        return {k: v.tolist() for k, v in self.weights.items()}

    @staticmethod
    def from_dict(d: Dict[str, list]) -> "ModelWeights":
        return ModelWeights({k: np.array(v, dtype=np.float32) for k, v in d.items()})


# ── Client update record ───────────────────────────────────────────────────────

class ClientUpdate:
    """One round of weight deltas submitted by a client."""

    def __init__(
        self,
        client_id: str,
        delta: ModelWeights,
        n_samples: int,
        round_num: int,
        timestamp: float,
    ):
        self.client_id = client_id
        self.delta = delta
        self.n_samples = n_samples          # used as weight in FedAvg
        self.round_num = round_num
        self.timestamp = timestamp


# ── Aggregation strategies ─────────────────────────────────────────────────────

def _fedavg(updates: List[ClientUpdate]) -> ModelWeights:
    """Standard FedAvg: weighted average by sample count."""
    total_samples = sum(u.n_samples for u in updates)
    if total_samples == 0:
        total_samples = len(updates)

    result = ModelWeights.zero_like(updates[0].delta)
    for u in updates:
        weight = u.n_samples / total_samples
        result.add_(u.delta, scale=weight)
    return result


def _gompertz_rank_weight(rank: int, n_clients: int) -> float:
    """Gompertz function weight for fuzzy rank aggregation (J. King Saud Univ., 2025).

    Higher-ranked clients (better local accuracy) get exponentially more weight.
    rank=1 is best. Returns weight in [0, 1].
    """
    a, b, c = 1.0, -1.0, 0.5
    x = rank / max(n_clients, 1)
    return a * math.exp(b * math.exp(c * x))


def _fuzzy_fedavg(updates: List[ClientUpdate]) -> ModelWeights:
    """Fuzzy ensemble: Gompertz-ranked weighted average by n_samples."""
    # Rank clients by n_samples (more data = better rank = rank 1 is highest)
    sorted_updates = sorted(updates, key=lambda u: u.n_samples, reverse=True)
    n = len(sorted_updates)

    raw_weights = [_gompertz_rank_weight(i + 1, n) for i in range(n)]
    total_weight = sum(raw_weights)

    result = ModelWeights.zero_like(updates[0].delta)
    for u, w in zip(sorted_updates, raw_weights):
        result.add_(u.delta, scale=w / total_weight)
    return result


# ── Main coordinator ───────────────────────────────────────────────────────────

class FederatedEEGTrainer:
    """Server-side federated learning coordinator.

    Thread-safe via a single lock. Intended to be instantiated once per
    FastAPI process (singleton via get_federated_trainer()).

    Usage:
        trainer = FederatedEEGTrainer(initial_weights, aggregation='fuzzy_fedavg')
        trainer.receive_update(client_id, delta_dict, n_samples)
        if trainer.should_aggregate():
            trainer.aggregate()
        global_weights = trainer.get_global_weights()
    """

    def __init__(
        self,
        initial_weights: Optional[Dict[str, list]] = None,
        aggregation: str = "fuzzy_fedavg",
        min_clients: int = MIN_CLIENTS_TO_AGGREGATE,
    ):
        self._lock = threading.Lock()
        self.aggregation = aggregation
        self.min_clients = min_clients

        self._global_weights: Optional[ModelWeights] = (
            ModelWeights.from_dict(initial_weights) if initial_weights else None
        )
        self._pending_updates: List[ClientUpdate] = []
        self._round_num: int = 0
        self._round_history: List[Dict] = []   # for /federated/status
        self._opted_in: Dict[str, bool] = {}   # client_id → consent

    # ── Consent ──────────────────────────────────────────────────────────────

    def opt_in(self, client_id: str) -> None:
        with self._lock:
            self._opted_in[client_id] = True
            logger.info("Client %s opted in to federated learning", client_id)

    def opt_out(self, client_id: str) -> None:
        with self._lock:
            self._opted_in[client_id] = False
            logger.info("Client %s opted out of federated learning", client_id)

    def is_opted_in(self, client_id: str) -> bool:
        with self._lock:
            return self._opted_in.get(client_id, False)

    # ── Update receiving ─────────────────────────────────────────────────────

    def receive_update(
        self,
        client_id: str,
        delta_dict: Dict[str, list],
        n_samples: int,
    ) -> Dict:
        """Accept a weight delta from a client.

        Returns status dict: round_num, pending_count, will_aggregate.
        """
        if not self.is_opted_in(client_id):
            return {"accepted": False, "reason": "client not opted in"}

        with self._lock:
            delta = ModelWeights.from_dict(delta_dict)
            update = ClientUpdate(
                client_id=client_id,
                delta=delta,
                n_samples=max(n_samples, 1),
                round_num=self._round_num,
                timestamp=time.time(),
            )
            # One update per client per round — replace if exists
            self._pending_updates = [
                u for u in self._pending_updates if u.client_id != client_id
            ]
            self._pending_updates.append(update)
            pending = len(self._pending_updates)

        will_agg = pending >= self.min_clients
        return {
            "accepted": True,
            "round_num": self._round_num,
            "pending_count": pending,
            "will_aggregate": will_agg,
        }

    # ── Aggregation ───────────────────────────────────────────────────────────

    def should_aggregate(self) -> bool:
        with self._lock:
            if len(self._pending_updates) < self.min_clients:
                return False
            # Force aggregate if oldest update is too old
            oldest = min(u.timestamp for u in self._pending_updates)
            if time.time() - oldest > AGGREGATION_TIMEOUT_SECONDS:
                return True
            return len(self._pending_updates) >= self.min_clients

    def aggregate(self) -> Optional[ModelWeights]:
        """Run FedAvg/fuzzy aggregation on pending updates.

        Returns the delta to apply to the global model, or None if not ready.
        """
        with self._lock:
            if not self._pending_updates:
                return None

            updates = list(self._pending_updates)
            self._pending_updates = []

        if self.aggregation == "fuzzy_fedavg":
            aggregated_delta = _fuzzy_fedavg(updates)
        else:
            aggregated_delta = _fedavg(updates)

        with self._lock:
            if self._global_weights is None:
                # Bootstrap: use first aggregated delta as the global model
                self._global_weights = aggregated_delta
            else:
                self._global_weights.add_(aggregated_delta)

            self._round_num += 1
            self._round_history.append({
                "round": self._round_num,
                "n_clients": len(updates),
                "total_samples": sum(u.n_samples for u in updates),
                "timestamp": time.time(),
                "aggregation": self.aggregation,
            })
            if len(self._round_history) > MAX_ROUNDS_STORED:
                self._round_history = self._round_history[-MAX_ROUNDS_STORED:]

            logger.info(
                "FL round %d complete: %d clients, %d total samples",
                self._round_num,
                len(updates),
                sum(u.n_samples for u in updates),
            )
            return copy.deepcopy(self._global_weights)

    # ── Broadcasting ──────────────────────────────────────────────────────────

    def get_global_weights(self) -> Optional[Dict[str, list]]:
        """Return current global model weights as JSON-serializable dict."""
        with self._lock:
            if self._global_weights is None:
                return None
            return self._global_weights.to_dict()

    def get_status(self) -> Dict:
        with self._lock:
            return {
                "round_num": self._round_num,
                "pending_updates": len(self._pending_updates),
                "min_clients_required": self.min_clients,
                "aggregation_strategy": self.aggregation,
                "opted_in_clients": sum(1 for v in self._opted_in.values() if v),
                "has_global_model": self._global_weights is not None,
                "recent_rounds": self._round_history[-5:],
            }


# ── Singleton ──────────────────────────────────────────────────────────────────

_trainer_instance: Optional[FederatedEEGTrainer] = None
_trainer_lock = threading.Lock()


def get_federated_trainer() -> FederatedEEGTrainer:
    global _trainer_instance
    with _trainer_lock:
        if _trainer_instance is None:
            _trainer_instance = FederatedEEGTrainer(
                initial_weights=None,
                aggregation="fuzzy_fedavg",
                min_clients=MIN_CLIENTS_TO_AGGREGATE,
            )
    return _trainer_instance
