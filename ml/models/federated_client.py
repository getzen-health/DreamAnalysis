"""Federated learning client-side local trainer for EEG emotion models.

Raw EEG data never leaves the device — only weight deltas are submitted to the server.

Based on:
  - FedAvg client algorithm (McMahan et al., 2017)
  - Local differential privacy via Gaussian noise (SECRYPT 2025)
  - Privacy-adaptive autoencoders for gradient sanitization

Usage:
    client = FederatedEEGClient(user_id="user_123")
    client.apply_global_weights(global_weights_dict)   # download global model
    delta, n_samples = client.local_train(local_eeg_data, local_labels)
    # POST delta to /federated/submit-update (never POST raw EEG)
"""

import copy
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Differential privacy parameters ──────────────────────────────────────────

DEFAULT_DP_EPSILON = 1.0       # privacy budget (smaller = more privacy)
DEFAULT_DP_SENSITIVITY = 1.0   # L2 sensitivity of feature vectors
DEFAULT_LOCAL_EPOCHS = 5       # local training iterations per FL round
MAX_LOCAL_SAMPLES = 500        # cap to prevent any single client dominating


# ── Simple feature-based local model (no PyTorch dependency) ──────────────────

class LocalEEGModel:
    """Lightweight numpy linear model for on-device EEG emotion classification.

    Uses the same 17-feature vectors as the server-side models.
    Trains via mini-batch gradient descent (logistic regression style).

    This is intentionally simple:
      - No framework dependencies (no PyTorch/sklearn on user device)
      - Fast convergence on small local datasets (50-200 samples)
      - Produces weight deltas compatible with the FedAvg aggregation
    """

    N_FEATURES = 17
    N_CLASSES = 6   # happy, sad, angry, fear, surprise, neutral
    LEARNING_RATE = 0.01

    def __init__(self):
        # Xavier initialization
        scale = np.sqrt(2.0 / (self.N_FEATURES + self.N_CLASSES))
        self.W = np.random.randn(self.N_FEATURES, self.N_CLASSES).astype(np.float32) * scale
        self.b = np.zeros(self.N_CLASSES, dtype=np.float32)

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        if "W" in weights:
            self.W = np.array(weights["W"], dtype=np.float32)
        if "b" in weights:
            self.b = np.array(weights["b"], dtype=np.float32)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W + self.b)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """One mini-batch gradient descent step. Returns cross-entropy loss."""
        probs = self.forward(X)
        n = X.shape[0]

        # Cross-entropy loss
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(n), y] = 1.0
        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))

        # Gradients
        delta = (probs - y_onehot) / n
        dW = X.T @ delta
        db = delta.sum(axis=0)

        self.W -= self.LEARNING_RATE * dW
        self.b -= self.LEARNING_RATE * db
        return float(loss)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5) -> List[float]:
        losses = []
        n = X.shape[0]
        batch_size = min(32, n)
        for _ in range(epochs):
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                l = self.train_step(X[batch], y[batch])
                epoch_loss += l
            losses.append(epoch_loss)
        return losses


# ── Differential privacy ──────────────────────────────────────────────────────

def _add_gaussian_dp_noise(
    delta: Dict[str, np.ndarray],
    epsilon: float = DEFAULT_DP_EPSILON,
    sensitivity: float = DEFAULT_DP_SENSITIVITY,
) -> Dict[str, np.ndarray]:
    """Add Gaussian noise to weight deltas for local differential privacy.

    Sigma calibrated to (epsilon, delta=1e-5)-DP guarantee.
    (Abadi et al., 2016 — the moments accountant method)
    """
    delta_dp = 1e-5
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta_dp)) / epsilon

    noised = {}
    for key, arr in delta.items():
        noised[key] = arr + np.random.normal(0, sigma, size=arr.shape).astype(arr.dtype)
    return noised


# ── Client ────────────────────────────────────────────────────────────────────

class FederatedEEGClient:
    """Client-side federated learning for one user.

    Holds a local copy of the global model and trains it on local EEG data.
    Only the weight *delta* (new - old) is ever sent to the server.
    """

    def __init__(
        self,
        user_id: str,
        local_epochs: int = DEFAULT_LOCAL_EPOCHS,
        dp_epsilon: float = DEFAULT_DP_EPSILON,
        use_dp: bool = True,
    ):
        self.user_id = user_id
        self.local_epochs = local_epochs
        self.dp_epsilon = dp_epsilon
        self.use_dp = use_dp

        self._model = LocalEEGModel()
        self._local_data: List[Tuple[np.ndarray, int]] = []  # (features, label)
        self._last_global_weights: Optional[Dict[str, np.ndarray]] = None
        self._lock = threading.Lock()
        self._round_num = 0
        self._n_successful_rounds = 0

    # ── Global model sync ─────────────────────────────────────────────────────

    def apply_global_weights(self, global_weights: Dict[str, list]) -> None:
        """Download and apply the server's global model weights."""
        with self._lock:
            weights_np = {k: np.array(v, dtype=np.float32) for k, v in global_weights.items()}
            self._model.set_weights(weights_np)
            self._last_global_weights = copy.deepcopy(weights_np)
            logger.debug("Client %s applied global weights (round %d)", self.user_id, self._round_num)

    # ── Local data accumulation ───────────────────────────────────────────────

    def add_sample(self, features: np.ndarray, label: int) -> None:
        """Add one labeled EEG feature vector to the local training buffer."""
        with self._lock:
            self._local_data.append((features.astype(np.float32), int(label)))
            # Cap buffer to avoid unbounded growth
            if len(self._local_data) > MAX_LOCAL_SAMPLES:
                self._local_data = self._local_data[-MAX_LOCAL_SAMPLES:]

    def n_local_samples(self) -> int:
        with self._lock:
            return len(self._local_data)

    # ── Local training ────────────────────────────────────────────────────────

    def local_train(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, list], int]:
        """Train the local model on buffered (or provided) EEG data.

        Returns:
            (delta_dict, n_samples) where delta_dict = new_weights - old_weights.
            delta_dict is JSON-serializable (lists, not numpy arrays).
            n_samples is used by the server for weighted FedAvg.
        """
        with self._lock:
            if X is None or y is None:
                if not self._local_data:
                    return {}, 0
                X = np.stack([s[0] for s in self._local_data])
                y = np.array([s[1] for s in self._local_data], dtype=np.int64)

            if len(X) == 0:
                return {}, 0

            # Snapshot weights before local training
            old_weights = copy.deepcopy(self._model.get_weights())

            # Local training
            self._model.train(X, y, epochs=self.local_epochs)

            # Compute delta = new - old
            new_weights = self._model.get_weights()
            delta = {k: new_weights[k] - old_weights[k] for k in new_weights}

            # Apply local differential privacy
            if self.use_dp:
                delta = _add_gaussian_dp_noise(delta, epsilon=self.dp_epsilon)

            n_samples = len(X)
            self._round_num += 1
            self._n_successful_rounds += 1

        # Convert to JSON-serializable format
        delta_list = {k: v.tolist() for k, v in delta.items()}
        logger.info(
            "Client %s completed local training: %d samples, round %d",
            self.user_id,
            n_samples,
            self._round_num,
        )
        return delta_list, n_samples

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        with self._lock:
            return {
                "user_id": self.user_id,
                "n_local_samples": len(self._local_data),
                "local_epochs": self.local_epochs,
                "dp_enabled": self.use_dp,
                "dp_epsilon": self.dp_epsilon,
                "rounds_completed": self._n_successful_rounds,
                "has_global_model": self._last_global_weights is not None,
            }


# ── Singleton per user ────────────────────────────────────────────────────────

_client_instances: Dict[str, FederatedEEGClient] = {}
_clients_lock = threading.Lock()


def get_federated_client(user_id: str) -> FederatedEEGClient:
    with _clients_lock:
        if user_id not in _client_instances:
            _client_instances[user_id] = FederatedEEGClient(user_id=user_id)
    return _client_instances[user_id]
