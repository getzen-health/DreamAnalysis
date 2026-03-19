"""Privacy-preserving federated learning for cross-user EEG emotion model improvement.

Enables multiple users to collaboratively improve a shared emotion classifier
without sharing raw EEG data.  Each user trains locally, computes weight deltas,
and submits privacy-protected updates to a central aggregator.

Key mechanisms:
  - Federated averaging (FedAvg, McMahan et al. 2017): weighted aggregation of
    model updates from N users without sharing raw data.
  - Differential privacy (Dwork & Roth 2014): calibrated Gaussian noise added to
    gradients before sharing, satisfying (epsilon, delta)-DP guarantees.
  - Privacy budget tracking: cumulative epsilon per user; updates are refused
    once the budget is exhausted, preventing gradual information leakage.
  - Model versioning: tracks global model rounds and per-user contribution counts.

Architecture:
    User device -> train_local_model() on private EEG data
                -> compute_weight_delta() = new_weights - global_weights
                -> apply_differential_privacy() adds calibrated noise
                -> POST delta to /federated/local-update
    Server      -> aggregate_deltas() via weighted FedAvg
                -> broadcast updated global model
                -> raw EEG never stored on server

References:
    McMahan et al. (2017) -- Communication-Efficient Learning of Deep Networks
    Dwork & Roth (2014) -- The Algorithmic Foundations of Differential Privacy
    Abadi et al. (2016) -- Deep Learning with Differential Privacy
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# -- Constants ----------------------------------------------------------------

N_FEATURES = 17
N_CLASSES = 6  # happy, sad, angry, fear, surprise, neutral

DEFAULT_EPSILON = 1.0        # per-round DP epsilon
DEFAULT_DELTA = 1e-5         # per-round DP delta
DEFAULT_SENSITIVITY = 1.0    # L2 sensitivity of clipped gradients
MAX_PRIVACY_BUDGET = 10.0    # cumulative epsilon cap per user
DEFAULT_LOCAL_EPOCHS = 5     # local SGD iterations per round
DEFAULT_LEARNING_RATE = 0.01
MIN_SAMPLES_FOR_TRAINING = 10


# -- Data classes -------------------------------------------------------------

@dataclass
class LocalUpdate:
    """A single local model update submitted by a user."""
    user_id: str
    weight_delta: Dict[str, np.ndarray]
    n_samples: int
    quality_score: float = 1.0  # 0-1, higher = cleaner data / better signal


@dataclass
class FederatedRound:
    """Record of a single aggregation round."""
    round_id: int
    n_participants: int
    participant_ids: List[str]
    total_samples: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class PrivacyBudget:
    """Per-user cumulative privacy budget tracker."""
    user_id: str
    epsilon_spent: float = 0.0
    delta_spent: float = 0.0
    n_contributions: int = 0
    max_epsilon: float = MAX_PRIVACY_BUDGET
    created_at: float = field(default_factory=time.time)
    last_contribution_at: Optional[float] = None

    @property
    def epsilon_remaining(self) -> float:
        return max(0.0, self.max_epsilon - self.epsilon_spent)

    @property
    def is_exhausted(self) -> bool:
        return self.epsilon_spent >= self.max_epsilon


@dataclass
class FederatedStatus:
    """Global federated learning status snapshot."""
    current_round: int
    total_participants: int
    total_contributions: int
    pending_updates: int
    global_model_available: bool
    rounds_history: List[FederatedRound]


# -- Core state ---------------------------------------------------------------

_global_weights: Optional[Dict[str, np.ndarray]] = None
_pending_updates: List[LocalUpdate] = []
_rounds_history: List[FederatedRound] = []
_privacy_budgets: Dict[str, PrivacyBudget] = {}
_contribution_counts: Dict[str, int] = {}
_current_round: int = 0


def _reset_state() -> None:
    """Reset all module-level state.  Used by tests."""
    global _global_weights, _pending_updates, _rounds_history
    global _privacy_budgets, _contribution_counts, _current_round
    _global_weights = None
    _pending_updates = []
    _rounds_history = []
    _privacy_budgets = {}
    _contribution_counts = {}
    _current_round = 0


# -- Local model training -----------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def train_local_model(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: Optional[Dict[str, np.ndarray]] = None,
    n_epochs: int = DEFAULT_LOCAL_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> Dict[str, np.ndarray]:
    """Train a logistic regression model on local user data.

    Args:
        features: (n_samples, N_FEATURES) EEG feature matrix.
        labels: (n_samples,) integer class labels in [0, N_CLASSES).
        initial_weights: Starting weights (from global model).  If None,
            Xavier-initialized from scratch.
        n_epochs: Number of gradient descent passes over the data.
        learning_rate: Step size for SGD.

    Returns:
        Dict with keys "W" and "b" -- the trained weight matrix and bias.
    """
    n_samples = features.shape[0]
    if n_samples < MIN_SAMPLES_FOR_TRAINING:
        raise ValueError(
            f"Need at least {MIN_SAMPLES_FOR_TRAINING} samples, got {n_samples}"
        )

    # Initialize weights
    if initial_weights is not None:
        W = initial_weights["W"].copy().astype(np.float32)
        b = initial_weights["b"].copy().astype(np.float32)
    else:
        scale = np.sqrt(2.0 / (N_FEATURES + N_CLASSES))
        W = np.random.randn(N_FEATURES, N_CLASSES).astype(np.float32) * scale
        b = np.zeros(N_CLASSES, dtype=np.float32)

    X = features.astype(np.float32)

    # One-hot encode labels
    Y = np.zeros((n_samples, N_CLASSES), dtype=np.float32)
    for i, lbl in enumerate(labels):
        if 0 <= int(lbl) < N_CLASSES:
            Y[i, int(lbl)] = 1.0

    # Mini-batch SGD
    for _ in range(n_epochs):
        logits = X @ W + b
        probs = _softmax(logits)
        error = probs - Y
        grad_W = (X.T @ error) / n_samples
        grad_b = np.mean(error, axis=0)
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b

    return {"W": W, "b": b}


# -- Weight delta computation --------------------------------------------------

def compute_weight_delta(
    trained_weights: Dict[str, np.ndarray],
    global_weights: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute the difference between locally trained weights and the global model.

    Args:
        trained_weights: Weights after local training.
        global_weights: Weights of the current global model.

    Returns:
        Dict of numpy arrays representing the delta (trained - global).
    """
    delta: Dict[str, np.ndarray] = {}
    for key in trained_weights:
        if key in global_weights:
            delta[key] = trained_weights[key] - global_weights[key]
        else:
            delta[key] = trained_weights[key].copy()
    return delta


# -- Differential privacy ------------------------------------------------------

def apply_differential_privacy(
    weight_delta: Dict[str, np.ndarray],
    epsilon: float = DEFAULT_EPSILON,
    delta: float = DEFAULT_DELTA,
    sensitivity: float = DEFAULT_SENSITIVITY,
    clip_norm: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Add calibrated Gaussian noise to weight deltas for (epsilon, delta)-DP.

    Implements the Gaussian mechanism:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Gradients are first clipped to bound sensitivity, then noise is added.

    Args:
        weight_delta: Raw weight deltas from local training.
        epsilon: Privacy parameter (smaller = more private).
        delta: Privacy failure probability.
        sensitivity: L2 sensitivity after clipping.
        clip_norm: Maximum L2 norm for gradient clipping.

    Returns:
        Noised weight delta satisfying (epsilon, delta)-DP.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("Delta must be in (0, 1)")

    # Clip gradients to bound sensitivity
    clipped = _clip_weight_delta(weight_delta, clip_norm)

    # Compute noise scale via Gaussian mechanism
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    noised: Dict[str, np.ndarray] = {}
    for key, arr in clipped.items():
        noise = np.random.normal(0, sigma, size=arr.shape).astype(arr.dtype)
        noised[key] = arr + noise

    return noised


def _clip_weight_delta(
    weight_delta: Dict[str, np.ndarray],
    max_norm: float,
) -> Dict[str, np.ndarray]:
    """Clip weight delta so its overall L2 norm does not exceed max_norm."""
    # Compute global L2 norm across all parameter arrays
    total_sq = sum(np.sum(arr ** 2) for arr in weight_delta.values())
    global_norm = math.sqrt(float(total_sq))

    if global_norm <= max_norm:
        return {k: v.copy() for k, v in weight_delta.items()}

    scale = max_norm / global_norm
    return {k: v * scale for k, v in weight_delta.items()}


# -- Privacy budget tracking ---------------------------------------------------

def compute_privacy_budget(user_id: str) -> PrivacyBudget:
    """Get or create the privacy budget for a user.

    Returns the current PrivacyBudget dataclass.
    """
    if user_id not in _privacy_budgets:
        _privacy_budgets[user_id] = PrivacyBudget(user_id=user_id)
    return _privacy_budgets[user_id]


def _record_privacy_spend(
    user_id: str,
    epsilon: float,
    delta: float,
) -> PrivacyBudget:
    """Record that a user spent privacy budget for one round.

    Uses simple sequential composition: total epsilon = sum of per-round epsilons.
    """
    budget = compute_privacy_budget(user_id)
    budget.epsilon_spent += epsilon
    budget.delta_spent += delta
    budget.n_contributions += 1
    budget.last_contribution_at = time.time()
    return budget


# -- Federated aggregation -----------------------------------------------------

def aggregate_deltas(
    updates: Optional[List[LocalUpdate]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Aggregate pending local updates via weighted FedAvg.

    Each update is weighted by (n_samples * quality_score), so users with
    more data and higher signal quality have proportionally more influence.

    If *updates* is None, uses and clears the module-level _pending_updates.

    Returns:
        Aggregated weight delta dict, or None if no updates available.
    """
    global _global_weights, _pending_updates, _current_round

    source = updates if updates is not None else _pending_updates

    if not source:
        return None

    # Compute weights for weighted average
    raw_weights = [
        u.n_samples * u.quality_score for u in source
    ]
    total_weight = sum(raw_weights)
    if total_weight <= 0:
        return None

    normalized = [w / total_weight for w in raw_weights]

    # Compute weighted average of deltas
    ref_delta = source[0].weight_delta
    aggregated: Dict[str, np.ndarray] = {
        k: np.zeros_like(v) for k, v in ref_delta.items()
    }

    for update, w in zip(source, normalized):
        for key in aggregated:
            if key in update.weight_delta:
                aggregated[key] = aggregated[key] + w * update.weight_delta[key]

    # Apply aggregated delta to global model
    if _global_weights is None:
        # First round: initialize from aggregated delta as base
        _global_weights = {k: v.copy() for k, v in aggregated.items()}
    else:
        for key in _global_weights:
            if key in aggregated:
                _global_weights[key] = _global_weights[key] + aggregated[key]

    # Record round
    _current_round += 1
    participant_ids = [u.user_id for u in source]
    total_samples = sum(u.n_samples for u in source)
    round_record = FederatedRound(
        round_id=_current_round,
        n_participants=len(source),
        participant_ids=participant_ids,
        total_samples=total_samples,
    )
    _rounds_history.append(round_record)

    # Update contribution counts
    for u in source:
        _contribution_counts[u.user_id] = _contribution_counts.get(u.user_id, 0) + 1

    # Clear pending updates only when we consumed the module-level list
    if updates is None:
        _pending_updates = []

    logger.info(
        "Federated round %d complete: %d participants, %d total samples",
        _current_round,
        len(source),
        total_samples,
    )

    return aggregated


def submit_local_update(
    user_id: str,
    weight_delta: Dict[str, np.ndarray],
    n_samples: int,
    quality_score: float = 1.0,
    epsilon: float = DEFAULT_EPSILON,
    delta: float = DEFAULT_DELTA,
) -> Dict:
    """Submit a local model update to the pending queue.

    Validates privacy budget, records spend, and enqueues the update.

    Returns:
        Dict with success status and metadata.
    """
    # Check privacy budget
    budget = compute_privacy_budget(user_id)
    if budget.is_exhausted:
        return {
            "accepted": False,
            "reason": "privacy_budget_exhausted",
            "epsilon_remaining": 0.0,
            "message": (
                f"Privacy budget exhausted for user {user_id}. "
                f"Total epsilon spent: {budget.epsilon_spent:.2f} / {budget.max_epsilon:.2f}."
            ),
        }

    if budget.epsilon_remaining < epsilon:
        return {
            "accepted": False,
            "reason": "insufficient_privacy_budget",
            "epsilon_remaining": budget.epsilon_remaining,
            "epsilon_requested": epsilon,
            "message": (
                f"Insufficient privacy budget. Remaining: {budget.epsilon_remaining:.2f}, "
                f"requested: {epsilon:.2f}."
            ),
        }

    # Record privacy spend
    _record_privacy_spend(user_id, epsilon, delta)

    # Create and enqueue update
    update = LocalUpdate(
        user_id=user_id,
        weight_delta=weight_delta,
        n_samples=n_samples,
        quality_score=quality_score,
    )
    _pending_updates.append(update)

    return {
        "accepted": True,
        "user_id": user_id,
        "n_samples": n_samples,
        "pending_count": len(_pending_updates),
        "epsilon_spent_this_round": epsilon,
        "epsilon_remaining": compute_privacy_budget(user_id).epsilon_remaining,
    }


# -- Global model status -------------------------------------------------------

def get_global_model_status() -> FederatedStatus:
    """Return the current federated learning status."""
    return FederatedStatus(
        current_round=_current_round,
        total_participants=len(_contribution_counts),
        total_contributions=sum(_contribution_counts.values()),
        pending_updates=len(_pending_updates),
        global_model_available=_global_weights is not None,
        rounds_history=list(_rounds_history),
    )


def get_global_weights() -> Optional[Dict[str, np.ndarray]]:
    """Return the current global model weights, or None if not yet initialized."""
    if _global_weights is None:
        return None
    return {k: v.copy() for k, v in _global_weights.items()}


# -- Serialization -------------------------------------------------------------

def federated_to_dict(status: FederatedStatus) -> Dict:
    """Serialize a FederatedStatus to a JSON-safe dict."""
    return {
        "current_round": status.current_round,
        "total_participants": status.total_participants,
        "total_contributions": status.total_contributions,
        "pending_updates": status.pending_updates,
        "global_model_available": status.global_model_available,
        "rounds": [
            {
                "round_id": r.round_id,
                "n_participants": r.n_participants,
                "participant_ids": r.participant_ids,
                "total_samples": r.total_samples,
                "timestamp": r.timestamp,
            }
            for r in status.rounds_history
        ],
    }


def privacy_budget_to_dict(budget: PrivacyBudget) -> Dict:
    """Serialize a PrivacyBudget to a JSON-safe dict."""
    return {
        "user_id": budget.user_id,
        "epsilon_spent": round(budget.epsilon_spent, 4),
        "delta_spent": round(budget.delta_spent, 8),
        "epsilon_remaining": round(budget.epsilon_remaining, 4),
        "max_epsilon": budget.max_epsilon,
        "n_contributions": budget.n_contributions,
        "is_exhausted": budget.is_exhausted,
        "created_at": budget.created_at,
        "last_contribution_at": budget.last_contribution_at,
    }
