"""Emotional digital twin -- learns a user's personal emotional response function.

Maps context vectors (stress_level, novelty, social_change, routine_disruption,
sleep_quality, exercise) to emotional state transitions using a simple temporal
model:

    state_{t+1} = f(context_t, state_t)

The transition function is a weighted linear combination with per-user weights.
Starts with population priors and personalises via Bayesian update as user data
accumulates.  Supports counterfactual simulation: "how would I feel if X
happened?"

Issue #419.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------

CONTEXT_FIELDS: Tuple[str, ...] = (
    "stress_level",
    "novelty",
    "social_change",
    "routine_disruption",
    "sleep_quality",
    "exercise",
)

STATE_FIELDS: Tuple[str, ...] = ("valence", "arousal", "stress", "energy")


@dataclass
class EmotionalState:
    """Instantaneous emotional state vector."""

    valence: float = 0.0   # -1 (negative) .. +1 (positive)
    arousal: float = 0.5   # 0 (calm) .. 1 (activated)
    stress: float = 0.3    # 0 (none) .. 1 (extreme)
    energy: float = 0.5    # 0 (depleted) .. 1 (full)

    def to_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.stress, self.energy],
                        dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "EmotionalState":
        return cls(
            valence=float(np.clip(arr[0], -1.0, 1.0)),
            arousal=float(np.clip(arr[1], 0.0, 1.0)),
            stress=float(np.clip(arr[2], 0.0, 1.0)),
            energy=float(np.clip(arr[3], 0.0, 1.0)),
        )


@dataclass
class ContextVector:
    """External context that may drive emotional change."""

    stress_level: float = 0.0       # 0..1
    novelty: float = 0.0            # 0..1
    social_change: float = 0.0      # -1..1 (negative=isolation, positive=connection)
    routine_disruption: float = 0.0 # 0..1
    sleep_quality: float = 0.5      # 0..1
    exercise: float = 0.0           # 0..1

    def to_array(self) -> np.ndarray:
        return np.array([
            self.stress_level,
            self.novelty,
            self.social_change,
            self.routine_disruption,
            self.sleep_quality,
            self.exercise,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ContextVector":
        return cls(
            stress_level=float(arr[0]),
            novelty=float(arr[1]),
            social_change=float(arr[2]),
            routine_disruption=float(arr[3]),
            sleep_quality=float(arr[4]),
            exercise=float(arr[5]),
        )


@dataclass
class ScenarioDefinition:
    """Counterfactual scenario to simulate."""

    name: str = "unnamed"
    context_overrides: Optional[Dict[str, float]] = None
    duration_steps: int = 5
    description: str = ""


@dataclass
class SimulationResult:
    """Result of running a counterfactual scenario."""

    scenario_name: str
    trajectory: List[Dict[str, float]]
    final_state: Dict[str, float]
    duration_steps: int
    initial_state: Dict[str, float]


# ---------------------------------------------------------------------------
# Population priors for the transition model
# ---------------------------------------------------------------------------

# Weights shape: (n_state, n_context + n_state)
# Each row produces the *delta* for one state dimension.
# Columns: [stress_level, novelty, social_change, routine_disruption,
#            sleep_quality, exercise, valence, arousal, stress, energy]

_N_CTX = len(CONTEXT_FIELDS)   # 6
_N_STATE = len(STATE_FIELDS)   # 4
_INPUT_DIM = _N_CTX + _N_STATE # 10

# Population average weights -- reasonable starting point.
# Row order: delta_valence, delta_arousal, delta_stress, delta_energy
_POPULATION_WEIGHTS = np.array([
    # stress  novelty social  routine sleep  exercise | val   aro   stress energy
    [-0.20,   0.05,   0.15,  -0.05,   0.15,  0.10,    0.40, -0.05, -0.10,  0.05],  # d_valence
    [ 0.10,   0.15,   0.05,   0.10,  -0.05,  0.15,    0.05,  0.30,  0.10, -0.05],  # d_arousal
    [ 0.25,   0.05,  -0.10,   0.15,  -0.15, -0.10,   -0.05,  0.10,  0.30, -0.05],  # d_stress
    [-0.15,   0.05,   0.10,  -0.10,   0.20,  0.20,    0.10, -0.05, -0.15,  0.30],  # d_energy
], dtype=np.float64)

# Population bias (mean delta per step at zero input)
_POPULATION_BIAS = np.zeros(_N_STATE, dtype=np.float64)

# Learning rate for delta-based Bayesian update
_DEFAULT_LR = 0.05

# Momentum toward population prior (regularisation)
_PRIOR_STRENGTH = 0.1


# ---------------------------------------------------------------------------
# EmotionalTwin
# ---------------------------------------------------------------------------

@dataclass
class EmotionalTwin:
    """Per-user emotional digital twin.

    Maintains personalised transition weights, current state estimate,
    and calibration tracking.
    """

    user_id: str
    weights: np.ndarray = field(default_factory=lambda: _POPULATION_WEIGHTS.copy())
    bias: np.ndarray = field(default_factory=lambda: _POPULATION_BIAS.copy())
    current_state: EmotionalState = field(default_factory=EmotionalState)
    prediction_errors: List[float] = field(default_factory=list)
    n_updates: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    learning_rate: float = _DEFAULT_LR


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _transition(
    state: np.ndarray,
    context: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Compute next state from current state + context.

    state_{t+1} = clip(state_t + W @ [context_t ; state_t] + b)
    """
    x = np.concatenate([context, state])
    delta = weights @ x + bias
    # Dampen deltas so single steps don't swing wildly
    delta = np.clip(delta, -0.3, 0.3)
    next_state = state + delta
    # Enforce bounds
    next_state[0] = np.clip(next_state[0], -1.0, 1.0)  # valence
    next_state[1:] = np.clip(next_state[1:], 0.0, 1.0)  # arousal/stress/energy
    return next_state


def train_twin(
    twin: EmotionalTwin,
    contexts: List[ContextVector],
    observed_states: List[EmotionalState],
) -> EmotionalTwin:
    """Train/update the twin from longitudinal (context, observed_state) pairs.

    Uses online gradient descent with L2 regularisation toward population
    priors (Bayesian flavour).

    Args:
        twin: Existing twin to update in-place.
        contexts: Sequence of context vectors (chronological).
        observed_states: Corresponding observed emotional states.

    Returns:
        The same twin object, updated.
    """
    if len(contexts) < 2 or len(observed_states) < 2:
        logger.warning("train_twin requires >= 2 time-steps; skipping.")
        return twin

    n = min(len(contexts), len(observed_states))
    lr = twin.learning_rate

    for t in range(n - 1):
        ctx_arr = contexts[t].to_array()
        state_arr = observed_states[t].to_array()
        target = observed_states[t + 1].to_array()

        predicted = _transition(state_arr, ctx_arr, twin.weights, twin.bias)
        error = target - predicted
        twin.prediction_errors.append(float(np.mean(error ** 2)))

        # Gradient update (MSE loss, partial derivatives)
        x = np.concatenate([ctx_arr, state_arr])
        # dL/dW = -2 * error * x^T  (outer product for each output dim)
        for i in range(_N_STATE):
            grad = -2.0 * error[i] * x
            # L2 regularisation toward population prior
            reg = _PRIOR_STRENGTH * (twin.weights[i] - _POPULATION_WEIGHTS[i])
            twin.weights[i] -= lr * (grad + reg)

        # Bias update
        twin.bias -= lr * (-2.0 * error)
        twin.n_updates += 1

    # Update current state to last observed
    twin.current_state = observed_states[n - 1]
    twin.last_updated = time.time()
    return twin


def simulate_scenario(
    twin: EmotionalTwin,
    scenario: ScenarioDefinition,
    base_context: Optional[ContextVector] = None,
) -> SimulationResult:
    """Simulate a counterfactual scenario.

    Applies context overrides to the base_context for ``duration_steps``
    transitions and records the trajectory.

    Args:
        twin: The user's trained twin.
        scenario: The scenario definition with context overrides.
        base_context: Starting context (defaults to neutral if None).

    Returns:
        SimulationResult with full trajectory.
    """
    if base_context is None:
        base_context = ContextVector()

    ctx_arr = base_context.to_array()
    # Apply overrides
    if scenario.context_overrides:
        for key, val in scenario.context_overrides.items():
            if key in CONTEXT_FIELDS:
                idx = CONTEXT_FIELDS.index(key)
                ctx_arr[idx] = val

    state = twin.current_state.to_array().copy()
    initial = EmotionalState.from_array(state)

    trajectory: List[Dict[str, float]] = []
    trajectory.append(asdict(EmotionalState.from_array(state)))

    steps = max(1, scenario.duration_steps)
    for _ in range(steps):
        state = _transition(state, ctx_arr, twin.weights, twin.bias)
        trajectory.append(asdict(EmotionalState.from_array(state)))

    final = EmotionalState.from_array(state)
    return SimulationResult(
        scenario_name=scenario.name,
        trajectory=trajectory,
        final_state=asdict(final),
        duration_steps=steps,
        initial_state=asdict(initial),
    )


def predict_trajectory(
    twin: EmotionalTwin,
    future_contexts: List[ContextVector],
) -> List[Dict[str, float]]:
    """Predict a sequence of future states given planned contexts.

    Args:
        twin: The user's trained twin.
        future_contexts: Sequence of future context vectors.

    Returns:
        List of predicted state dicts (one per step, starting with current).
    """
    state = twin.current_state.to_array().copy()
    trajectory: List[Dict[str, float]] = [
        asdict(EmotionalState.from_array(state))
    ]
    for ctx in future_contexts:
        ctx_arr = ctx.to_array()
        state = _transition(state, ctx_arr, twin.weights, twin.bias)
        trajectory.append(asdict(EmotionalState.from_array(state)))
    return trajectory


def compute_calibration_score(twin: EmotionalTwin) -> Dict[str, Any]:
    """Compute calibration metrics from the twin's prediction error history.

    Returns mean error, recent error (last 10), n_updates, and a
    calibration_score in [0, 1] where 1 = perfect predictions.
    """
    if not twin.prediction_errors:
        return {
            "calibration_score": 0.0,
            "mean_error": None,
            "recent_error": None,
            "n_updates": twin.n_updates,
            "enough_data": False,
        }

    errors = twin.prediction_errors
    mean_err = float(np.mean(errors))
    recent = errors[-10:] if len(errors) >= 10 else errors
    recent_err = float(np.mean(recent))

    # Map MSE -> 0..1 score.  MSE of 0.1 -> score ~0.5.
    # score = exp(-5 * MSE) gives decent calibration curve.
    score = float(math.exp(-5.0 * recent_err))

    return {
        "calibration_score": round(score, 4),
        "mean_error": round(mean_err, 6),
        "recent_error": round(recent_err, 6),
        "n_updates": twin.n_updates,
        "enough_data": twin.n_updates >= 10,
    }


def twin_to_dict(twin: EmotionalTwin) -> Dict[str, Any]:
    """Serialise an EmotionalTwin to a JSON-safe dictionary."""
    cal = compute_calibration_score(twin)
    return {
        "user_id": twin.user_id,
        "current_state": asdict(twin.current_state),
        "n_updates": twin.n_updates,
        "created_at": twin.created_at,
        "last_updated": twin.last_updated,
        "calibration": cal,
        "weights_shape": list(twin.weights.shape),
        "learning_rate": twin.learning_rate,
    }
