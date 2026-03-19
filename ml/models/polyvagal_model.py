"""Polyvagal state tracking from EEG + HRV fusion.

Maps autonomic nervous system states based on Porges' Polyvagal Theory (1994).

Three hierarchical states:
  - ventral_vagal: Safe/social engagement. High vagal tone, calm but
    connected. Characterized by high HRV, moderate HR, high alpha power,
    low beta/alpha ratio, normal respiration.
  - sympathetic: Fight/flight mobilization. SNS activation with reduced
    vagal brake. Low HRV, elevated HR, suppressed alpha, elevated beta.
  - dorsal_vagal: Freeze/shutdown (immobilization). Primitive vagal
    response. Very low HRV, low HR, high theta, low overall arousal.

The model:
  1. Classifies the current polyvagal state from physiological inputs.
  2. Tracks state transitions over time (trajectory analysis).
  3. Computes autonomic flexibility -- how fluidly the user moves
     between states, a marker of nervous system health.

Inputs (fused EEG + HRV):
  - HRV (RMSSD in ms)
  - Heart rate (bpm)
  - EEG alpha power (8-12 Hz relative power, 0-1)
  - EEG beta/alpha ratio
  - Respiratory rate (breaths per minute)

References:
  Porges SW (1994). The polyvagal theory. In: Psychophysiology of the
    Autonomic Nervous System.
  Porges SW (2011). The Polyvagal Theory. Norton.
  Dana D (2018). The Polyvagal Theory in Therapy. Norton.
  Issue #439: Polyvagal state tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -- Data classes -------------------------------------------------------


@dataclass
class AutonomicSample:
    """Single time-point of autonomic measurements.

    All values should come from real sensors or validated simulations.
    """

    hrv_rmssd: float       # Heart rate variability RMSSD (ms)
    heart_rate: float      # Heart rate (bpm)
    alpha_power: float     # EEG alpha relative power (0-1)
    beta_alpha_ratio: float  # EEG beta / alpha ratio
    resp_rate: float       # Respiratory rate (breaths per minute)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PolyvagalState:
    """Classified polyvagal state at a single time point."""

    state: str               # "ventral_vagal" | "sympathetic" | "dorsal_vagal"
    confidence: float        # 0-1
    probabilities: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PolyvagalProfile:
    """Full polyvagal profile computed from a history of samples."""

    dominant_state: str
    state_distribution: Dict[str, float]
    transition_matrix: Dict[str, Dict[str, float]]
    mean_dwell_times: Dict[str, float]
    autonomic_flexibility: float
    n_samples: int
    n_transitions: int
    trajectory: List[PolyvagalState]


# -- Constants ----------------------------------------------------------

STATES = ("ventral_vagal", "sympathetic", "dorsal_vagal")

# Thresholds derived from literature on HRV norms, EEG band power
# distributions, and polyvagal mapping heuristics.
#
# HRV RMSSD norms (ms):
#   High vagal tone:  > 40
#   Low vagal tone:   20-40
#   Very low:         < 20
#
# Heart rate (bpm):
#   Moderate:  60-80
#   Elevated:  > 90
#   Low:       < 60
#
# Alpha power (relative, 0-1):
#   High:  > 0.3
#   Low:   < 0.15
#
# Beta/alpha ratio:
#   Low (relaxed):   < 1.0
#   High (aroused):  > 1.5

_HRV_HIGH = 40.0
_HRV_LOW = 20.0
_HR_HIGH = 90.0
_HR_LOW = 60.0
_ALPHA_HIGH = 0.30
_ALPHA_LOW = 0.15
_BA_RATIO_HIGH = 1.5
_BA_RATIO_LOW = 1.0
_RESP_HIGH = 20.0
_RESP_LOW = 10.0


# -- Classification logic -----------------------------------------------


def _compute_state_scores(sample: AutonomicSample) -> Dict[str, float]:
    """Compute raw scores for each polyvagal state.

    Each physiological indicator contributes a partial score toward
    each state. Scores are NOT normalized to sum to 1 here --
    that happens in classify_polyvagal_state.
    """
    vv = 0.0  # ventral vagal
    sy = 0.0  # sympathetic
    dv = 0.0  # dorsal vagal

    hrv = sample.hrv_rmssd
    hr = sample.heart_rate
    alpha = sample.alpha_power
    ba = sample.beta_alpha_ratio
    rr = sample.resp_rate

    # -- HRV contribution (most important marker) --
    if hrv >= _HRV_HIGH:
        vv += 2.0
    elif hrv >= _HRV_LOW:
        vv += 1.0
        sy += 0.5
    else:
        # Very low HRV
        dv += 1.5
        sy += 1.0

    # -- Heart rate contribution --
    if hr > _HR_HIGH:
        sy += 2.0
    elif hr >= _HR_LOW:
        vv += 1.5
    else:
        # Bradycardia range
        dv += 2.0

    # -- Alpha power contribution --
    if alpha >= _ALPHA_HIGH:
        vv += 1.5
    elif alpha >= _ALPHA_LOW:
        vv += 0.5
        sy += 0.5
    else:
        # Suppressed alpha
        sy += 0.5
        dv += 1.0

    # -- Beta/alpha ratio contribution --
    if ba > _BA_RATIO_HIGH:
        sy += 1.5
    elif ba > _BA_RATIO_LOW:
        sy += 0.5
        vv += 0.5
    else:
        vv += 1.0

    # -- Respiratory rate contribution --
    if rr > _RESP_HIGH:
        sy += 1.0
    elif rr >= _RESP_LOW:
        vv += 1.0
    else:
        dv += 1.0

    return {
        "ventral_vagal": vv,
        "sympathetic": sy,
        "dorsal_vagal": dv,
    }


def classify_polyvagal_state(sample: AutonomicSample) -> PolyvagalState:
    """Classify the current polyvagal state from a single autonomic sample.

    Returns a PolyvagalState with the winning state, confidence, and
    probability distribution across all three states.
    """
    scores = _compute_state_scores(sample)

    total = sum(scores.values())
    if total < 1e-10:
        probs = {s: 1.0 / 3.0 for s in STATES}
    else:
        probs = {s: scores[s] / total for s in STATES}

    winning_state = max(probs, key=probs.get)  # type: ignore[arg-type]
    confidence = probs[winning_state]

    # Round probabilities for clean output
    probs = {s: round(p, 4) for s, p in probs.items()}

    return PolyvagalState(
        state=winning_state,
        confidence=round(confidence, 4),
        probabilities=probs,
        timestamp=sample.timestamp,
    )


# -- Trajectory & transitions -------------------------------------------


def compute_state_trajectory(
    samples: List[AutonomicSample],
) -> List[PolyvagalState]:
    """Classify each sample in a time series and return the trajectory.

    Args:
        samples: Chronologically ordered autonomic samples.

    Returns:
        List of PolyvagalState, one per sample.
    """
    return [classify_polyvagal_state(s) for s in samples]


def _compute_transition_matrix(
    trajectory: List[PolyvagalState],
) -> Tuple[Dict[str, Dict[str, float]], int]:
    """Compute state transition probability matrix.

    Returns:
        (transition_matrix, n_transitions)
        transition_matrix[from_state][to_state] = probability
    """
    counts: Dict[str, Dict[str, int]] = {
        s: {t: 0 for t in STATES} for s in STATES
    }
    n_transitions = 0

    for i in range(len(trajectory) - 1):
        from_s = trajectory[i].state
        to_s = trajectory[i + 1].state
        counts[from_s][to_s] += 1
        if from_s != to_s:
            n_transitions += 1

    matrix: Dict[str, Dict[str, float]] = {}
    for s in STATES:
        row_total = sum(counts[s].values())
        if row_total == 0:
            matrix[s] = {t: 0.0 for t in STATES}
        else:
            matrix[s] = {
                t: round(counts[s][t] / row_total, 4) for t in STATES
            }

    return matrix, n_transitions


def _compute_dwell_times(
    trajectory: List[PolyvagalState],
) -> Dict[str, float]:
    """Compute mean dwell time per state (in number of samples).

    Dwell time = average consecutive run length for each state.
    """
    if not trajectory:
        return {s: 0.0 for s in STATES}

    runs: Dict[str, List[int]] = {s: [] for s in STATES}
    current_state = trajectory[0].state
    current_length = 1

    for i in range(1, len(trajectory)):
        if trajectory[i].state == current_state:
            current_length += 1
        else:
            runs[current_state].append(current_length)
            current_state = trajectory[i].state
            current_length = 1
    # Final run
    runs[current_state].append(current_length)

    dwell: Dict[str, float] = {}
    for s in STATES:
        if runs[s]:
            dwell[s] = round(float(np.mean(runs[s])), 2)
        else:
            dwell[s] = 0.0

    return dwell


# -- Autonomic flexibility ----------------------------------------------


def compute_autonomic_flexibility(
    trajectory: List[PolyvagalState],
) -> float:
    """Compute autonomic flexibility score (0-1).

    Autonomic flexibility measures how fluidly the nervous system
    transitions between states. Higher flexibility is generally
    healthier -- it means the system can mobilize (sympathetic)
    when needed and return to safety (ventral vagal) efficiently.

    Components:
      1. State diversity: Are all three states represented? (entropy)
      2. Transition rate: How often do transitions occur?
      3. Recovery tendency: Does the system tend to return to
         ventral vagal after sympathetic/dorsal activation?

    Returns:
        Float in [0, 1]. Higher = more flexible.
    """
    if len(trajectory) < 3:
        return 0.0

    # 1. State diversity (normalized entropy)
    state_counts = {s: 0 for s in STATES}
    for ps in trajectory:
        state_counts[ps.state] += 1
    total = len(trajectory)
    probs = [state_counts[s] / total for s in STATES if state_counts[s] > 0]
    if len(probs) <= 1:
        entropy = 0.0
    else:
        entropy = -sum(p * np.log2(p) for p in probs)
    max_entropy = np.log2(len(STATES))
    diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0

    # 2. Transition rate (fraction of adjacent pairs that differ)
    n_diff = sum(
        1
        for i in range(len(trajectory) - 1)
        if trajectory[i].state != trajectory[i + 1].state
    )
    transition_rate = n_diff / (len(trajectory) - 1)

    # 3. Recovery tendency: proportion of sympathetic/dorsal episodes
    #    that are followed (within 3 steps) by ventral_vagal
    activated_count = 0
    recovered_count = 0
    for i, ps in enumerate(trajectory):
        if ps.state in ("sympathetic", "dorsal_vagal"):
            activated_count += 1
            # Look ahead up to 3 steps for recovery to ventral
            for j in range(i + 1, min(i + 4, len(trajectory))):
                if trajectory[j].state == "ventral_vagal":
                    recovered_count += 1
                    break

    if activated_count > 0:
        recovery_score = recovered_count / activated_count
    else:
        # No activation episodes -- if all ventral_vagal, moderate flex
        recovery_score = 0.5

    # Weighted composite
    flexibility = (
        0.35 * diversity_score
        + 0.30 * transition_rate
        + 0.35 * recovery_score
    )

    return round(float(np.clip(flexibility, 0.0, 1.0)), 4)


# -- Full profile -------------------------------------------------------


def compute_polyvagal_profile(
    samples: List[AutonomicSample],
) -> PolyvagalProfile:
    """Compute a full polyvagal profile from a history of samples.

    Combines classification, trajectory analysis, transition probabilities,
    dwell times, and autonomic flexibility into a single profile.

    Args:
        samples: At least 3 chronologically ordered autonomic samples.

    Returns:
        PolyvagalProfile dataclass.

    Raises:
        ValueError: If fewer than 3 samples are provided.
    """
    if len(samples) < 3:
        raise ValueError(
            f"Need at least 3 samples for a profile, got {len(samples)}"
        )

    trajectory = compute_state_trajectory(samples)

    # State distribution
    state_counts = {s: 0 for s in STATES}
    for ps in trajectory:
        state_counts[ps.state] += 1
    total = len(trajectory)
    distribution = {s: round(state_counts[s] / total, 4) for s in STATES}

    dominant = max(distribution, key=distribution.get)  # type: ignore[arg-type]

    trans_matrix, n_trans = _compute_transition_matrix(trajectory)
    dwell = _compute_dwell_times(trajectory)
    flexibility = compute_autonomic_flexibility(trajectory)

    return PolyvagalProfile(
        dominant_state=dominant,
        state_distribution=distribution,
        transition_matrix=trans_matrix,
        mean_dwell_times=dwell,
        autonomic_flexibility=flexibility,
        n_samples=total,
        n_transitions=n_trans,
        trajectory=trajectory,
    )


def profile_to_dict(profile: PolyvagalProfile) -> Dict[str, Any]:
    """Serialize a PolyvagalProfile to a JSON-safe dict."""
    return {
        "dominant_state": profile.dominant_state,
        "state_distribution": profile.state_distribution,
        "transition_matrix": profile.transition_matrix,
        "mean_dwell_times": profile.mean_dwell_times,
        "autonomic_flexibility": profile.autonomic_flexibility,
        "n_samples": profile.n_samples,
        "n_transitions": profile.n_transitions,
        "trajectory": [
            {
                "state": ps.state,
                "confidence": ps.confidence,
                "probabilities": ps.probabilities,
                "timestamp": ps.timestamp,
            }
            for ps in profile.trajectory
        ],
    }
