"""Real-time flow state neurofeedback with distraction defense (issue #441).

Provides deep-work productivity mode driven by EEG signals:
- Flow state detection from alpha/theta coherence, moderate beta, low delta
- Distraction detection from sudden beta spikes, alpha dropout, theta surges
- Flow protection system: suppress non-urgent notifications during flow
- Session tracking with entry time, duration, depth, exit reason
- Optimal conditions learning: time of day, sound, caffeine state per user

Scientific basis:
- Csikszentmihalyi (1990) — Flow: the psychology of optimal experience
- Dietrich (2004) — Transient hypofrontality theory
- Katahira et al. (2018) — EEG correlates of flow state
- Ulrich et al. (2014) — Neural correlates of experimentally induced flow
- Warm et al. (2008) — Vigilance decrement over time-on-task

Flow score (0-100) sub-dimensions:
  focus_depth       — sustained beta/theta ratio + alpha suppression
  creative_engagement — alpha/theta coherence (divergent thinking readiness)
  time_distortion   — theta dominance pattern (absorption marker)
  effortlessness    — low high-beta (no anxiety) + moderate total beta

Distraction types:
  beta_spike      — sudden beta increase (external interruption or internal worry)
  alpha_dropout   — rapid alpha loss (startle, context switch)
  theta_surge     — theta spike (drowsiness onset)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Flow score thresholds
_FLOW_DEEP = 75
_FLOW_MODERATE = 50
_FLOW_LIGHT = 30

# Distraction detection thresholds (z-score relative to recent baseline)
_BETA_SPIKE_Z = 2.0
_ALPHA_DROP_Z = -2.0
_THETA_SURGE_Z = 2.0

# Flow protection: minimum score to activate shielding
_PROTECTION_THRESHOLD = 55

# Rolling window size for baseline stats (number of readings)
_BASELINE_WINDOW = 20

# Optimal conditions: minimum sessions for reliable recommendations
_MIN_SESSIONS_FOR_OPTIMAL = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FlowReading:
    """Single flow state measurement."""

    flow_score: float             # 0-100 composite
    focus_depth: float            # 0-100
    creative_engagement: float    # 0-100
    time_distortion: float        # 0-100
    effortlessness: float         # 0-100
    flow_level: str               # "none" | "light" | "moderate" | "deep"
    timestamp: float              # time.time()


@dataclass
class DistractionEvent:
    """Detected distraction."""

    distraction_type: str         # "beta_spike" | "alpha_dropout" | "theta_surge"
    severity: float               # 0-1
    description: str
    recovery_suggestion: str
    timestamp: float


@dataclass
class FlowSession:
    """One continuous flow session."""

    session_id: str
    user_id: str
    entry_time: float
    exit_time: Optional[float] = None
    peak_score: float = 0.0
    mean_score: float = 0.0
    duration_minutes: float = 0.0
    depth_category: str = "none"
    exit_reason: Optional[str] = None
    distraction_count: int = 0
    readings: List[float] = field(default_factory=list)
    conditions: Optional[Dict[str, Any]] = None


@dataclass
class OptimalConditions:
    """Learned optimal flow conditions for a user."""

    best_time_of_day: Optional[str] = None          # "morning" | "afternoon" | "evening" | "night"
    best_sound_environment: Optional[str] = None     # "silence" | "music" | "nature" | "white_noise"
    caffeine_effect: Optional[str] = None            # "positive" | "negative" | "neutral"
    avg_entry_latency_minutes: float = 0.0
    avg_flow_duration_minutes: float = 0.0
    best_day_of_week: Optional[str] = None
    session_count: int = 0
    confidence: float = 0.0


@dataclass
class FlowProfile:
    """Complete flow profile for a user."""

    user_id: str
    total_sessions: int
    total_flow_minutes: float
    avg_flow_score: float
    avg_session_duration_minutes: float
    peak_score_ever: float
    optimal_conditions: OptimalConditions
    recent_trend: str             # "improving" | "stable" | "declining"
    distraction_rate: float       # distractions per hour of flow


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------

def compute_flow_score(eeg_features: Dict[str, float]) -> Dict[str, Any]:
    """Compute real-time flow score from EEG band power features.

    Parameters
    ----------
    eeg_features : dict
        Must contain keys: alpha, beta, theta, delta.
        Optional: high_beta, low_beta, gamma.

    Returns
    -------
    dict with flow_score (0-100), sub-dimensions, flow_level, band_contributions.
    """
    alpha = max(eeg_features.get("alpha", 0.0), 0.0)
    beta = max(eeg_features.get("beta", 0.0), 0.0)
    theta = max(eeg_features.get("theta", 0.0), 0.0)
    delta = max(eeg_features.get("delta", 0.0), 0.0)
    high_beta = max(eeg_features.get("high_beta", beta * 0.4), 0.0)
    low_beta = max(eeg_features.get("low_beta", beta * 0.6), 0.0)

    eps = 1e-10

    # -- Sub-dimension 1: Focus Depth (30% weight) --
    # High beta/theta ratio + alpha suppression = deep focus
    bt_ratio = beta / (theta + eps)
    bt_score = float(np.clip(np.tanh(bt_ratio * 0.5) * 100, 0, 100))

    total_power = alpha + beta + theta + delta + eps
    alpha_fraction = alpha / total_power
    alpha_suppression = float(np.clip((1.0 - alpha_fraction * 2.5) * 100, 0, 100))

    focus_depth = 0.6 * bt_score + 0.4 * alpha_suppression

    # -- Sub-dimension 2: Creative Engagement (25% weight) --
    # Alpha/theta coherence — both elevated together = creative flow
    at_sum = alpha + theta
    at_ratio = at_sum / (beta + eps)
    # Moderate alpha+theta relative to beta = creative engagement
    # Sigmoid centered at ratio 2.0
    creative_raw = float(1.0 / (1.0 + np.exp(-1.5 * (at_ratio - 2.0))))
    creative_engagement = float(np.clip(creative_raw * 100, 0, 100))

    # -- Sub-dimension 3: Time Distortion (20% weight) --
    # Theta dominance over delta (absorption, not drowsiness)
    theta_fraction = theta / total_power
    delta_fraction = delta / total_power
    # High theta but NOT high delta = absorption
    theta_engagement = float(np.clip(theta_fraction * 3.0, 0, 1))
    delta_penalty = float(np.clip(delta_fraction * 2.0, 0, 1))
    time_distortion = float(np.clip((theta_engagement - 0.5 * delta_penalty) * 100, 0, 100))

    # -- Sub-dimension 4: Effortlessness (25% weight) --
    # Low high-beta (no anxiety) + moderate total beta (engaged but not straining)
    high_beta_fraction = high_beta / (beta + eps)
    anxiety_penalty = float(np.clip(high_beta_fraction * 2.0, 0, 1))
    # Beta should be moderate — neither too low (disengaged) nor too high (straining)
    beta_fraction = beta / total_power
    optimal_beta = 0.25
    beta_deviation = abs(beta_fraction - optimal_beta)
    beta_optimality = float(np.clip(1.0 - beta_deviation * 4.0, 0, 1))
    effortlessness = float(np.clip((beta_optimality - 0.5 * anxiety_penalty) * 100, 0, 100))

    # -- Composite flow score --
    flow_score = (
        0.30 * focus_depth
        + 0.25 * creative_engagement
        + 0.20 * time_distortion
        + 0.25 * effortlessness
    )
    flow_score = float(np.clip(flow_score, 0, 100))

    # -- Flow level --
    if flow_score >= _FLOW_DEEP:
        flow_level = "deep"
    elif flow_score >= _FLOW_MODERATE:
        flow_level = "moderate"
    elif flow_score >= _FLOW_LIGHT:
        flow_level = "light"
    else:
        flow_level = "none"

    return {
        "flow_score": round(flow_score, 1),
        "focus_depth": round(focus_depth, 1),
        "creative_engagement": round(creative_engagement, 1),
        "time_distortion": round(time_distortion, 1),
        "effortlessness": round(effortlessness, 1),
        "flow_level": flow_level,
        "band_contributions": {
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "theta": round(theta, 4),
            "delta": round(delta, 4),
            "high_beta": round(high_beta, 4),
            "low_beta": round(low_beta, 4),
        },
    }


def detect_distraction(
    eeg_features: Dict[str, float],
    baseline_stats: Dict[str, Tuple[float, float]],
) -> Optional[Dict[str, Any]]:
    """Detect distraction events from EEG features relative to baseline.

    Parameters
    ----------
    eeg_features : dict
        Current EEG band powers (alpha, beta, theta, delta).
    baseline_stats : dict
        Rolling baseline per band: {"alpha": (mean, std), "beta": (mean, std), ...}.

    Returns
    -------
    dict with distraction_type, severity, description, recovery_suggestion
    or None if no distraction detected.
    """
    if not baseline_stats:
        return None

    eps = 1e-10
    detections: List[Tuple[str, float, str, str]] = []

    # -- Beta spike: external interruption or sudden worry --
    beta = eeg_features.get("beta", 0.0)
    beta_mean, beta_std = baseline_stats.get("beta", (0.0, 1.0))
    if beta_std > eps:
        beta_z = (beta - beta_mean) / (beta_std + eps)
        if beta_z > _BETA_SPIKE_Z:
            severity = float(np.clip((beta_z - _BETA_SPIKE_Z) / 3.0, 0, 1))
            detections.append((
                "beta_spike",
                severity,
                "Sudden beta increase detected — possible external interruption or worry onset",
                "Take 3 slow breaths. If an external trigger caused this, note it and return to task.",
            ))

    # -- Alpha dropout: startle or context switch --
    alpha = eeg_features.get("alpha", 0.0)
    alpha_mean, alpha_std = baseline_stats.get("alpha", (0.0, 1.0))
    if alpha_std > eps:
        alpha_z = (alpha - alpha_mean) / (alpha_std + eps)
        if alpha_z < _ALPHA_DROP_Z:
            severity = float(np.clip((-alpha_z - abs(_ALPHA_DROP_Z)) / 3.0, 0, 1))
            detections.append((
                "alpha_dropout",
                severity,
                "Rapid alpha loss — possible context switch or startle response",
                "Close your eyes for 10 seconds. Let alpha rhythm rebuild before resuming.",
            ))

    # -- Theta surge: drowsiness onset --
    theta = eeg_features.get("theta", 0.0)
    theta_mean, theta_std = baseline_stats.get("theta", (0.0, 1.0))
    if theta_std > eps:
        theta_z = (theta - theta_mean) / (theta_std + eps)
        if theta_z > _THETA_SURGE_Z:
            severity = float(np.clip((theta_z - _THETA_SURGE_Z) / 3.0, 0, 1))
            detections.append((
                "theta_surge",
                severity,
                "Theta surge detected — drowsiness onset likely",
                "Stand up and stretch. Consider a 5-minute walk or cold water on your face.",
            ))

    if not detections:
        return None

    # Return the most severe distraction
    detections.sort(key=lambda d: d[1], reverse=True)
    dtype, sev, desc, recovery = detections[0]

    return {
        "distraction_type": dtype,
        "severity": round(sev, 3),
        "description": desc,
        "recovery_suggestion": recovery,
        "timestamp": time.time(),
    }


def should_protect_flow(
    flow_score: float,
    flow_duration_minutes: float,
    notification_priority: str = "normal",
) -> Dict[str, Any]:
    """Decide whether to suppress a notification to protect flow state.

    Parameters
    ----------
    flow_score : float
        Current flow score (0-100).
    flow_duration_minutes : float
        How long user has been in flow.
    notification_priority : str
        "urgent", "high", "normal", or "low".

    Returns
    -------
    dict with should_suppress (bool), reason, flow_shield_active (bool).
    """
    in_flow = flow_score >= _PROTECTION_THRESHOLD

    priority_bypass = {
        "urgent": True,    # always deliver
        "high": False,     # suppress during deep flow only
        "normal": False,   # suppress during any flow
        "low": False,      # always suppress during flow
    }

    always_deliver = priority_bypass.get(notification_priority, False)

    if always_deliver:
        return {
            "should_suppress": False,
            "reason": f"Priority '{notification_priority}' bypasses flow protection",
            "flow_shield_active": in_flow,
            "flow_score": round(flow_score, 1),
        }

    if not in_flow:
        return {
            "should_suppress": False,
            "reason": "Flow score below protection threshold",
            "flow_shield_active": False,
            "flow_score": round(flow_score, 1),
        }

    # In flow — decide based on depth and duration
    deep_flow = flow_score >= _FLOW_DEEP
    long_session = flow_duration_minutes >= 15

    if notification_priority == "high" and not deep_flow:
        return {
            "should_suppress": False,
            "reason": "High-priority notification delivered during non-deep flow",
            "flow_shield_active": True,
            "flow_score": round(flow_score, 1),
        }

    reason = (
        f"Suppressing '{notification_priority}' notification — "
        f"flow score {flow_score:.0f}, "
        f"{'deep' if deep_flow else 'moderate'} flow for {flow_duration_minutes:.0f} min"
    )

    return {
        "should_suppress": True,
        "reason": reason,
        "flow_shield_active": True,
        "flow_score": round(flow_score, 1),
    }


def track_flow_session(
    session_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Process and summarize a completed flow session.

    Parameters
    ----------
    session_data : dict
        Keys: user_id, session_id, entry_time, exit_time, readings (list of floats),
        exit_reason (optional), distraction_count (optional),
        conditions (optional dict with time_of_day, sound_env, caffeine).

    Returns
    -------
    dict with session summary including duration, depth, productivity correlation.
    """
    user_id = session_data.get("user_id", "unknown")
    session_id = session_data.get("session_id", "unknown")
    entry_time = session_data.get("entry_time", 0.0)
    exit_time = session_data.get("exit_time", entry_time)
    readings = session_data.get("readings", [])
    exit_reason = session_data.get("exit_reason", "natural")
    distraction_count = session_data.get("distraction_count", 0)
    conditions = session_data.get("conditions")

    duration_minutes = max(0.0, (exit_time - entry_time) / 60.0)

    if readings:
        scores = np.array(readings, dtype=float)
        peak_score = float(np.max(scores))
        mean_score = float(np.mean(scores))
    else:
        peak_score = 0.0
        mean_score = 0.0

    # Depth category based on mean score
    if mean_score >= _FLOW_DEEP:
        depth = "deep"
    elif mean_score >= _FLOW_MODERATE:
        depth = "moderate"
    elif mean_score >= _FLOW_LIGHT:
        depth = "light"
    else:
        depth = "none"

    # Productivity correlation: higher flow + longer duration + fewer distractions = better
    if duration_minutes > 0:
        distraction_rate = distraction_count / (duration_minutes / 60.0) if duration_minutes > 0 else 0.0
        productivity_index = float(np.clip(
            (mean_score / 100.0) * (1.0 - min(1.0, distraction_rate / 10.0)),
            0.0, 1.0,
        ))
    else:
        distraction_rate = 0.0
        productivity_index = 0.0

    return {
        "user_id": user_id,
        "session_id": session_id,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "duration_minutes": round(duration_minutes, 1),
        "peak_score": round(peak_score, 1),
        "mean_score": round(mean_score, 1),
        "depth_category": depth,
        "exit_reason": exit_reason,
        "distraction_count": distraction_count,
        "distraction_rate_per_hour": round(distraction_rate, 2),
        "productivity_index": round(productivity_index, 3),
        "conditions": conditions,
    }


def compute_optimal_conditions(
    session_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Learn optimal flow conditions from historical sessions.

    Parameters
    ----------
    session_history : list of session dicts
        Each dict should have: mean_score, duration_minutes, conditions
        (with time_of_day, sound_env, caffeine).

    Returns
    -------
    dict with best_time_of_day, best_sound_environment, caffeine_effect,
    avg_entry_latency, avg_duration, confidence.
    """
    if len(session_history) < _MIN_SESSIONS_FOR_OPTIMAL:
        return {
            "best_time_of_day": None,
            "best_sound_environment": None,
            "caffeine_effect": None,
            "avg_entry_latency_minutes": 0.0,
            "avg_flow_duration_minutes": 0.0,
            "best_day_of_week": None,
            "session_count": len(session_history),
            "confidence": 0.0,
        }

    # Group sessions by condition categories
    time_scores: Dict[str, List[float]] = {}
    sound_scores: Dict[str, List[float]] = {}
    caffeine_scores: Dict[str, List[float]] = {}
    day_scores: Dict[str, List[float]] = {}
    durations: List[float] = []

    for session in session_history:
        score = session.get("mean_score", 0.0)
        dur = session.get("duration_minutes", 0.0)
        durations.append(dur)
        conds = session.get("conditions") or {}

        tod = conds.get("time_of_day")
        if tod:
            time_scores.setdefault(tod, []).append(score)

        sound = conds.get("sound_env")
        if sound:
            sound_scores.setdefault(sound, []).append(score)

        caffeine = conds.get("caffeine")
        if caffeine is not None:
            key = "with_caffeine" if caffeine else "without_caffeine"
            caffeine_scores.setdefault(key, []).append(score)

        day = conds.get("day_of_week")
        if day:
            day_scores.setdefault(day, []).append(score)

    def _best_category(grouped: Dict[str, List[float]]) -> Optional[str]:
        if not grouped:
            return None
        averages = {k: float(np.mean(v)) for k, v in grouped.items() if len(v) >= 2}
        if not averages:
            return None
        return max(averages, key=averages.get)  # type: ignore[arg-type]

    best_time = _best_category(time_scores)
    best_sound = _best_category(sound_scores)
    best_day = _best_category(day_scores)

    # Caffeine effect
    caffeine_effect = None
    if "with_caffeine" in caffeine_scores and "without_caffeine" in caffeine_scores:
        with_avg = float(np.mean(caffeine_scores["with_caffeine"]))
        without_avg = float(np.mean(caffeine_scores["without_caffeine"]))
        diff = with_avg - without_avg
        if diff > 5:
            caffeine_effect = "positive"
        elif diff < -5:
            caffeine_effect = "negative"
        else:
            caffeine_effect = "neutral"

    avg_duration = float(np.mean(durations)) if durations else 0.0
    confidence = min(1.0, len(session_history) / 20.0)

    return {
        "best_time_of_day": best_time,
        "best_sound_environment": best_sound,
        "caffeine_effect": caffeine_effect,
        "avg_entry_latency_minutes": 0.0,  # would need entry_latency data
        "avg_flow_duration_minutes": round(avg_duration, 1),
        "best_day_of_week": best_day,
        "session_count": len(session_history),
        "confidence": round(confidence, 2),
    }


def compute_flow_profile(
    user_id: str,
    session_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build complete flow profile for a user.

    Parameters
    ----------
    user_id : str
    session_history : list of session summary dicts

    Returns
    -------
    dict with profile fields including optimal conditions and trend.
    """
    if not session_history:
        optimal = compute_optimal_conditions([])
        return {
            "user_id": user_id,
            "total_sessions": 0,
            "total_flow_minutes": 0.0,
            "avg_flow_score": 0.0,
            "avg_session_duration_minutes": 0.0,
            "peak_score_ever": 0.0,
            "optimal_conditions": optimal,
            "recent_trend": "stable",
            "distraction_rate": 0.0,
        }

    total_sessions = len(session_history)
    durations = [s.get("duration_minutes", 0.0) for s in session_history]
    scores = [s.get("mean_score", 0.0) for s in session_history]
    peaks = [s.get("peak_score", 0.0) for s in session_history]
    distractions = [s.get("distraction_count", 0) for s in session_history]
    total_flow_minutes = sum(durations)
    total_hours = total_flow_minutes / 60.0 if total_flow_minutes > 0 else 1.0
    total_distractions = sum(distractions)

    # Trend: compare last 3 sessions to previous 3
    if len(scores) >= 6:
        recent_avg = float(np.mean(scores[-3:]))
        earlier_avg = float(np.mean(scores[-6:-3]))
        diff = recent_avg - earlier_avg
        if diff > 5:
            trend = "improving"
        elif diff < -5:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "stable"

    optimal = compute_optimal_conditions(session_history)

    return {
        "user_id": user_id,
        "total_sessions": total_sessions,
        "total_flow_minutes": round(total_flow_minutes, 1),
        "avg_flow_score": round(float(np.mean(scores)), 1),
        "avg_session_duration_minutes": round(float(np.mean(durations)), 1),
        "peak_score_ever": round(float(max(peaks)), 1),
        "optimal_conditions": optimal,
        "recent_trend": trend,
        "distraction_rate": round(total_distractions / total_hours, 2),
    }


def profile_to_dict(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure profile dict is JSON-serializable. Passthrough for dict profiles."""
    # Already a dict from compute_flow_profile; ensure nested dicts are clean
    result = dict(profile)
    if "optimal_conditions" in result and isinstance(result["optimal_conditions"], dict):
        result["optimal_conditions"] = dict(result["optimal_conditions"])
    return result
