"""Emotion-aware adaptive education system.

EEG-driven learning that responds to student emotional state in real time.
Detects learning states (engaged, confused, bored, frustrated, flow),
adapts difficulty and pacing, tracks attention span, and identifies
optimal learning windows.

Learning state detection from EEG:
  - Engaged: moderate beta, low theta
  - Confused: high theta, frontal activation (high beta)
  - Bored: high alpha, low beta
  - Frustrated: high beta, negative valence (high theta + high beta + fatigue)
  - Flow: alpha-theta border (balanced alpha/theta, moderate beta)

References:
    Pekrun (2006) — Control-Value Theory of Achievement Emotions
    Csikszentmihalyi (1990) — Flow and optimal learning
    Springer 2024 — 7 educational emotions from 5 EEG channels
    BrainAccess 2025 — Real-time EEG feedback for educators

GitHub issue: #450
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LearningEEGFeatures:
    """EEG band power features relevant to learning states."""

    theta: float = 0.0      # 4-8 Hz: memory encoding, confusion marker
    alpha: float = 0.0      # 8-12 Hz: relaxation, boredom marker
    beta: float = 0.0       # 12-30 Hz: active thinking, engagement/frustration
    valence: float = 0.0    # -1 to 1: emotional valence from FAA
    fatigue: float = 0.0    # 0-1: fatigue index
    timestamp: float = 0.0  # Unix timestamp or session-relative seconds


@dataclass
class LearningState:
    """Detected learning state with scores and recommendations."""

    state: str = "unknown"           # engaged, confused, bored, frustrated, flow
    confidence: float = 0.0          # 0-1
    scores: Dict[str, float] = field(default_factory=dict)
    difficulty_recommendation: str = "maintain"   # increase, decrease, maintain
    pacing_recommendation: str = "maintain"       # speed_up, slow_down, maintain, take_break
    engagement_level: float = 0.0    # 0-1
    attention_declining: bool = False


@dataclass
class EducationProfile:
    """Comprehensive education profile for a student session."""

    learning_state: LearningState = field(default_factory=LearningState)
    attention_span_minutes: float = 0.0
    optimal_window_hours: List[float] = field(default_factory=list)
    effectiveness_by_state: Dict[str, float] = field(default_factory=dict)
    session_engagement_mean: float = 0.0
    session_duration_minutes: float = 0.0
    difficulty_level: float = 0.5
    state_distribution: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    recommendations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEARNING_STATES = ["engaged", "confused", "bored", "frustrated", "flow"]

DIFFICULTY_ACTIONS = {
    "engaged": "maintain",
    "confused": "decrease",
    "bored": "increase",
    "frustrated": "decrease",
    "flow": "increase",
}

PACING_ACTIONS = {
    "engaged": "maintain",
    "confused": "slow_down",
    "bored": "speed_up",
    "frustrated": "take_break",
    "flow": "maintain",
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def detect_learning_state(features: LearningEEGFeatures) -> LearningState:
    """Detect the current learning state from EEG features.

    Classification logic:
      - Engaged: moderate beta (0.3-0.6), low theta (<0.4)
      - Confused: high theta (>0.5), high beta from frontal activation
      - Bored: high alpha (>0.5), low beta (<0.3)
      - Frustrated: high beta (>0.5), negative valence, high fatigue
      - Flow: alpha-theta border, balanced features, low fatigue

    Args:
        features: EEG band power features.

    Returns:
        LearningState with detected state and recommendations.
    """
    theta = float(np.clip(features.theta, 0, 1))
    alpha = float(np.clip(features.alpha, 0, 1))
    beta = float(np.clip(features.beta, 0, 1))
    valence = float(np.clip(features.valence, -1, 1))
    fatigue = float(np.clip(features.fatigue, 0, 1))

    scores = _compute_state_scores(theta, alpha, beta, valence, fatigue)
    state = max(scores, key=scores.get)
    confidence = float(scores[state])

    # Engagement level: beta-driven, penalized by alpha dominance
    engagement = float(np.clip(
        0.45 * beta + 0.30 * (1.0 - alpha) + 0.25 * (1.0 - theta / max(beta, 1e-10) / 3.0),
        0, 1,
    ))

    return LearningState(
        state=state,
        confidence=round(confidence, 4),
        scores={k: round(v, 4) for k, v in scores.items()},
        difficulty_recommendation=DIFFICULTY_ACTIONS[state],
        pacing_recommendation=PACING_ACTIONS[state],
        engagement_level=round(engagement, 4),
        attention_declining=False,  # set by track_attention_span
    )


def recommend_difficulty_adjustment(
    state: LearningState,
    current_difficulty: float,
) -> Dict:
    """Recommend difficulty adjustment based on current learning state.

    Args:
        state: Current detected learning state.
        current_difficulty: Current difficulty level 0-1.

    Returns:
        Dict with adjustment, new_difficulty, and reason.
    """
    current_difficulty = float(np.clip(current_difficulty, 0, 1))

    adjustment_rates = {
        "engaged": 0.0,
        "confused": -0.05,
        "bored": 0.05,
        "frustrated": -0.08,
        "flow": 0.03,
    }

    rate = adjustment_rates.get(state.state, 0.0) * state.confidence
    new_difficulty = float(np.clip(current_difficulty + rate, 0, 1))

    return {
        "action": state.difficulty_recommendation,
        "adjustment": round(rate, 4),
        "previous_difficulty": round(current_difficulty, 4),
        "new_difficulty": round(new_difficulty, 4),
        "reason": _difficulty_reason(state.state),
    }


def recommend_pacing(
    state: LearningState,
    session_minutes: float = 0.0,
    consecutive_same_state: int = 0,
) -> Dict:
    """Recommend content pacing based on learning state.

    Args:
        state: Current detected learning state.
        session_minutes: Minutes elapsed in current session.
        consecutive_same_state: How many consecutive readings in this state.

    Returns:
        Dict with pacing, speed_factor, and suggestions.
    """
    base_pacing = PACING_ACTIONS.get(state.state, "maintain")
    speed_factors = {
        "speed_up": 1.25,
        "slow_down": 0.75,
        "maintain": 1.0,
        "take_break": 0.0,
    }

    # Override: prolonged frustration -> force break
    if state.state == "frustrated" and consecutive_same_state >= 5:
        base_pacing = "take_break"

    # Override: prolonged confusion -> slow down more
    if state.state == "confused" and consecutive_same_state >= 8:
        base_pacing = "take_break"

    # Time-based break suggestion (Pomodoro-style)
    if session_minutes > 25 and base_pacing != "take_break":
        suggestions = [
            "Consider a short break - 25 minutes elapsed",
            _pacing_suggestion(state.state),
        ]
    else:
        suggestions = [_pacing_suggestion(state.state)]

    return {
        "pacing": base_pacing,
        "speed_factor": speed_factors[base_pacing],
        "suggestions": suggestions,
        "state": state.state,
        "confidence": state.confidence,
        "session_minutes": round(session_minutes, 1),
    }


def track_attention_span(
    engagement_history: List[float],
    window_size: int = 10,
    decline_threshold: float = 0.15,
) -> Dict:
    """Track attention span from engagement history.

    Identifies the point at which engagement begins declining and
    estimates how long the student can maintain focus.

    Args:
        engagement_history: List of engagement scores (0-1) over time.
        window_size: Rolling window size for trend detection.
        decline_threshold: Minimum drop to consider a decline.

    Returns:
        Dict with attention_span_minutes, is_declining, peak_engagement,
        current_engagement, and trend.
    """
    if len(engagement_history) < 2:
        return {
            "attention_span_minutes": 0.0,
            "is_declining": False,
            "peak_engagement": engagement_history[0] if engagement_history else 0.0,
            "current_engagement": engagement_history[-1] if engagement_history else 0.0,
            "trend": "insufficient_data",
        }

    arr = np.array(engagement_history, dtype=float)
    peak_idx = int(np.argmax(arr))
    peak_val = float(arr[peak_idx])
    current_val = float(arr[-1])

    # Detect sustained decline after peak
    # Each sample ~2 seconds (typical EEG epoch hop), so index / 30 ~ minutes
    decline_started = None
    if len(arr) >= window_size:
        for i in range(window_size, len(arr)):
            window_mean = float(np.mean(arr[i - window_size:i]))
            prev_mean = float(np.mean(arr[max(0, i - 2 * window_size):i - window_size]))
            if prev_mean - window_mean > decline_threshold:
                decline_started = i - window_size
                break

    is_declining = decline_started is not None
    # Estimate attention span in minutes (assume 2-second epochs)
    if decline_started is not None:
        attention_samples = decline_started
    else:
        attention_samples = len(arr)
    attention_span_min = round(attention_samples * 2.0 / 60.0, 1)

    # Trend
    if len(arr) >= window_size:
        recent = float(np.mean(arr[-window_size:]))
        older = float(np.mean(arr[:window_size]))
        diff = recent - older
        if diff > 0.05:
            trend = "improving"
        elif diff < -0.05:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    return {
        "attention_span_minutes": attention_span_min,
        "is_declining": is_declining,
        "peak_engagement": round(peak_val, 4),
        "current_engagement": round(current_val, 4),
        "trend": trend,
    }


def compute_learning_windows(
    hourly_engagement: Dict[int, List[float]],
) -> Dict:
    """Compute optimal learning windows from historical hourly engagement.

    Args:
        hourly_engagement: Dict mapping hour (0-23) to list of engagement
            scores recorded during that hour across sessions.

    Returns:
        Dict with optimal_hours, worst_hours, hourly_means, and
        recommendation.
    """
    if not hourly_engagement:
        return {
            "optimal_hours": [],
            "worst_hours": [],
            "hourly_means": {},
            "recommendation": "Not enough data to determine optimal learning windows.",
        }

    hourly_means: Dict[int, float] = {}
    for hour, scores in hourly_engagement.items():
        h = int(hour)
        if scores:
            hourly_means[h] = round(float(np.mean(scores)), 4)

    if not hourly_means:
        return {
            "optimal_hours": [],
            "worst_hours": [],
            "hourly_means": {},
            "recommendation": "Not enough data to determine optimal learning windows.",
        }

    sorted_hours = sorted(hourly_means, key=hourly_means.get, reverse=True)
    # Top 3 hours as optimal, bottom 3 as worst
    n_hours = len(sorted_hours)
    top_n = min(3, n_hours)
    optimal = sorted_hours[:top_n]
    worst = sorted_hours[-top_n:]

    # Format recommendation
    optimal_strs = [f"{h}:00" for h in optimal]
    recommendation = f"Best learning times: {', '.join(optimal_strs)}."

    return {
        "optimal_hours": optimal,
        "worst_hours": worst,
        "hourly_means": hourly_means,
        "recommendation": recommendation,
    }


def compute_education_profile(
    features_history: List[LearningEEGFeatures],
    hourly_engagement: Optional[Dict[int, List[float]]] = None,
) -> EducationProfile:
    """Compute a comprehensive education profile from session data.

    Args:
        features_history: List of EEG feature snapshots from the session.
        hourly_engagement: Optional historical hourly engagement data.

    Returns:
        EducationProfile with all metrics computed.
    """
    if not features_history:
        return EducationProfile()

    # Detect states for each sample
    states: List[LearningState] = []
    engagements: List[float] = []
    difficulty = 0.5

    for feat in features_history:
        ls = detect_learning_state(feat)
        states.append(ls)
        engagements.append(ls.engagement_level)

        # Adapt difficulty
        adj = recommend_difficulty_adjustment(ls, difficulty)
        difficulty = adj["new_difficulty"]

    # Current state is the latest
    current_state = states[-1]

    # Attention span
    attention_info = track_attention_span(engagements)
    current_state.attention_declining = attention_info["is_declining"]

    # State distribution
    from collections import Counter
    state_counts = Counter(s.state for s in states)
    n = len(states)
    state_dist = {s: round(state_counts.get(s, 0) / n * 100, 1) for s in LEARNING_STATES}

    # Effectiveness by state: mean engagement when in each state
    effectiveness: Dict[str, float] = {}
    for s_name in LEARNING_STATES:
        state_engs = [eng for eng, ls in zip(engagements, states) if ls.state == s_name]
        if state_engs:
            effectiveness[s_name] = round(float(np.mean(state_engs)), 4)

    # Learning windows
    if hourly_engagement:
        windows = compute_learning_windows(hourly_engagement)
        optimal_hours = windows["optimal_hours"]
    else:
        optimal_hours = []

    # Session duration estimate (2 sec per epoch)
    session_dur = round(len(features_history) * 2.0 / 60.0, 1)

    # Generate recommendations
    recs = _generate_recommendations(current_state, state_dist, attention_info, session_dur)

    return EducationProfile(
        learning_state=current_state,
        attention_span_minutes=attention_info["attention_span_minutes"],
        optimal_window_hours=optimal_hours,
        effectiveness_by_state=effectiveness,
        session_engagement_mean=round(float(np.mean(engagements)), 4),
        session_duration_minutes=session_dur,
        difficulty_level=round(difficulty, 4),
        state_distribution=state_dist,
        n_samples=n,
        recommendations=recs,
    )


def profile_to_dict(profile: EducationProfile) -> Dict:
    """Convert an EducationProfile to a JSON-serializable dict.

    Args:
        profile: The education profile to convert.

    Returns:
        Dict representation.
    """
    ls = profile.learning_state
    return {
        "learning_state": {
            "state": ls.state,
            "confidence": ls.confidence,
            "scores": ls.scores,
            "difficulty_recommendation": ls.difficulty_recommendation,
            "pacing_recommendation": ls.pacing_recommendation,
            "engagement_level": ls.engagement_level,
            "attention_declining": ls.attention_declining,
        },
        "attention_span_minutes": profile.attention_span_minutes,
        "optimal_window_hours": profile.optimal_window_hours,
        "effectiveness_by_state": profile.effectiveness_by_state,
        "session_engagement_mean": profile.session_engagement_mean,
        "session_duration_minutes": profile.session_duration_minutes,
        "difficulty_level": profile.difficulty_level,
        "state_distribution": profile.state_distribution,
        "n_samples": profile.n_samples,
        "recommendations": profile.recommendations,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_state_scores(
    theta: float, alpha: float, beta: float, valence: float, fatigue: float,
) -> Dict[str, float]:
    """Compute raw scores for each learning state.

    Args:
        theta: Theta power 0-1.
        alpha: Alpha power 0-1.
        beta: Beta power 0-1.
        valence: Emotional valence -1 to 1.
        fatigue: Fatigue index 0-1.

    Returns:
        Dict mapping state name to score (0-1).
    """
    # Engaged: moderate beta, low theta, LOW alpha (alert, not relaxed)
    beta_opt = 1.0 - abs(beta - 0.45) * 2.5
    engaged = float(np.clip(
        0.35 * max(beta_opt, 0)
        + 0.30 * (1.0 - theta)
        + 0.15 * (1.0 - fatigue)
        + 0.20 * (1.0 - alpha),
        0, 1,
    ))

    # Confused: high theta, high beta (frontal activation), low alpha
    confused = float(np.clip(
        0.40 * theta
        + 0.25 * beta
        + 0.20 * (1.0 - alpha)
        + 0.15 * max(0, -valence),
        0, 1,
    ))

    # Bored: high alpha, low beta, low theta
    bored = float(np.clip(
        0.45 * alpha
        + 0.30 * (1.0 - beta)
        + 0.25 * (1.0 - theta),
        0, 1,
    ))

    # Frustrated: high beta, negative valence, fatigue contribution
    neg_val = max(0, -valence)
    frustrated = float(np.clip(
        0.30 * beta
        + 0.25 * neg_val
        + 0.20 * fatigue
        + 0.15 * theta
        + 0.10 * (1.0 - alpha),
        0, 1,
    ))

    # Flow: alpha-theta border (balanced alpha/theta), moderate beta, low fatigue
    # Alpha-theta balance is the hallmark of flow — weighted heavily
    alpha_theta_balance = 1.0 - abs(alpha - theta) * 2.0
    fatigue_penalty = max(0, fatigue - 0.3) * 1.5
    flow = float(np.clip(
        0.35 * max(alpha_theta_balance, 0)
        + 0.25 * beta
        + 0.15 * (1.0 - fatigue)
        + 0.25 * max(0, valence),
        0, 1,
    )) - fatigue_penalty
    flow = float(np.clip(flow, 0, 1))

    return {
        "engaged": engaged,
        "confused": confused,
        "bored": bored,
        "frustrated": frustrated,
        "flow": flow,
    }


def _difficulty_reason(state: str) -> str:
    """Human-readable reason for difficulty adjustment."""
    reasons = {
        "engaged": "Student is engaged; maintain current difficulty.",
        "confused": "Student is confused; reduce difficulty to prevent overload.",
        "bored": "Student is bored; increase difficulty for more challenge.",
        "frustrated": "Student is frustrated; reduce difficulty and consider a different approach.",
        "flow": "Student is in flow; slightly increase difficulty to maintain challenge.",
    }
    return reasons.get(state, "Unknown state.")


def _pacing_suggestion(state: str) -> str:
    """Pacing suggestion for the current state."""
    suggestions = {
        "engaged": "Maintain current pace; student is actively learning.",
        "confused": "Slow down and revisit prerequisites before continuing.",
        "bored": "Pick up the pace or introduce more challenging material.",
        "frustrated": "Take a break and try a different approach when resuming.",
        "flow": "Do not interrupt; student is in an optimal learning state.",
    }
    return suggestions.get(state, "Continue at current pace.")


def _generate_recommendations(
    current_state: LearningState,
    state_dist: Dict[str, float],
    attention_info: Dict,
    session_dur: float,
) -> List[str]:
    """Generate actionable recommendations for the education profile."""
    recs: List[str] = []

    # Current state recommendation
    recs.append(_pacing_suggestion(current_state.state))

    # Attention span
    if attention_info.get("is_declining"):
        recs.append(
            f"Attention has been declining. Consider a break after "
            f"{attention_info['attention_span_minutes']:.0f} minutes of focus."
        )

    # State distribution warnings
    bored_pct = state_dist.get("bored", 0)
    frustrated_pct = state_dist.get("frustrated", 0)
    confused_pct = state_dist.get("confused", 0)

    if bored_pct > 30:
        recs.append("High boredom detected; consider more challenging material.")
    if frustrated_pct > 25:
        recs.append("Significant frustration detected; simplify content or change approach.")
    if confused_pct > 30:
        recs.append("Frequent confusion; review prerequisite knowledge.")

    # Session length
    if session_dur > 45:
        recs.append("Long session detected; breaks improve retention.")

    return recs
