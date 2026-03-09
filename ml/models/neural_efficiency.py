"""Neural Efficiency / Skill Mastery Tracker.

Tracks alpha power trends across sessions to measure neural efficiency —
the idea that experts' brains use *less* energy (higher alpha = more cortical
idling) to perform the same task as novices.

Scientific basis:
- Neural efficiency hypothesis (Neubauer & Fink, 2009): higher-ability
  individuals show lower cortical activation during cognitive tasks.
- In EEG terms, more alpha power during task performance = more efficient
  neural processing = higher mastery.
- Longitudinal alpha increase across practice sessions indicates skill
  acquisition (Babiloni et al., 2010).

Usage:
    tracker = NeuralEfficiencyTracker()
    # First session establishes baseline
    tracker.record_session("user1", alpha_power=0.25, task_type="meditation")
    # Subsequent sessions measure efficiency relative to baseline
    result = tracker.assess(alpha_power=0.30, user_id="user1", task_type="meditation")
    # result["efficiency_score"] = 0.30 / 0.25 = 1.2
    # result["mastery_stage"] = "proficient"
"""

import numpy as np
from typing import Dict, List, Optional


# Mastery stage thresholds (efficiency_score boundaries)
_MASTERY_THRESHOLDS = {
    "novice": 1.0,       # score < 1.0 or < 5 sessions
    "developing": 1.15,  # score 1.0 - 1.15
    "proficient": 1.3,   # score 1.15 - 1.3
    "expert": float("inf"),  # score > 1.3
}

MASTERY_STAGES = ["novice", "developing", "proficient", "expert"]

# Minimum sessions before leaving novice classification
_MIN_SESSIONS_FOR_PROGRESSION = 5


class NeuralEfficiencyTracker:
    """Track alpha power across sessions to measure neural efficiency.

    Maintains per-user, per-task session history. The first recorded
    session establishes the baseline alpha power; subsequent sessions
    compute efficiency_score = current_alpha / baseline_alpha.

    Thread safety: not thread-safe. Use one instance per worker or
    wrap calls with a lock in production.
    """

    def __init__(self) -> None:
        # _history[user_id][task_type] = list of alpha power floats
        self._history: Dict[str, Dict[str, List[float]]] = {}

    def record_session(
        self,
        user_id: str,
        alpha_power: float,
        task_type: str = "default",
    ) -> Dict:
        """Record alpha power for a completed session.

        The first session for a (user_id, task_type) pair becomes the
        baseline. Alpha power should be extracted from task-epoch EEG
        (not resting state).

        Args:
            user_id: Unique user identifier.
            alpha_power: Mean alpha band power during the session (any
                         unit — ratio is what matters).
            task_type: Task category (e.g., "meditation", "focus", "default").

        Returns:
            Dict with session_number and baseline_alpha.
        """
        if user_id not in self._history:
            self._history[user_id] = {}
        if task_type not in self._history[user_id]:
            self._history[user_id][task_type] = []

        self._history[user_id][task_type].append(float(alpha_power))

        sessions = self._history[user_id][task_type]
        return {
            "session_number": len(sessions),
            "baseline_alpha": sessions[0],
        }

    def assess(
        self,
        alpha_power: float,
        user_id: str = "default",
        task_type: str = "default",
    ) -> Dict:
        """Assess neural efficiency for a given alpha power reading.

        If no prior sessions exist for this (user_id, task_type), the
        provided alpha_power is treated as the first session (baseline)
        and automatically recorded.

        Args:
            alpha_power: Current session's mean alpha band power.
            user_id: Unique user identifier.
            task_type: Task category.

        Returns:
            Dict with efficiency_score, mastery_stage, sessions_completed,
            alpha_trend, and improvement_pct.
        """
        # Auto-record if no history exists yet
        history = self.get_history(user_id, task_type)
        if len(history) == 0:
            self.record_session(user_id, alpha_power, task_type)
            return {
                "efficiency_score": 1.0,
                "mastery_stage": "novice",
                "sessions_completed": 1,
                "alpha_trend": "stable",
                "improvement_pct": 0.0,
                "baseline_alpha": float(alpha_power),
                "current_alpha": float(alpha_power),
            }

        baseline_alpha = history[0]
        sessions_completed = len(history)

        # Guard against zero / near-zero baseline
        if abs(baseline_alpha) < 1e-10:
            efficiency_score = 1.0
        else:
            efficiency_score = float(alpha_power) / baseline_alpha

        # Mastery stage
        mastery_stage = self._classify_mastery(
            efficiency_score, sessions_completed
        )

        # Alpha trend from last 5 sessions (including current reading)
        recent = list(history[-4:]) + [float(alpha_power)]
        alpha_trend = self._compute_trend(recent)

        # Improvement percentage vs first session
        if abs(baseline_alpha) < 1e-10:
            improvement_pct = 0.0
        else:
            improvement_pct = round(
                ((float(alpha_power) - baseline_alpha) / abs(baseline_alpha)) * 100,
                2,
            )

        return {
            "efficiency_score": round(efficiency_score, 4),
            "mastery_stage": mastery_stage,
            "sessions_completed": sessions_completed,
            "alpha_trend": alpha_trend,
            "improvement_pct": improvement_pct,
            "baseline_alpha": round(baseline_alpha, 6),
            "current_alpha": round(float(alpha_power), 6),
        }

    def get_history(
        self, user_id: str, task_type: str = "default"
    ) -> List[float]:
        """Return session alpha history for a user and task type.

        Args:
            user_id: Unique user identifier.
            task_type: Task category.

        Returns:
            List of alpha power values, one per session, in chronological
            order. Empty list if no sessions recorded.
        """
        return list(
            self._history.get(user_id, {}).get(task_type, [])
        )

    def reset(self, user_id: str) -> None:
        """Clear all session history for a user (all task types).

        Args:
            user_id: Unique user identifier.
        """
        if user_id in self._history:
            del self._history[user_id]

    # ── Private helpers ─────────────────────────────────────────────

    @staticmethod
    def _classify_mastery(efficiency_score: float, sessions: int) -> str:
        """Map efficiency score + session count to a mastery stage.

        Users with fewer than _MIN_SESSIONS_FOR_PROGRESSION sessions
        are always classified as "novice" regardless of score, to
        prevent premature classification from a lucky early session.
        """
        if sessions < _MIN_SESSIONS_FOR_PROGRESSION or efficiency_score < 1.0:
            return "novice"
        if efficiency_score < 1.15:
            return "developing"
        if efficiency_score < 1.3:
            return "proficient"
        return "expert"

    @staticmethod
    def _compute_trend(values: List[float]) -> str:
        """Determine directional trend from a list of values.

        Uses linear regression slope on the last N values.

        Returns:
            "rising", "declining", or "stable".
        """
        if len(values) < 2:
            return "stable"

        arr = np.array(values, dtype=np.float64)
        x = np.arange(len(arr), dtype=np.float64)

        # Least-squares slope
        x_mean = np.mean(x)
        y_mean = np.mean(arr)
        denom = np.sum((x - x_mean) ** 2)

        if denom < 1e-10:
            return "stable"

        slope = np.sum((x - x_mean) * (arr - y_mean)) / denom

        # Normalize slope by mean to get relative rate of change
        if abs(y_mean) < 1e-10:
            return "stable"

        relative_slope = slope / abs(y_mean)

        # Threshold: 2% per session is considered meaningful change
        if relative_slope > 0.02:
            return "rising"
        elif relative_slope < -0.02:
            return "declining"
        return "stable"
