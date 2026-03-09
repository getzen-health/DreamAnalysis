"""Adaptive learning speed detector via frontal theta dynamics.

Monitors frontal midline theta (FMT) to recommend optimal learning pace:
- High FMT = active encoding (good pace)
- Very high FMT = cognitive overload (slow down)
- Low FMT = disengagement (increase challenge or take break)

References:
    Cavanagh & Frank (2014) — Frontal theta and cognitive control
    Onton et al. (2005) — Frontal midline theta and learning
"""
from typing import Dict, List, Optional

import numpy as np


class AdaptiveLearningDetector:
    """Detect optimal learning pace from frontal theta dynamics."""

    def __init__(self):
        self._theta_history: Dict[str, List[float]] = {}
        self._baseline: Dict[str, float] = {}
        # Normalized theta range considered optimal for encoding
        self._optimal_low = 0.3
        self._optimal_high = 0.7

    def set_baseline(self, fmt_power: float, user_id: str = "default"):
        """Set resting-state FMT baseline for a user.

        Args:
            fmt_power: FMT power during resting state.
            user_id: User identifier.
        """
        if fmt_power <= 0:
            return
        self._baseline[user_id] = fmt_power

    def assess_pace(
        self,
        fmt_power: float,
        cognitive_load: float = 0.5,
        time_on_task_min: float = 0.0,
        user_id: str = "default",
    ) -> Dict:
        """Recommend learning pace based on FMT dynamics.

        Args:
            fmt_power: Current frontal midline theta power.
            cognitive_load: 0-1 cognitive load estimate.
            time_on_task_min: Minutes spent on current task.
            user_id: User identifier.

        Returns:
            Dict with recommendation, reason, theta_ratio, encoding_quality,
            and pace_history fields.
        """
        # Track history
        if user_id not in self._theta_history:
            self._theta_history[user_id] = []
        self._theta_history[user_id].append(fmt_power)
        if len(self._theta_history[user_id]) > 60:
            self._theta_history[user_id] = self._theta_history[user_id][-60:]

        # Compute theta ratio relative to baseline
        baseline = self._baseline.get(user_id, None)
        if baseline and baseline > 0:
            theta_ratio = fmt_power / baseline
        else:
            # No baseline — use absolute heuristic
            theta_ratio = fmt_power / max(fmt_power, 1e-6)  # 1.0 = self-reference
            theta_ratio = 1.0  # can't normalize without baseline

        # Normalize to 0-1 range for scoring
        # Typical elevated theta is 1.2-2.0x baseline
        normalized = np.clip((theta_ratio - 0.5) / 1.5, 0, 1)

        # Encoding quality score
        # Best encoding at moderate theta elevation (1.2-1.5x baseline)
        encoding_quality = float(1.0 - abs(normalized - 0.45) / 0.55)
        encoding_quality = float(np.clip(encoding_quality, 0, 1))

        # Compute trend from recent history
        trend = self._compute_trend(user_id)

        # Decision logic
        if baseline is None or baseline <= 0:
            # No baseline — use cognitive load as proxy
            recommendation, reason = self._assess_from_load(
                cognitive_load, time_on_task_min
            )
        elif theta_ratio > 1.8:
            recommendation = "slow_down"
            reason = "cognitive_overload"
        elif theta_ratio > 1.5 and cognitive_load > 0.75:
            recommendation = "slow_down"
            reason = "high_load_high_theta"
        elif theta_ratio < 0.7 and time_on_task_min > 5:
            recommendation = "increase_challenge"
            reason = "disengaged"
        elif theta_ratio < 0.8 and cognitive_load < 0.2:
            recommendation = "increase_challenge"
            reason = "under_stimulated"
        elif 1.1 <= theta_ratio <= 1.5 and cognitive_load < 0.7:
            recommendation = "optimal_pace"
            reason = "active_encoding"
        elif trend == "declining" and time_on_task_min > 20:
            recommendation = "take_break"
            reason = "theta_declining"
        else:
            recommendation = "maintain_pace"
            reason = "adequate_engagement"

        return {
            "recommendation": recommendation,
            "reason": reason,
            "theta_ratio": float(theta_ratio) if baseline else None,
            "encoding_quality": encoding_quality,
            "cognitive_load": float(cognitive_load),
            "time_on_task_min": float(time_on_task_min),
            "theta_trend": trend,
            "fmt_power": float(fmt_power),
        }

    def _assess_from_load(self, cognitive_load: float, time_on_task_min: float):
        """Fallback assessment when no FMT baseline is available."""
        if cognitive_load > 0.8:
            return "slow_down", "high_cognitive_load"
        elif cognitive_load < 0.2 and time_on_task_min > 5:
            return "increase_challenge", "low_cognitive_load"
        elif time_on_task_min > 30:
            return "take_break", "extended_session"
        else:
            return "maintain_pace", "no_baseline_available"

    def _compute_trend(self, user_id: str) -> str:
        """Compute FMT trend from recent history."""
        history = self._theta_history.get(user_id, [])
        if len(history) < 6:
            return "insufficient_data"

        recent = history[-6:]
        older = history[-12:-6] if len(history) >= 12 else history[:6]
        diff = float(np.mean(recent)) - float(np.mean(older))

        if diff > 0.05:
            return "rising"
        elif diff < -0.05:
            return "declining"
        return "stable"

    def get_history(self, user_id: str = "default") -> List[float]:
        """Get FMT power history for a user."""
        return list(self._theta_history.get(user_id, []))

    def reset(self, user_id: str = "default"):
        """Clear history and baseline for a user."""
        self._theta_history.pop(user_id, None)
        self._baseline.pop(user_id, None)
