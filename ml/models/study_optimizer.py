"""EEG-guided adaptive study session optimizer.

Uses existing attention, cognitive load, and frontal midline theta (FMT)
to recommend optimal study actions: continue, take break, adjust difficulty,
or trigger spaced repetition review.
"""
import time
from typing import Dict, Optional

import numpy as np


class StudyOptimizer:
    """Rule-based study session optimizer using EEG engagement signals."""

    def __init__(self):
        self._session_start: Optional[float] = None
        self._last_break: Optional[float] = None
        self._attention_history: list = []  # rolling window
        self._fmt_baseline: Optional[float] = None

    def set_baseline(self, fmt_power: float):
        """Set FMT baseline from calibration."""
        self._fmt_baseline = fmt_power

    def start_session(self):
        """Mark session start time."""
        self._session_start = time.time()
        self._last_break = time.time()
        self._attention_history = []

    def recommend(
        self,
        attention_score: float,
        cognitive_load: float,
        fmt_power: float = 0.0,
        session_duration_min: float = 0.0,
        time_since_break_min: float = 0.0,
    ) -> Dict:
        """Generate study recommendation based on current brain state.

        Args:
            attention_score: 0-1 attention level from attention classifier
            cognitive_load: 0-1 cognitive load from load estimator
            fmt_power: frontal midline theta power (memory encoding signal)
            session_duration_min: minutes since session started
            time_since_break_min: minutes since last break

        Returns:
            Dict with action, reason, engagement_trend, and context fields.
        """
        # Track attention history (keep last 30 data points ~ 1 minute at 2s intervals)
        self._attention_history.append(attention_score)
        if len(self._attention_history) > 30:
            self._attention_history = self._attention_history[-30:]

        # Compute engagement trend
        trend = self._compute_trend()

        # FMT encoding detection
        fmt_elevated = False
        if self._fmt_baseline and self._fmt_baseline > 0:
            fmt_elevated = fmt_power > self._fmt_baseline * 1.3

        # Rule-based recommendations (priority order)

        # 1. Microsleep / severe disengagement -- urgent break
        if attention_score < 0.2 and time_since_break_min > 10:
            return {
                "action": "take_break",
                "reason": "Very low attention detected -- rest needed",
                "break_duration_min": 10,
                "urgency": "high",
                "engagement_trend": trend,
                "encoding_state": fmt_elevated,
            }

        # 2. Sustained low attention -- suggest break
        if attention_score < 0.4 and time_since_break_min > 20:
            return {
                "action": "take_break",
                "reason": "Attention declining -- a short break will help",
                "break_duration_min": 5,
                "urgency": "medium",
                "engagement_trend": trend,
                "encoding_state": fmt_elevated,
            }

        # 3. Cognitive overload -- reduce difficulty
        if cognitive_load > 0.8 and session_duration_min > 15:
            return {
                "action": "reduce_difficulty",
                "reason": "High cognitive load -- switch to easier material",
                "urgency": "medium",
                "engagement_trend": trend,
                "encoding_state": fmt_elevated,
            }

        # 4. Optimal review window -- FMT elevated (high memory encoding)
        if fmt_elevated and attention_score > 0.5:
            return {
                "action": "review_now",
                "reason": "High memory encoding state -- optimal time to review key concepts",
                "urgency": "low",
                "engagement_trend": trend,
                "encoding_state": True,
            }

        # 5. Boredom / under-stimulation -- increase difficulty
        if attention_score > 0.7 and cognitive_load < 0.3 and session_duration_min > 10:
            return {
                "action": "increase_difficulty",
                "reason": "Low cognitive load with good attention -- try harder material",
                "urgency": "low",
                "engagement_trend": trend,
                "encoding_state": fmt_elevated,
            }

        # 6. Long session without break
        if time_since_break_min > 45:
            return {
                "action": "take_break",
                "reason": "Extended study session -- take a break to maintain performance",
                "break_duration_min": 10,
                "urgency": "medium",
                "engagement_trend": trend,
                "encoding_state": fmt_elevated,
            }

        # 7. All good -- continue
        return {
            "action": "continue",
            "reason": "Good engagement -- keep going",
            "urgency": "none",
            "engagement_trend": trend,
            "encoding_state": fmt_elevated,
        }

    def _compute_trend(self) -> str:
        """Compute attention trend from history."""
        if len(self._attention_history) < 6:
            return "insufficient_data"

        recent = self._attention_history[-6:]  # last ~12 seconds
        older = (
            self._attention_history[-12:-6]
            if len(self._attention_history) >= 12
            else self._attention_history[:6]
        )

        recent_mean = float(np.mean(recent))
        older_mean = float(np.mean(older))
        diff = recent_mean - older_mean

        if diff > 0.1:
            return "rising"
        elif diff < -0.1:
            return "declining"
        return "stable"

    def get_session_stats(self) -> Dict:
        """Get summary stats for current study session."""
        if not self._attention_history:
            return {
                "mean_attention": 0,
                "min_attention": 0,
                "max_attention": 0,
                "n_samples": 0,
            }

        return {
            "mean_attention": float(np.mean(self._attention_history)),
            "min_attention": float(np.min(self._attention_history)),
            "max_attention": float(np.max(self._attention_history)),
            "n_samples": len(self._attention_history),
        }
