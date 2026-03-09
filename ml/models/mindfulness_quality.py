"""Mindfulness Quality Classifier for active meditation sessions.

Detects mind-wandering during meditation using frontal theta/alpha patterns
and tracks quality over the session duration with a rolling window.

Key biomarkers:
- Theta/alpha ratio: high ratio = focused meditation; drop = mind-wandering
  (Brandmeyer & Delorme, 2018; Hasenkamp et al., 2012)
- Theta power stability: stable theta = sustained focus; variable = wandering
  (Braboszcz & Delorme, 2011)
- Alpha power: increased alpha relative to theta = relaxation/disengagement
  rather than active meditation focus (Lomas et al., 2015)

Quality states:
  focused          — sustained attention on meditation object (high theta/alpha)
  light_wandering  — brief lapses in focus, recoverable
  mind_wandering   — attention has drifted to spontaneous thought
  disengaged       — drowsiness or loss of meditative effort

Reference: Brandmeyer & Delorme (2018), Hasenkamp et al. (2012),
           Braboszcz & Delorme (2011), Lomas et al. (2015)
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional


QUALITY_STATES = ["focused", "light_wandering", "mind_wandering", "disengaged"]


class MindfulnessQualityDetector:
    """Detects mind-wandering during active meditation using EEG band power ratios.

    Tracks per-user session history and computes rolling stability metrics
    to give actionable feedback about meditation quality.
    """

    # Theta/alpha ratio thresholds for quality classification.
    # Based on Brandmeyer & Delorme (2018): focused meditation produces
    # theta/alpha > 1.2 at frontal sites; mind-wandering drops below 0.8.
    FOCUSED_THRESHOLD = 1.2
    LIGHT_WANDERING_THRESHOLD = 0.8
    MIND_WANDERING_THRESHOLD = 0.5

    # Rolling window size for stability computation (number of assessments)
    STABILITY_WINDOW = 10

    def __init__(self) -> None:
        # Per-user session data: user_id -> list of (timestamp, state, ratio) tuples
        self._sessions: Dict[str, List[Dict]] = defaultdict(list)

    def assess(
        self,
        theta_power: float,
        alpha_power: float,
        beta_power: float = 0.0,
        meditation_duration_sec: float = 0.0,
        user_id: str = "default",
    ) -> Dict:
        """Assess mindfulness quality from band power values.

        Args:
            theta_power: Theta band power (4-8 Hz). Higher = more meditative focus.
            alpha_power: Alpha band power (8-12 Hz). Higher = relaxation/disengagement.
            beta_power: Beta band power (12-30 Hz). Used as secondary restlessness signal.
            meditation_duration_sec: How many seconds into the session. Used for
                session-aware feedback (e.g., early wandering vs late fatigue).
            user_id: User identifier for per-user session tracking.

        Returns:
            Dict with quality state, metrics, and recommendation.
        """
        # Compute theta/alpha ratio (primary biomarker)
        safe_alpha = max(alpha_power, 1e-10)
        focus_ratio = theta_power / safe_alpha

        # Classify quality state
        quality = self._classify_quality(focus_ratio, beta_power)

        # Record this assessment
        entry = {
            "timestamp": time.time(),
            "state": quality,
            "focus_ratio": focus_ratio,
            "theta_power": theta_power,
            "alpha_power": alpha_power,
            "beta_power": beta_power,
            "meditation_duration_sec": meditation_duration_sec,
        }
        self._sessions[user_id].append(entry)

        session = self._sessions[user_id]

        # Compute stability score from recent theta power values
        stability_score = self._compute_stability(session)

        # Compute mind-wandering percentage over session
        mind_wandering_pct = self._compute_wandering_pct(session)

        # Compute current streak (consecutive seconds in current state)
        current_streak_sec = self._compute_streak(session, quality)

        # Generate actionable recommendation
        recommendation = self._generate_recommendation(
            quality, mind_wandering_pct, stability_score, meditation_duration_sec
        )

        return {
            "quality": quality,
            "focus_ratio": round(focus_ratio, 4),
            "mind_wandering_pct": round(mind_wandering_pct, 2),
            "current_streak_sec": round(current_streak_sec, 1),
            "stability_score": round(stability_score, 4),
            "recommendation": recommendation,
        }

    def get_session_summary(self, user_id: str = "default") -> Dict:
        """Return session statistics for a user.

        Args:
            user_id: User identifier.

        Returns:
            Dict with total_focused_sec, total_wandering_sec, focus_pct,
            and quality_timeline (list of state strings).
        """
        session = self._sessions.get(user_id, [])
        if not session:
            return {
                "total_focused_sec": 0.0,
                "total_wandering_sec": 0.0,
                "focus_pct": 0.0,
                "quality_timeline": [],
            }

        # Estimate time per assessment: use timestamp differences where available,
        # otherwise default to 1 second per assessment.
        focused_count = 0
        wandering_count = 0
        quality_timeline: List[str] = []

        for entry in session:
            state = entry["state"]
            quality_timeline.append(state)
            if state == "focused":
                focused_count += 1
            else:
                wandering_count += 1

        total = focused_count + wandering_count

        # Compute actual durations using timestamps.
        # Use a minimum of 1 second per entry to handle rapid-fire calls
        # (e.g., tests or batch processing where timestamps are near-identical).
        total_focused_sec = 0.0
        total_wandering_sec = 0.0
        for i, entry in enumerate(session):
            if i + 1 < len(session):
                dt = session[i + 1]["timestamp"] - entry["timestamp"]
                dt = max(dt, 1.0)  # Minimum 1 second per assessment
            else:
                # Last entry: use 1 second as default duration
                dt = 1.0
            # Cap individual dt to 10 seconds to handle gaps
            dt = min(dt, 10.0)
            if entry["state"] == "focused":
                total_focused_sec += dt
            else:
                total_wandering_sec += dt

        total_time = total_focused_sec + total_wandering_sec
        focus_pct = (total_focused_sec / total_time * 100.0) if total_time > 0 else 0.0

        return {
            "total_focused_sec": round(total_focused_sec, 1),
            "total_wandering_sec": round(total_wandering_sec, 1),
            "focus_pct": round(focus_pct, 1),
            "quality_timeline": quality_timeline,
        }

    def reset(self, user_id: str = "default") -> None:
        """Clear session data for a user.

        Args:
            user_id: User identifier whose session data to clear.
        """
        if user_id in self._sessions:
            del self._sessions[user_id]

    # ── Private helpers ──────────────────────────────────────────────────────

    def _classify_quality(self, focus_ratio: float, beta_power: float) -> str:
        """Classify quality state from theta/alpha ratio and beta power.

        Classification logic:
        - focus_ratio >= 1.2 → focused (strong theta dominance)
        - 0.8 <= focus_ratio < 1.2 → light_wandering (theta still present)
        - 0.5 <= focus_ratio < 0.8 → mind_wandering (alpha taking over)
        - focus_ratio < 0.5 → disengaged (theta collapsed, drowsy or distracted)

        Beta power acts as a secondary signal: elevated beta (> 0.3) during
        low focus_ratio pushes toward mind_wandering (restless thinking)
        rather than disengaged (drowsy).
        """
        if focus_ratio >= self.FOCUSED_THRESHOLD:
            return "focused"
        elif focus_ratio >= self.LIGHT_WANDERING_THRESHOLD:
            return "light_wandering"
        elif focus_ratio >= self.MIND_WANDERING_THRESHOLD:
            # High beta + low theta/alpha = restless mind-wandering (not drowsy)
            return "mind_wandering"
        else:
            # Very low theta/alpha could be drowsy or distracted.
            # High beta suggests active distraction; low beta suggests drowsiness.
            if beta_power > 0.3:
                return "mind_wandering"
            return "disengaged"

    def _compute_stability(self, session: List[Dict]) -> float:
        """Compute stability score (0-1) from recent theta power variance.

        Uses coefficient of variation (CV) of theta power over the last
        STABILITY_WINDOW assessments. Low CV = stable = high score.
        """
        if len(session) < 2:
            return 1.0  # Not enough data; assume stable

        window = session[-self.STABILITY_WINDOW:]
        theta_values = [e["theta_power"] for e in window]
        mean_theta = sum(theta_values) / len(theta_values)
        if mean_theta < 1e-10:
            return 0.0  # Zero theta = no meditation signal

        variance = sum((t - mean_theta) ** 2 for t in theta_values) / len(theta_values)
        std_theta = variance ** 0.5
        cv = std_theta / mean_theta  # Coefficient of variation

        # Map CV to stability score: CV=0 → 1.0, CV>=1 → 0.0
        stability = max(0.0, min(1.0, 1.0 - cv))
        return stability

    def _compute_wandering_pct(self, session: List[Dict]) -> float:
        """Compute percentage of session spent in any non-focused state."""
        if not session:
            return 0.0
        non_focused = sum(1 for e in session if e["state"] != "focused")
        return (non_focused / len(session)) * 100.0

    def _compute_streak(self, session: List[Dict], current_state: str) -> float:
        """Compute consecutive seconds in the current state.

        Walks backwards from the most recent assessment, counting entries
        that share the current state, and sums their timestamp deltas.
        """
        if not session:
            return 0.0

        streak_sec = 0.0
        for i in range(len(session) - 1, -1, -1):
            if session[i]["state"] != current_state:
                break
            if i + 1 < len(session):
                dt = session[i + 1]["timestamp"] - session[i]["timestamp"]
                streak_sec += min(dt, 10.0)
            else:
                # Last (most recent) entry: count 1 second
                streak_sec += 1.0

        return streak_sec

    def _generate_recommendation(
        self,
        quality: str,
        wandering_pct: float,
        stability_score: float,
        duration_sec: float,
    ) -> str:
        """Generate actionable feedback based on current state and session context."""
        if quality == "focused":
            if stability_score >= 0.8:
                return "Excellent focus. Maintain this steady awareness."
            return "Good focus. Try to keep your attention stable."

        if quality == "light_wandering":
            if duration_sec < 120:
                return "Gently redirect attention to your breath. Early wandering is normal."
            return "Notice where your mind went and gently return to the meditation object."

        if quality == "mind_wandering":
            if wandering_pct > 60:
                return "Frequent wandering detected. Try counting breaths 1-10 to anchor attention."
            return "Mind has wandered. Acknowledge the thought without judgment and return to focus."

        # disengaged
        if duration_sec > 600:
            return "You may be fatigued. Consider ending the session or taking a brief pause."
        return "Attention has dropped significantly. Take a deep breath and re-engage with your practice."
