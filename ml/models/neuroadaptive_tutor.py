"""Neuroadaptive learning system — EEG-driven adaptive tutoring.

Monitors cognitive state during study sessions and outputs real-time
learning recommendations: difficulty adjustment, break timing,
content switching, and engagement trend analysis.

Maintains the learner in Vygotsky's Zone of Proximal Development
via neural feedback — not too easy (boredom), not too hard (confusion).

References:
    Springer 2024 — 7 educational emotions from 5 EEG channels
    BrainAccess 2025 — Real-time EEG feedback for educators
    Social Network Analysis & Mining 2025 — CNN engagement classification
"""
from typing import Dict, List, Optional

import numpy as np

LEARNING_ZONES = ["boredom", "confusion", "frustration", "optimal", "flow"]

INTERVENTIONS = {
    "boredom": "increase_difficulty",
    "confusion": "simplify_or_hint",
    "frustration": "change_approach",
    "optimal": "maintain",
    "flow": "do_not_interrupt",
}


class NeuroadaptiveTutor:
    """EEG-driven adaptive learning controller.

    Takes band powers and outputs learning zone classification,
    difficulty adjustment, break recommendations, and session analytics.
    """

    def __init__(self, max_history: int = 500):
        self._history: Dict[str, List[Dict]] = {}
        self._difficulty: Dict[str, float] = {}
        self._consec: Dict[str, Dict[str, int]] = {}
        self._break_cooldown: Dict[str, int] = {}
        self._max_history = max_history

    def assess(
        self,
        theta_power: float,
        alpha_power: float,
        beta_power: float,
        fatigue_index: float = 0.0,
        session_minutes: float = 0.0,
        user_id: str = "default",
    ) -> Dict:
        """Assess learning zone and output adaptation recommendations.

        Args:
            theta_power: Theta band (4-8 Hz), 0-1.
            alpha_power: Alpha band (8-12 Hz), 0-1.
            beta_power: Beta band (12-30 Hz), 0-1.
            fatigue_index: External fatigue score 0-1 (optional).
            session_minutes: Minutes since session start.
            user_id: User identifier.

        Returns:
            Dict with learning_zone, intervention, difficulty_adjustment,
            difficulty_level, break_recommended, engagement_score,
            zone_confidence, n_samples.
        """
        # Initialize per-user state
        if user_id not in self._difficulty:
            self._difficulty[user_id] = 0.5
        if user_id not in self._consec:
            self._consec[user_id] = {z: 0 for z in LEARNING_ZONES}
        if user_id not in self._break_cooldown:
            self._break_cooldown[user_id] = 0

        # Compute zone scores
        scores = self._compute_zone_scores(theta_power, alpha_power, beta_power, fatigue_index)
        zone = max(scores, key=scores.get)
        confidence = float(scores[zone])

        # Update consecutive counters
        for z in LEARNING_ZONES:
            if z == zone:
                self._consec[user_id][z] += 1
            else:
                self._consec[user_id][z] = 0

        # Difficulty adjustment
        diff_adj = self._difficulty_adjustment(zone, confidence)
        difficulty = self._difficulty[user_id]
        difficulty = float(np.clip(difficulty + diff_adj * 0.03, 0, 1))
        self._difficulty[user_id] = difficulty

        # Break recommendation
        self._break_cooldown[user_id] += 1
        break_rec = self._should_break(
            zone, fatigue_index, session_minutes,
            self._consec[user_id], self._break_cooldown[user_id]
        )

        # Engagement score (beta-dominant = engaged)
        engagement = float(np.clip(
            0.50 * beta_power / max(theta_power, 1e-10) / 3.0
            + 0.30 * (1 - alpha_power)
            + 0.20 * beta_power,
            0, 1
        ))

        # Difficulty label
        if difficulty < 0.3:
            diff_label = "easy"
        elif difficulty < 0.7:
            diff_label = "moderate"
        else:
            diff_label = "hard"

        result = {
            "learning_zone": zone,
            "zone_scores": {k: round(v, 4) for k, v in scores.items()},
            "zone_confidence": round(confidence, 4),
            "intervention": INTERVENTIONS[zone],
            "difficulty_adjustment": round(diff_adj, 4),
            "difficulty_level": round(difficulty, 4),
            "difficulty_label": diff_label,
            "engagement_score": round(engagement, 4),
            "break_recommended": break_rec,
            "consecutive_zone": self._consec[user_id][zone],
            "session_minutes": round(session_minutes, 1),
            "n_samples": len(self._history.get(user_id, [])) + 1,
        }

        # Record history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > self._max_history:
            self._history[user_id] = self._history[user_id][-self._max_history:]

        return result

    def acknowledge_break(self, user_id: str = "default"):
        """Reset break cooldown after user takes a break."""
        self._break_cooldown[user_id] = 0

    def get_session_summary(self, user_id: str = "default") -> Dict:
        """Get session learning analytics."""
        history = self._history.get(user_id, [])
        if not history:
            return {"n_samples": 0}

        zones = [h["learning_zone"] for h in history]
        engagements = [h["engagement_score"] for h in history]
        n = len(history)

        from collections import Counter
        zone_counts = Counter(zones)

        return {
            "n_samples": n,
            "mean_engagement": round(float(np.mean(engagements)), 4),
            "optimal_pct": round(zone_counts.get("optimal", 0) / n * 100, 1),
            "flow_pct": round(zone_counts.get("flow", 0) / n * 100, 1),
            "boredom_pct": round(zone_counts.get("boredom", 0) / n * 100, 1),
            "confusion_pct": round(zone_counts.get("confusion", 0) / n * 100, 1),
            "frustration_pct": round(zone_counts.get("frustration", 0) / n * 100, 1),
            "dominant_zone": zone_counts.most_common(1)[0][0],
            "current_difficulty": round(self._difficulty.get(user_id, 0.5), 4),
            "breaks_taken": sum(1 for h in history if h["break_recommended"]),
        }

    def get_history(self, user_id: str = "default", last_n: Optional[int] = None) -> List[Dict]:
        """Get assessment history."""
        history = self._history.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return history

    def reset(self, user_id: str = "default"):
        """Clear all data for a user."""
        self._history.pop(user_id, None)
        self._difficulty.pop(user_id, None)
        self._consec.pop(user_id, None)
        self._break_cooldown.pop(user_id, None)

    # ── Private helpers ──────────────────────────────────────────

    def _compute_zone_scores(
        self, theta: float, alpha: float, beta: float, fatigue: float
    ) -> Dict[str, float]:
        """Score each learning zone based on EEG features."""
        # Boredom: high alpha + low beta + low theta (understimulated)
        boredom = 0.40 * alpha + 0.35 * (1 - beta) + 0.25 * (1 - theta)

        # Confusion: high theta + moderate beta (cognitive conflict)
        confusion = 0.45 * theta + 0.30 * (1 - alpha) + 0.25 * min(beta * 1.5, 1)

        # Frustration: high beta + high theta + low alpha + fatigue
        frustration = (
            0.30 * beta + 0.25 * theta + 0.20 * (1 - alpha) + 0.25 * fatigue
        )

        # Optimal: moderate everything, low fatigue
        # Best when theta ~0.3-0.5, beta ~0.4-0.6, alpha moderate
        theta_opt = 1.0 - abs(theta - 0.4) * 2
        beta_opt = 1.0 - abs(beta - 0.5) * 2
        optimal = (
            0.35 * max(theta_opt, 0) + 0.35 * max(beta_opt, 0)
            + 0.15 * (1 - alpha) + 0.15 * (1 - fatigue)
        )

        # Flow: high beta + moderate theta + low alpha + low fatigue
        # Fatigue is a strong flow-breaker — you can't be in flow when exhausted
        fatigue_penalty = max(0, fatigue - 0.3) * 1.5
        flow = (
            0.40 * beta + 0.25 * min(theta * 2, 1) + 0.10 * (1 - alpha)
            + 0.25 * (1 - fatigue)
        ) - fatigue_penalty

        # Normalize so they're relative
        scores = {
            "boredom": float(np.clip(boredom, 0, 1)),
            "confusion": float(np.clip(confusion, 0, 1)),
            "frustration": float(np.clip(frustration, 0, 1)),
            "optimal": float(np.clip(optimal, 0, 1)),
            "flow": float(np.clip(flow, 0, 1)),
        }
        return scores

    def _difficulty_adjustment(self, zone: str, confidence: float) -> float:
        """Compute difficulty change from -1 to +1."""
        if zone == "boredom":
            return confidence * 0.8   # increase difficulty
        elif zone == "confusion":
            return -confidence * 0.6  # decrease
        elif zone == "frustration":
            return -confidence * 0.8  # decrease more
        elif zone == "flow":
            return confidence * 0.3   # slight increase to maintain challenge
        else:  # optimal
            return 0.0

    def _should_break(
        self,
        zone: str,
        fatigue: float,
        session_min: float,
        consec: Dict[str, int],
        cooldown: int,
    ) -> bool:
        """Determine if a break should be recommended."""
        # Don't recommend breaks too frequently (min 50 epochs ~100 sec apart)
        if cooldown < 50:
            return False
        # Fatigue-based
        if fatigue > 0.7:
            return True
        # Time-based (every 25 min = Pomodoro)
        if session_min > 0 and session_min % 25 < 1 and cooldown > 200:
            return True
        # Prolonged frustration
        if consec.get("frustration", 0) >= 10:
            return True
        # Prolonged confusion
        if consec.get("confusion", 0) >= 15:
            return True
        return False
