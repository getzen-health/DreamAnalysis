"""EEG-driven neurogaming engine with adaptive difficulty.

Maps mental states (focus, relaxation, engagement) to game commands
and adapts game difficulty based on detected cognitive/emotional state.

Calibration protocol:
1. 30s eyes-closed relaxation → baseline alpha
2. 30s focused counting backwards → baseline beta/attention
3. 30s free state → idle threshold

References:
    Aleisa et al., HCII 2025 — Muse S game control with personalized calibration
    IEEE Spectrum 2024 — NeuroPlus ADHD brain-controlled games
    Neuroadaptive Gamification Review 2025 — Real-time BCI game mechanics
"""
from typing import Dict, List, Optional

import numpy as np

GAME_COMMANDS = ["focus_boost", "relax_action", "idle"]
DIFFICULTY_LEVELS = ["very_easy", "easy", "medium", "hard", "very_hard"]


class NeurogameEngine:
    """EEG-driven game state controller with adaptive difficulty.

    Converts EEG band powers into game commands and adjusts difficulty
    based on engagement, frustration, and boredom detection.
    """

    def __init__(self, max_history: int = 500):
        self._calibration: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}
        self._difficulty: Dict[str, float] = {}
        self._max_history = max_history

    def calibrate(
        self,
        focus_beta_theta: float,
        relax_alpha_beta: float,
        user_id: str = "default",
    ) -> Dict:
        """Set calibration thresholds from focus/relax baseline measurements.

        Args:
            focus_beta_theta: Beta/theta ratio during focused counting task.
            relax_alpha_beta: Alpha/beta ratio during relaxed eyes-closed.
            user_id: User identifier.

        Returns:
            Dict with calibration thresholds.
        """
        cal = {
            "focus_threshold": max(focus_beta_theta * 0.7, 0.5),
            "relax_threshold": max(relax_alpha_beta * 0.7, 0.5),
            "focus_mean": float(focus_beta_theta),
            "relax_mean": float(relax_alpha_beta),
            "calibrated": True,
        }
        self._calibration[user_id] = cal
        self._difficulty[user_id] = 0.5
        return cal

    def get_command(
        self,
        theta_power: float,
        alpha_power: float,
        beta_power: float,
        user_id: str = "default",
    ) -> Dict:
        """Convert EEG band powers into a game command.

        Args:
            theta_power: Theta (4-8 Hz) power, 0-1.
            alpha_power: Alpha (8-12 Hz) power, 0-1.
            beta_power: Beta (12-30 Hz) power, 0-1.
            user_id: User identifier.

        Returns:
            Dict with command, intensity, engagement_level,
            difficulty_adjustment, difficulty_level, game_state.
        """
        focus_score = beta_power / max(theta_power, 1e-10)
        relax_score = alpha_power / max(beta_power, 1e-10)

        # Engagement: beta relative to alpha+theta
        engagement = beta_power / (alpha_power + theta_power + 1e-10)
        engagement = float(np.clip(engagement, 0, 1))

        cal = self._calibration.get(user_id)

        if cal:
            focus_thresh = cal["focus_threshold"]
            relax_thresh = cal["relax_threshold"]
        else:
            focus_thresh = 1.5
            relax_thresh = 1.5

        # Determine command
        if focus_score > focus_thresh:
            command = "focus_boost"
            intensity = float(np.clip((focus_score - focus_thresh) / max(focus_thresh, 1e-10), 0, 1))
        elif relax_score > relax_thresh:
            command = "relax_action"
            intensity = float(np.clip((relax_score - relax_thresh) / max(relax_thresh, 1e-10), 0, 1))
        else:
            command = "idle"
            intensity = 0.0

        # Adaptive difficulty
        if user_id not in self._difficulty:
            self._difficulty[user_id] = 0.5

        difficulty = self._difficulty[user_id]
        diff_adj = self._compute_difficulty_adjustment(engagement, theta_power, alpha_power, beta_power)
        difficulty = float(np.clip(difficulty + diff_adj * 0.05, 0, 1))
        self._difficulty[user_id] = difficulty

        # Map to difficulty label
        if difficulty < 0.2:
            diff_label = "very_easy"
        elif difficulty < 0.4:
            diff_label = "easy"
        elif difficulty < 0.6:
            diff_label = "medium"
        elif difficulty < 0.8:
            diff_label = "hard"
        else:
            diff_label = "very_hard"

        result = {
            "command": command,
            "intensity": round(intensity, 4),
            "engagement_level": round(engagement, 4),
            "difficulty_adjustment": round(diff_adj, 4),
            "difficulty_value": round(difficulty, 4),
            "difficulty_level": diff_label,
            "focus_score": round(float(np.clip(focus_score, 0, 10)), 4),
            "relax_score": round(float(np.clip(relax_score, 0, 10)), 4),
            "calibrated": user_id in self._calibration,
        }

        # Record history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > self._max_history:
            self._history[user_id] = self._history[user_id][-self._max_history:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session gameplay statistics.

        Returns:
            Dict with command distribution, mean engagement,
            difficulty curve, and total commands.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {"total_commands": 0, "calibrated": user_id in self._calibration}

        commands = [h["command"] for h in history]
        engagements = [h["engagement_level"] for h in history]

        from collections import Counter
        cmd_counts = Counter(commands)

        return {
            "total_commands": len(history),
            "command_distribution": dict(cmd_counts),
            "mean_engagement": round(float(np.mean(engagements)), 4),
            "max_engagement": round(float(np.max(engagements)), 4),
            "current_difficulty": round(self._difficulty.get(user_id, 0.5), 4),
            "difficulty_level": history[-1]["difficulty_level"],
            "calibrated": user_id in self._calibration,
        }

    def get_history(self, user_id: str = "default", last_n: Optional[int] = None) -> List[Dict]:
        """Get command history for replay/analysis."""
        history = self._history.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return history

    def reset(self, user_id: str = "default"):
        """Clear history and calibration for a user."""
        self._history.pop(user_id, None)
        self._difficulty.pop(user_id, None)
        self._calibration.pop(user_id, None)

    def _compute_difficulty_adjustment(
        self, engagement: float, theta: float, alpha: float, beta: float
    ) -> float:
        """Compute difficulty adjustment from -1 to +1.

        +1 = increase difficulty (user is highly engaged / in flow)
        -1 = decrease difficulty (user is bored or frustrated)
        """
        # High engagement + moderate alpha = flow → increase
        # Low engagement + high alpha = bored → decrease
        # High beta + high theta = frustrated → decrease

        boredom = 0.50 * alpha + 0.30 * (1 - beta) + 0.20 * (1 - theta)
        frustration = 0.40 * beta + 0.35 * theta + 0.25 * (1 - alpha)

        if engagement > 0.6 and boredom < 0.5:
            # In flow — increase difficulty
            return float(np.clip(engagement - 0.5, 0, 1))
        elif boredom > 0.6:
            # Bored — decrease difficulty
            return -float(np.clip(boredom - 0.5, 0, 1))
        elif frustration > 0.7:
            # Frustrated — decrease difficulty
            return -float(np.clip(frustration - 0.6, 0, 1))
        else:
            return 0.0
