"""Student Engagement Detector from EEG band powers.

Classifies engagement into 3 states:
  - attentive:   actively processing, high beta/theta ratio
  - passive:     receiving information but not deeply engaged
  - disengaged:  mind-wandering, zoned out, minimal task engagement

Also detects 5 educational emotions:
  - boredom:       understimulated (high alpha, low beta/theta)
  - confusion:     cognitive conflict (high theta + high beta)
  - curiosity:     active exploration (moderate theta + moderate beta, low alpha)
  - frustration:   effortful struggle (very high beta + high theta)
  - concentration: deep focus (high beta, low theta, low alpha)

Scientific basis:
    Rehman et al. (2025) -- DDQN attention classification (98.2%)
    Springer (2024) -- Educational emotions from 5 EEG channels
    Pope et al. (1995) -- Beta/(alpha+theta) engagement index
    Berka et al. (2007) -- EEG engagement in educational contexts
"""

import numpy as np
from typing import Dict, List, Optional


ENGAGEMENT_STATES = ["attentive", "passive", "disengaged"]
EDUCATIONAL_EMOTIONS = ["boredom", "confusion", "curiosity", "frustration", "concentration"]


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


class EngagementDetector:
    """EEG-based student engagement and educational emotion detector.

    Uses frontal EEG markers:
    - Beta/theta ratio (primary attention index)
    - Alpha suppression (task engagement)
    - Frontal theta increase (mind-wandering)
    - Spectral entropy (cognitive complexity)

    References:
        Rehman et al. (2025) -- DDQN attention classification (98.2%)
        Springer (2024) -- Educational emotions from 5 EEG channels
    """

    def __init__(self, max_history: int = 300):
        self._history: Dict[str, List[Dict]] = {}
        self._baselines: Dict[str, Dict] = {}
        self._max_history = max_history

    def set_baseline(
        self,
        theta_power: float,
        alpha_power: float,
        beta_power: float,
        user_id: str = "default",
    ) -> None:
        """Record focused-state baseline (e.g., from counting task).

        Stores baseline beta/theta ratio and alpha level for
        baseline-relative engagement scoring.
        """
        bt_ratio = beta_power / max(theta_power, 1e-10)
        self._baselines[user_id] = {
            "theta": theta_power,
            "alpha": alpha_power,
            "beta": beta_power,
            "beta_theta_ratio": bt_ratio,
        }

    def assess(
        self,
        theta_power: float,
        alpha_power: float,
        beta_power: float,
        gamma_power: float = 0.0,
        user_id: str = "default",
    ) -> dict:
        """Assess engagement from band powers.

        Args:
            theta_power: 0-1 normalized theta (4-8 Hz)
            alpha_power: 0-1 normalized alpha (8-12 Hz)
            beta_power: 0-1 normalized beta (12-30 Hz)
            gamma_power: 0-1 normalized gamma (optional, used cautiously)
            user_id: User identifier

        Returns:
            dict with engagement_index, engagement_state,
            educational_emotion, emotion_scores, attention_index,
            mind_wandering_risk, n_samples.
        """
        # Clamp inputs to avoid negative values causing chaos
        theta = float(max(theta_power, 0.0))
        alpha = float(max(alpha_power, 0.0))
        beta = float(max(beta_power, 0.0))
        gamma = float(max(gamma_power, 0.0))

        # --- Engagement index ---
        beta_theta = beta / max(theta, 1e-10)
        alpha_suppression = 1.0 - min(alpha, 1.0)

        engagement_abs = (
            0.50 * float(np.clip(beta_theta / 3.0, 0.0, 1.0))
            + 0.30 * alpha_suppression
            + 0.20 * float(np.clip(beta, 0.0, 1.0))
        )
        engagement_abs = float(np.clip(engagement_abs, 0.0, 1.0))

        baseline = self._baselines.get(user_id)
        if baseline is not None:
            baseline_bt = baseline["beta_theta_ratio"]
            baseline_ratio = (beta_theta - baseline_bt) / max(baseline_bt, 1e-10)
            engagement_rel = float(np.clip(_sigmoid(baseline_ratio), 0.0, 1.0))
            engagement_index = 0.5 * engagement_abs + 0.5 * engagement_rel
        else:
            engagement_index = engagement_abs

        engagement_index = float(np.clip(engagement_index, 0.0, 1.0))

        # --- Engagement state ---
        if engagement_index > 0.6:
            engagement_state = "attentive"
        elif engagement_index >= 0.35:
            engagement_state = "passive"
        else:
            engagement_state = "disengaged"

        # --- Attention index (beta/theta normalized) ---
        attention_index = float(np.clip(beta_theta / 3.0, 0.0, 1.0))

        # --- Mind-wandering risk ---
        mind_wandering_risk = float(np.clip(
            0.50 * theta + 0.30 * alpha + 0.20 * (1.0 - min(beta, 1.0)),
            0.0,
            1.0,
        ))

        # --- Educational emotions ---
        emotion_scores = self._compute_educational_emotions(theta, alpha, beta)
        educational_emotion = max(emotion_scores, key=emotion_scores.get)

        # --- Store history ---
        if user_id not in self._history:
            self._history[user_id] = []

        entry = {
            "engagement_index": engagement_index,
            "engagement_state": engagement_state,
            "educational_emotion": educational_emotion,
        }
        self._history[user_id].append(entry)

        # Enforce cap
        if len(self._history[user_id]) > self._max_history:
            self._history[user_id] = self._history[user_id][-self._max_history:]

        n_samples = len(self._history[user_id])

        return {
            "engagement_index": round(engagement_index, 4),
            "engagement_state": engagement_state,
            "educational_emotion": educational_emotion,
            "emotion_scores": {k: round(v, 4) for k, v in emotion_scores.items()},
            "attention_index": round(attention_index, 4),
            "mind_wandering_risk": round(mind_wandering_risk, 4),
            "n_samples": n_samples,
        }

    def _compute_educational_emotions(
        self, theta: float, alpha: float, beta: float
    ) -> Dict[str, float]:
        """Compute scores for 5 educational emotions."""
        # Boredom: high alpha + low beta + low theta (understimulated)
        boredom = (
            0.40 * min(alpha, 1.0)
            + 0.35 * (1.0 - min(beta, 1.0))
            + 0.25 * (1.0 - min(theta, 1.0))
        )

        # Confusion: high theta + high beta (cognitive conflict)
        confusion = (
            0.45 * min(theta, 1.0)
            + 0.35 * min(beta, 1.0)
            + 0.20 * (1.0 - min(alpha, 1.0))
        )

        # Curiosity: moderate theta + moderate beta + low alpha
        # Peaks when theta and beta are both moderate (~0.4-0.6)
        # Use sharp peaked function: exp(-((x-0.5)/0.15)^2) -- Gaussian
        # centered at 0.5 with narrow width, drops sharply past 0.3/0.7
        theta_c = min(theta, 1.0)
        beta_c = min(beta, 1.0)
        theta_moderate = float(np.exp(-((theta_c - 0.5) / 0.2) ** 2))
        beta_moderate = float(np.exp(-((beta_c - 0.5) / 0.2) ** 2))
        curiosity = (
            0.35 * theta_moderate
            + 0.35 * beta_moderate
            + 0.30 * (1.0 - min(alpha, 1.0))
        )

        # Frustration: high beta + high theta + low alpha (effortful struggle)
        # Frustration > confusion when beta is higher
        frustration = (
            0.40 * min(beta, 1.0)
            + 0.35 * min(theta, 1.0)
            + 0.25 * (1.0 - min(alpha, 1.0))
        )

        # Concentration: high beta + low theta + low alpha
        concentration = (
            0.45 * min(beta, 1.0)
            + 0.30 * (1.0 - min(theta, 1.0))
            + 0.25 * (1.0 - min(alpha, 1.0))
        )

        scores = {
            "boredom": float(np.clip(boredom, 0.0, 1.0)),
            "confusion": float(np.clip(confusion, 0.0, 1.0)),
            "curiosity": float(np.clip(curiosity, 0.0, 1.0)),
            "frustration": float(np.clip(frustration, 0.0, 1.0)),
            "concentration": float(np.clip(concentration, 0.0, 1.0)),
        }
        return scores

    def get_engagement_curve(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> list:
        """Return engagement history for plotting.

        Each entry is a dict with engagement_index and engagement_state.
        """
        history = self._history.get(user_id, [])
        if last_n is not None:
            history = history[-last_n:]
        return [
            {
                "engagement_index": e["engagement_index"],
                "engagement_state": e["engagement_state"],
            }
            for e in history
        ]

    def get_session_summary(self, user_id: str = "default") -> dict:
        """Summary: mean engagement, time in each state, dominant emotion.

        Returns:
            dict with n_samples, mean_engagement, attentive_pct, passive_pct,
            disengaged_pct, dominant_emotion, mind_wandering_episodes.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {"n_samples": 0}

        n = len(history)
        engagements = [e["engagement_index"] for e in history]
        states = [e["engagement_state"] for e in history]
        emotions = [e["educational_emotion"] for e in history]

        attentive_count = sum(1 for s in states if s == "attentive")
        passive_count = sum(1 for s in states if s == "passive")
        disengaged_count = sum(1 for s in states if s == "disengaged")

        # Dominant emotion: most frequent
        emotion_counts: Dict[str, int] = {}
        for em in emotions:
            emotion_counts[em] = emotion_counts.get(em, 0) + 1
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        # Mind-wandering episodes: consecutive disengaged >= 3
        episodes = 0
        consecutive = 0
        for s in states:
            if s == "disengaged":
                consecutive += 1
                if consecutive == 3:
                    episodes += 1
                elif consecutive > 3:
                    # Each additional disengaged sample beyond 3 doesn't
                    # start a new episode; we count runs.
                    pass
            else:
                consecutive = 0

        return {
            "n_samples": n,
            "mean_engagement": round(float(np.mean(engagements)), 4),
            "attentive_pct": round(100.0 * attentive_count / n, 2),
            "passive_pct": round(100.0 * passive_count / n, 2),
            "disengaged_pct": round(100.0 * disengaged_count / n, 2),
            "dominant_emotion": dominant_emotion,
            "mind_wandering_episodes": episodes,
        }

    def reset(self, user_id: str = "default") -> None:
        """Clear history and baseline for a user."""
        if user_id in self._history:
            del self._history[user_id]
        if user_id in self._baselines:
            del self._baselines[user_id]
