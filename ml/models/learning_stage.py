"""Learning Stage Classifier from EEG band-power shifts during skill acquisition.

Classifies the cognitive learning stage based on how theta, alpha, and gamma
powers change as a learner progresses from novice to expert on a task.

Four stages (Fitts & Posner 1967, mapped to EEG signatures):

  encoding      — High theta, low alpha.  Effortful declarative learning.
                  The brain is actively encoding new information (hippocampal
                  theta loops).  Performance is low.

  consolidation — Moderate theta, moderate alpha.  Practice phase.
                  Procedural memory is forming; cortical efficiency begins
                  to increase.  Performance is moderate.

  automation    — High alpha, low theta.  Neural efficiency kicks in
                  (Neubauer & Fink 2009).  The task requires less conscious
                  effort; alpha power rises as cortical idling increases.

  mastery       — High alpha + theta-gamma coupling + high performance.
                  Expert integration: the brain efficiently orchestrates
                  fast gamma bursts within theta/alpha scaffolding
                  (Canolty & Knight 2010).

Scientific basis:
  - Neural efficiency hypothesis (Neubauer & Fink, 2009)
  - Theta-gamma coupling in learning (Lisman & Jensen, 2013)
  - Skill acquisition EEG shifts (Babiloni et al., 2010)
  - Fitts & Posner (1967) stages of motor learning

Usage:
    clf = LearningStageClassifier()
    result = clf.classify(theta_power=0.35, alpha_power=0.15)
    # result["stage"] == "encoding"

    clf.record_session(theta_power=0.35, alpha_power=0.15, performance=0.3)
    progression = clf.get_progression()
"""

import numpy as np
from typing import Dict, List, Optional


LEARNING_STAGES = ["encoding", "consolidation", "automation", "mastery"]

_STAGE_RECOMMENDATIONS = {
    "encoding": (
        "Focus on understanding fundamentals. Use active recall and "
        "spaced repetition. High theta indicates your brain is actively "
        "encoding — take breaks every 25 minutes to consolidate."
    ),
    "consolidation": (
        "You are in the practice phase. Increase repetitions and vary "
        "examples. Your brain is building procedural pathways — deliberate "
        "practice with feedback will accelerate this stage."
    ),
    "automation": (
        "Neural efficiency is emerging. Challenge yourself with harder "
        "variations to push toward mastery. Your brain is becoming more "
        "efficient — alpha dominance means less conscious effort is needed."
    ),
    "mastery": (
        "Expert-level integration detected. Maintain skill with periodic "
        "practice and teach others to deepen understanding. Theta-gamma "
        "coupling indicates sophisticated neural coordination."
    ),
}


class LearningStageClassifier:
    """Classify cognitive learning stage from EEG band powers.

    Maintains per-user, per-task session history for tracking progression
    across multiple learning sessions.

    Thread safety: not thread-safe. Use one instance per worker or wrap
    calls with a lock in production.
    """

    def __init__(self) -> None:
        # _history[user_id][task_type] = list of session dicts
        self._history: Dict[str, Dict[str, List[Dict]]] = {}

    def classify(
        self,
        theta_power: float,
        alpha_power: float,
        gamma_power: float = 0.0,
        beta_power: float = 0.0,
        session_count: int = 1,
        task_performance: float = 0.5,
    ) -> Dict:
        """Classify the current learning stage from EEG band powers.

        Args:
            theta_power: Theta band power (4-8 Hz). Higher during effortful
                         encoding and memory consolidation.
            alpha_power: Alpha band power (8-12 Hz). Higher during neural
                         efficiency / automation.
            gamma_power: Gamma band power (30-100 Hz). Presence alongside
                         high alpha suggests expert theta-gamma coupling.
                         Default 0.0 (not always available).
            beta_power:  Beta band power (12-30 Hz). Used as a secondary
                         signal for engagement. Default 0.0.
            session_count: Number of prior sessions on this task. Used to
                           modulate stage expectations (more sessions make
                           advanced stages more likely).
            task_performance: Task performance metric in [0, 1]. 0 = worst,
                              1 = best. Used to disambiguate stages.

        Returns:
            Dict with keys:
              - stage: one of LEARNING_STAGES
              - stage_scores: dict of scores for each stage
              - confidence: 0-1 classification confidence
              - theta_alpha_ratio: raw theta/alpha ratio
              - recommendation: stage-appropriate learning advice
        """
        # Clamp performance to [0, 1]
        perf = float(np.clip(task_performance, 0.0, 1.0))

        # Guard against division by zero
        total = theta_power + alpha_power
        if total < 1e-10:
            theta_norm = 0.5
            alpha_norm = 0.5
        else:
            theta_norm = theta_power / total
            alpha_norm = alpha_power / total

        # Theta/alpha ratio (guarded)
        if alpha_power < 1e-10:
            theta_alpha_ratio = 10.0 if theta_power > 1e-10 else 1.0
        else:
            theta_alpha_ratio = theta_power / alpha_power

        # Gamma presence signal (normalized, capped at 1.0)
        gamma_signal = 0.0
        if gamma_power > 0 and total > 1e-10:
            gamma_signal = float(np.clip(gamma_power / total, 0.0, 1.0))

        # Session experience factor: log-scaled, saturates around 20 sessions
        experience = float(np.clip(np.log1p(session_count) / np.log1p(20), 0.0, 1.0))

        # ── Stage scoring ──────────────────────────────────────────────

        # Encoding: high theta, low alpha, low performance
        encoding_score = (
            0.40 * theta_norm
            + 0.30 * (1.0 - alpha_norm)
            + 0.30 * (1.0 - perf)
        )

        # Consolidation: balanced theta/alpha, moderate performance
        # Peaks when theta ≈ alpha (balance = 1 - |theta_norm - alpha_norm|)
        balance = 1.0 - abs(theta_norm - alpha_norm)
        consolidation_score = (
            0.35 * balance
            + 0.35 * perf
            + 0.30 * (1.0 - abs(perf - 0.5) * 2.0)  # peaks at perf=0.5
        )

        # Automation: high alpha, low theta, good performance
        automation_score = (
            0.40 * alpha_norm
            + 0.30 * (1.0 - theta_norm)
            + 0.30 * perf
        )

        # Mastery: high alpha + gamma presence + high performance + experience
        # Gamma is the key differentiator from automation — weight it heavily
        mastery_score = (
            0.25 * alpha_norm
            + 0.30 * gamma_signal
            + 0.25 * perf
            + 0.20 * experience
        )

        # Normalize scores to sum to 1
        scores_raw = {
            "encoding": float(encoding_score),
            "consolidation": float(consolidation_score),
            "automation": float(automation_score),
            "mastery": float(mastery_score),
        }

        score_sum = sum(scores_raw.values())
        if score_sum < 1e-10:
            stage_scores = {s: 0.25 for s in LEARNING_STAGES}
        else:
            stage_scores = {s: round(v / score_sum, 4) for s, v in scores_raw.items()}

        # Winner
        stage = max(stage_scores, key=lambda s: stage_scores[s])

        # Confidence: how much the winner stands out from the rest
        sorted_scores = sorted(stage_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            confidence = float(np.clip(sorted_scores[0] - sorted_scores[1], 0.0, 1.0))
        else:
            confidence = 1.0

        return {
            "stage": stage,
            "stage_scores": stage_scores,
            "confidence": round(confidence, 4),
            "theta_alpha_ratio": round(float(theta_alpha_ratio), 4),
            "recommendation": _STAGE_RECOMMENDATIONS[stage],
        }

    def record_session(
        self,
        theta_power: float,
        alpha_power: float,
        gamma_power: float = 0.0,
        performance: float = 0.5,
        user_id: str = "default",
        task_type: str = "default",
    ) -> Dict:
        """Record a learning session for progression tracking.

        Args:
            theta_power: Theta band power during the session.
            alpha_power: Alpha band power during the session.
            gamma_power: Gamma band power during the session.
            performance: Task performance metric in [0, 1].
            user_id: Unique user identifier.
            task_type: Task category (e.g. "piano", "math", "coding").

        Returns:
            Dict with session_number and stage classification for this session.
        """
        if user_id not in self._history:
            self._history[user_id] = {}
        if task_type not in self._history[user_id]:
            self._history[user_id][task_type] = []

        sessions = self._history[user_id][task_type]
        session_count = len(sessions) + 1

        classification = self.classify(
            theta_power=theta_power,
            alpha_power=alpha_power,
            gamma_power=gamma_power,
            session_count=session_count,
            task_performance=performance,
        )

        session_data = {
            "theta_power": float(theta_power),
            "alpha_power": float(alpha_power),
            "gamma_power": float(gamma_power),
            "performance": float(np.clip(performance, 0.0, 1.0)),
            "stage": classification["stage"],
        }
        sessions.append(session_data)

        return {
            "session_number": session_count,
            "stage": classification["stage"],
            "stage_scores": classification["stage_scores"],
            "confidence": classification["confidence"],
        }

    def get_progression(
        self,
        user_id: str = "default",
        task_type: str = "default",
    ) -> Dict:
        """Return learning progression history for a user and task.

        Args:
            user_id: Unique user identifier.
            task_type: Task category.

        Returns:
            Dict with sessions (list of session dicts), current_stage,
            sessions_completed, and stage_history (list of stage strings).
        """
        sessions = list(
            self._history.get(user_id, {}).get(task_type, [])
        )

        if not sessions:
            return {
                "sessions": [],
                "current_stage": None,
                "sessions_completed": 0,
                "stage_history": [],
            }

        stage_history = [s["stage"] for s in sessions]

        return {
            "sessions": sessions,
            "current_stage": stage_history[-1],
            "sessions_completed": len(sessions),
            "stage_history": stage_history,
        }

    def reset(self, user_id: str = "default") -> None:
        """Clear all session history for a user (all task types).

        Args:
            user_id: Unique user identifier.
        """
        if user_id in self._history:
            del self._history[user_id]
