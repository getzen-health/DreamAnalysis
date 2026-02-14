"""Neurofeedback Protocol Engine.

Provides real-time neurofeedback evaluation with multiple training
protocols (alpha up-training, SMR, theta/beta ratio, alpha asymmetry).
Supports baseline calibration and session statistics.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque


PROTOCOLS = {
    "alpha_up": {
        "name": "Alpha Enhancement",
        "description": "Increase alpha power (8-12 Hz) for relaxation training",
        "target_band": "alpha",
        "direction": "increase",
        "default_threshold": 0.5,
    },
    "smr_up": {
        "name": "SMR Training",
        "description": "Increase sensorimotor rhythm (12-15 Hz) for focus training",
        "target_band": "smr",
        "direction": "increase",
        "default_threshold": 0.3,
    },
    "theta_beta_ratio": {
        "name": "Theta/Beta Ratio",
        "description": "Decrease theta/beta ratio for attention (ADHD protocol)",
        "target_band": "theta_beta",
        "direction": "decrease",
        "default_threshold": 2.5,
    },
    "alpha_asymmetry": {
        "name": "Alpha Asymmetry",
        "description": "Balance left/right alpha power for mood regulation",
        "target_band": "alpha",
        "direction": "balance",
        "default_threshold": 0.1,
    },
    "custom": {
        "name": "Custom Protocol",
        "description": "User-defined band and threshold",
        "target_band": "alpha",
        "direction": "increase",
        "default_threshold": 0.5,
    },
}


class NeurofeedbackProtocol:
    """Real-time neurofeedback evaluation engine.

    Evaluates incoming EEG band powers against a protocol's target,
    tracks rewards, streaks, and session statistics.
    """

    def __init__(
        self,
        protocol_type: str = "alpha_up",
        target_band: Optional[str] = None,
        threshold: Optional[float] = None,
        reward_type: str = "visual",
    ):
        if protocol_type not in PROTOCOLS:
            protocol_type = "alpha_up"

        proto = PROTOCOLS[protocol_type]
        self.protocol_type = protocol_type
        self.target_band = target_band or proto["target_band"]
        self.direction = proto["direction"]
        self.threshold = threshold or proto["default_threshold"]
        self.reward_type = reward_type

        # State
        self.baseline: Optional[float] = None
        self.baseline_samples: List[float] = []
        self.is_calibrating = False
        self.is_active = False

        # Rolling history (last 60 evaluations)
        self.history = deque(maxlen=60)
        self.streak = 0
        self.total_rewards = 0
        self.total_evaluations = 0

    def start_calibration(self):
        """Begin baseline calibration period."""
        self.is_calibrating = True
        self.baseline_samples = []

    def add_calibration_sample(self, band_powers: Dict[str, float]) -> bool:
        """Add a sample during calibration. Returns True when calibration is complete."""
        value = self._extract_target_value(band_powers)
        self.baseline_samples.append(value)
        # Calibration complete after 30 samples (~30 seconds at 1Hz eval rate)
        if len(self.baseline_samples) >= 30:
            self.baseline = float(np.mean(self.baseline_samples))
            self.is_calibrating = False
            self.is_active = True
            return True
        return False

    def start(self, baseline: Optional[float] = None):
        """Start the neurofeedback session with optional pre-set baseline."""
        if baseline is not None:
            self.baseline = baseline
        elif self.baseline is None:
            # Use protocol default as baseline
            self.baseline = self.threshold
        self.is_active = True
        self.streak = 0
        self.total_rewards = 0
        self.total_evaluations = 0
        self.history.clear()

    def stop(self) -> Dict:
        """Stop the session and return final statistics."""
        self.is_active = False
        self.is_calibrating = False
        return self.get_session_stats()

    def evaluate(
        self, band_powers: Dict[str, float], channel_powers: Optional[List[Dict]] = None
    ) -> Dict:
        """Evaluate current EEG against protocol target.

        Args:
            band_powers: Dict with band power values (delta, theta, alpha, beta, gamma).
            channel_powers: Optional per-channel band powers for asymmetry protocols.

        Returns:
            Dict with score, reward, feedback_value, streak.
        """
        if not self.is_active:
            return {"score": 0.0, "reward": False, "feedback_value": 0.0, "streak": 0}

        value = self._extract_target_value(band_powers, channel_powers)
        baseline = self.baseline or self.threshold

        # Compute score (0-100) based on direction
        if self.direction == "increase":
            # Score increases as value exceeds baseline
            ratio = value / max(baseline, 1e-10)
            score = min(100.0, max(0.0, ratio * 50.0))
            reward = value > baseline * (1.0 + self.threshold * 0.1)
        elif self.direction == "decrease":
            # Score increases as value falls below baseline
            ratio = baseline / max(value, 1e-10)
            score = min(100.0, max(0.0, ratio * 50.0))
            reward = value < baseline * (1.0 - self.threshold * 0.1)
        elif self.direction == "balance":
            # Score increases as value approaches zero (asymmetry)
            asymmetry = abs(value)
            score = min(100.0, max(0.0, (1.0 - asymmetry) * 100.0))
            reward = asymmetry < self.threshold
        else:
            score = 50.0
            reward = False

        # Update streak
        if reward:
            self.streak += 1
            self.total_rewards += 1
        else:
            self.streak = 0

        self.total_evaluations += 1

        # Feedback value normalized 0-1 for visual/audio feedback
        feedback_value = float(score / 100.0)

        result = {
            "score": float(score),
            "reward": bool(reward),
            "feedback_value": feedback_value,
            "streak": self.streak,
        }

        self.history.append(result)
        return result

    def get_session_stats(self) -> Dict:
        """Get cumulative session statistics."""
        if self.total_evaluations == 0:
            return {
                "total_rewards": 0,
                "reward_rate": 0.0,
                "avg_score": 0.0,
                "time_above_threshold": 0.0,
                "max_streak": 0,
                "total_evaluations": 0,
            }

        scores = [h["score"] for h in self.history]
        max_streak = 0
        current = 0
        for h in self.history:
            if h["reward"]:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0

        return {
            "total_rewards": self.total_rewards,
            "reward_rate": float(self.total_rewards / max(self.total_evaluations, 1)),
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "time_above_threshold": float(self.total_rewards / max(self.total_evaluations, 1)),
            "max_streak": max_streak,
            "total_evaluations": self.total_evaluations,
        }

    def _extract_target_value(
        self, band_powers: Dict[str, float], channel_powers: Optional[List[Dict]] = None
    ) -> float:
        """Extract the target metric value from band powers."""
        if self.target_band == "theta_beta":
            theta = band_powers.get("theta", 0.0)
            beta = band_powers.get("beta", 0.001)
            return theta / beta

        if self.target_band == "smr":
            # SMR is approximately low-beta (12-15 Hz), use beta as proxy
            return band_powers.get("beta", 0.0) * 0.4

        if self.direction == "balance" and channel_powers and len(channel_powers) >= 2:
            # Alpha asymmetry: left alpha - right alpha
            left_alpha = channel_powers[0].get("alpha", 0.0)
            right_alpha = channel_powers[1].get("alpha", 0.0)
            total = left_alpha + right_alpha + 1e-10
            return (right_alpha - left_alpha) / total

        return band_powers.get(self.target_band, 0.0)
