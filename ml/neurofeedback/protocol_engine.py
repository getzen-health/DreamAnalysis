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
        "evidence_grade": "A",
        "evidence_references": [
            "Gruzelier JH (2014). EEG-neurofeedback for optimising performance. Neurosci Biobehav Rev, 44, 279-304.",
            "Zoefel B et al. (2011). Neurofeedback training of the upper alpha frequency band in EEG improves cognitive performance. Clin Neurophysiol, 122(11), 2220-2227.",
        ],
    },
    "smr_up": {
        "name": "SMR Training",
        "description": "Increase sensorimotor rhythm (12-15 Hz) for focus training",
        "target_band": "smr",
        "direction": "increase",
        "default_threshold": 0.3,
        "evidence_grade": "A",
        "evidence_references": [
            "Arns M et al. (2009). Efficacy of neurofeedback treatment in ADHD. Clin EEG Neurosci, 40(3), 180-189.",
            "Enriquez-Geppert S et al. (2019). Neurofeedback as a treatment intervention in ADHD. Curr Psychiatry Rep, 21(6), 46.",
        ],
    },
    "theta_beta_ratio": {
        "name": "Theta/Beta Ratio",
        "description": "Decrease theta/beta ratio for attention (ADHD protocol)",
        "target_band": "theta_beta",
        "direction": "decrease",
        "default_threshold": 2.5,
        "evidence_grade": "A",
        "evidence_references": [
            "Arns M et al. (2009). Efficacy of neurofeedback treatment in ADHD. Clin EEG Neurosci, 40(3), 180-189.",
            "Lubar JF (1995). Neurofeedback for the management of attention-deficit/hyperactivity disorders. Biofeedback Self Regul, 20(2), 111-127.",
        ],
    },
    "alpha_asymmetry": {
        "name": "Alpha Asymmetry",
        "description": "Balance left/right alpha power for mood regulation",
        "target_band": "alpha",
        "direction": "balance",
        "default_threshold": 0.1,
        "evidence_grade": "B",
        "evidence_references": [
            "Baehr E et al. (2001). Clinical use of an alpha asymmetry neurofeedback protocol in the treatment of mood disorders. J Neurother, 4(4), 11-18.",
            "Peeters F et al. (2014). Oxytocin and neurofeedback for depression-related frontal alpha asymmetry. Psychiatry Res, 223(2), 180-184.",
        ],
    },
    "custom": {
        "name": "Custom Protocol",
        "description": "User-defined band and threshold",
        "target_band": "alpha",
        "direction": "increase",
        "default_threshold": 0.5,
        "evidence_grade": "N/A",
        "evidence_references": [],
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

    def get_rl_state(self, band_powers: Dict[str, float]) -> Dict:
        """Build the RL state dict used by PPOAgent.build_obs().

        Called by the route *after* evaluate(), so history already includes
        the latest step.

        Returns:
            Dict with keys: avg_score, reward_rate, streak, total_evaluations,
            threshold, band_ratio, trend, volatility.
        """
        recent = list(self.history)[-10:] if self.history else []
        scores = [h["score"] for h in recent]

        avg_score = float(np.mean(scores)) if scores else 0.0
        reward_rate = float(sum(1 for h in recent if h["reward"]) / max(len(recent), 1))

        # Linear slope of last-10 scores, normalised by 10
        if len(scores) >= 2:
            x = np.arange(len(scores), dtype=float)
            trend = float(np.polyfit(x, scores, 1)[0] / 10.0)
        else:
            trend = 0.0

        volatility = float(np.std(scores) / 100.0) if len(scores) >= 2 else 0.0

        # band_ratio: target band value relative to baseline
        target_value = self._extract_target_value(band_powers)
        baseline = self.baseline or self.threshold
        band_ratio = float(target_value / max(baseline, 1e-10))

        return {
            "avg_score": avg_score,
            "reward_rate": reward_rate,
            "streak": self.streak,
            "total_evaluations": self.total_evaluations,
            "threshold": self.threshold,
            "band_ratio": band_ratio,
            "trend": trend,
            "volatility": volatility,
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
