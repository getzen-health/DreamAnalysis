"""RL training environment for adaptive neurofeedback threshold adjustment.

Wraps NeurofeedbackProtocol with a synthetic band-power simulator so the
PPO agent can train offline without real EEG hardware.

Each episode = 300 steps. User ability drifts upward (+0.002 per reward)
to simulate genuine learning over the session.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from neurofeedback.protocol_engine import NeurofeedbackProtocol, PROTOCOLS


# ── Synthetic band-power profiles ────────────────────────────────────────────
# For each protocol, define mean band powers for a "resting" user. The target
# band receives an extra boost proportional to _user_ability so the agent
# observes genuine learning progress when it sets a good difficulty level.
STATE_PROFILES: Dict[str, Dict[str, float]] = {
    "alpha_up": {
        "delta": 0.35,
        "theta": 0.20,
        "alpha": 0.25,
        "beta": 0.15,
        "gamma": 0.05,
    },
    "smr_up": {
        "delta": 0.30,
        "theta": 0.20,
        "alpha": 0.20,
        "beta": 0.25,   # SMR uses beta as proxy
        "gamma": 0.05,
    },
    "theta_beta_ratio": {
        "delta": 0.30,
        "theta": 0.30,
        "alpha": 0.20,
        "beta": 0.15,
        "gamma": 0.05,
    },
    "alpha_asymmetry": {
        "delta": 0.30,
        "theta": 0.20,
        "alpha": 0.30,
        "beta": 0.15,
        "gamma": 0.05,
    },
    "custom": {
        "delta": 0.35,
        "theta": 0.20,
        "alpha": 0.25,
        "beta": 0.15,
        "gamma": 0.05,
    },
}

# Which band is boosted by user ability (for each protocol)
_ABILITY_BAND: Dict[str, str] = {
    "alpha_up": "alpha",
    "smr_up": "beta",
    "theta_beta_ratio": "beta",   # higher beta → lower theta/beta ratio → reward
    "alpha_asymmetry": "alpha",
    "custom": "alpha",
}

EPISODE_LENGTH = 300
SIGMA = 0.03          # Gaussian noise std for simulated band powers
ABILITY_INCREMENT = 0.002   # per reward


class NeurofeedbackEnv:
    """Gym-style environment wrapping NeurofeedbackProtocol.

    Observation (8-dim float32):
        [avg_score/100, reward_rate_last10, streak/20, eval_progress,
         threshold/2.5, band_ratio/3.0, score_trend, score_volatility]

    Actions (discrete):
        0 → threshold -= 0.05  (easier)
        1 → hold
        2 → threshold += 0.05  (harder)

    Threshold clamped to [0.10, 2.50].

    Reward:
        (score / 100) + flow_bonus - stability_penalty
    """

    def __init__(self, protocol_type: str = "alpha_up"):
        if protocol_type not in PROTOCOLS:
            protocol_type = "alpha_up"
        self.protocol_type = protocol_type
        self._protocol: Optional[NeurofeedbackProtocol] = None
        self._user_ability: float = 0.0
        self._step_count: int = 0
        self._rng = np.random.default_rng()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset to a fresh episode. Returns initial 8-dim observation."""
        proto = PROTOCOLS[self.protocol_type]
        self._protocol = NeurofeedbackProtocol(
            protocol_type=self.protocol_type,
            threshold=proto["default_threshold"],
        )
        self._protocol.start()
        self._user_ability = 0.0
        self._step_count = 0
        return self._make_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply threshold action, simulate one EEG sample, return transition.

        Args:
            action: 0 (easier), 1 (hold), 2 (harder)

        Returns:
            (obs, reward, done, info)
        """
        assert self._protocol is not None, "Call reset() before step()"
        assert action in (0, 1, 2), f"Invalid action {action}"

        # --- Adjust threshold ---
        delta = (action - 1) * 0.05
        self._protocol.threshold = float(
            np.clip(self._protocol.threshold + delta, 0.10, 2.50)
        )

        # --- Simulate band powers ---
        band_powers = self._simulate_band_powers()

        # --- Evaluate protocol ---
        eval_result = self._protocol.evaluate(band_powers)
        score = eval_result["score"]
        got_reward = eval_result["reward"]

        # --- Update simulated user ability ---
        if got_reward:
            self._user_ability = min(1.0, self._user_ability + ABILITY_INCREMENT)

        # --- Compute RL reward ---
        rl_state = self._protocol.get_rl_state(band_powers)
        reward_rate = rl_state["reward_rate"]

        rl_reward = (score / 100.0)
        if 0.40 <= reward_rate <= 0.75:
            rl_reward += 0.20    # flow-zone bonus
        else:
            rl_reward -= 0.10   # outside flow zone
        if action != 1:
            rl_reward -= 0.02   # stability penalty for adjusting

        # --- Advance step counter ---
        self._step_count += 1
        done = self._step_count >= EPISODE_LENGTH

        obs = self._make_obs()
        info = {
            "score": score,
            "got_reward": got_reward,
            "threshold": self._protocol.threshold,
            "reward_rate": reward_rate,
            "user_ability": self._user_ability,
        }
        return obs, float(rl_reward), done, info

    # ── Private helpers ───────────────────────────────────────────────────────

    def _simulate_band_powers(self) -> Dict[str, float]:
        """Sample band powers from Gaussian distributions centred on the profile.

        The target band receives a boost of (user_ability * 0.30) so the agent
        observes genuine learning progress.
        """
        profile = STATE_PROFILES.get(self.protocol_type, STATE_PROFILES["alpha_up"])
        ability_band = _ABILITY_BAND.get(self.protocol_type, "alpha")

        powers: Dict[str, float] = {}
        for band, mean in profile.items():
            boost = self._user_ability * 0.30 if band == ability_band else 0.0
            val = float(self._rng.normal(mean + boost, SIGMA))
            powers[band] = max(0.001, val)

        # Normalise so they sum to 1.0
        total = sum(powers.values())
        return {b: v / total for b, v in powers.items()}

    def _make_obs(self) -> np.ndarray:
        """Build the 8-dim observation vector from current protocol state."""
        if self._protocol is None:
            return np.zeros(8, dtype=np.float32)

        band_powers = self._simulate_band_powers()
        rl_state = self._protocol.get_rl_state(band_powers)

        obs = np.array([
            rl_state["avg_score"] / 100.0,
            rl_state["reward_rate"],
            min(rl_state["streak"] / 20.0, 1.0),
            min(self._step_count / EPISODE_LENGTH, 1.0),
            min(rl_state["threshold"] / 2.5, 1.0),
            min(rl_state["band_ratio"] / 3.0, 1.0),
            rl_state["trend"],
            rl_state["volatility"],
        ], dtype=np.float32)
        return obs
