"""PPO-based adaptive agent for neurofeedback threshold control.

Implements a minimal Proximal Policy Optimisation agent from scratch using
PyTorch. No external RL libraries required — only PyTorch 2.4.1.

Architecture:
    Actor:  Linear(8→64) → Tanh → Linear(64→64) → Tanh → Linear(64→3) → Softmax
    Critic: Linear(8→64) → Tanh → Linear(64→64) → Tanh → Linear(64→1)

Usage:
    agent = PPOAgent()
    obs = env.reset()

    # Collect rollout
    for _ in range(rollout_length):
        action, log_prob = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        rollout.append(...)

    agent.update(rollout)
    agent.save("models/saved/rl_nf_agent.pt")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Hyper-parameters ─────────────────────────────────────────────────────────
OBS_DIM = 8
N_ACTIONS = 3
HIDDEN = 64
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
CLIP_EPS = 0.2
MINI_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95


# ── Neural network modules ────────────────────────────────────────────────────

def _mlp(in_dim: int, out_dim: int, hidden: int) -> "nn.Sequential":
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _mlp(OBS_DIM, N_ACTIONS, HIDDEN)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.softmax(self.net(x), dim=-1)


class _Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _mlp(OBS_DIM, 1, HIDDEN)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# ── Rollout container ─────────────────────────────────────────────────────────

class Rollout:
    """Collects a single rollout (episode or fixed-length buffer)."""

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.rewards)


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """Proximal Policy Optimisation agent controlling neurofeedback threshold.

    Actions:
        0 → decrease threshold by 0.05  (make easier)
        1 → hold threshold
        2 → increase threshold by 0.05  (make harder)
    """

    def __init__(self):
        self.is_trained: bool = False

        if not _TORCH_AVAILABLE:
            self._actor = None
            self._critic = None
            return

        self._actor = _Actor()
        self._critic = _Critic()
        self._opt_actor = optim.Adam(self._actor.parameters(), lr=LR_ACTOR)
        self._opt_critic = optim.Adam(self._critic.parameters(), lr=LR_CRITIC)

    # ── Inference ─────────────────────────────────────────────────────────────

    def act(self, obs: np.ndarray) -> Tuple[int, float]:
        """Sample an action from the policy.

        Args:
            obs: 8-dim float32 observation array.

        Returns:
            (action_int, log_prob_float). Returns (1, 0.0) if untrained or
            PyTorch is unavailable.
        """
        if not self.is_trained or self._actor is None:
            return 1, 0.0

        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0)
            probs = self._actor(t).squeeze(0)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def value(self, obs: np.ndarray) -> float:
        """Estimate state value (used during rollout collection)."""
        if self._critic is None:
            return 0.0
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0)
            v = self._critic(t)
        return float(v.item())

    # ── Training ──────────────────────────────────────────────────────────────

    def update(self, rollout: Rollout, last_obs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Run PPO update on a completed rollout.

        Args:
            rollout: Collected experience buffer.
            last_obs: Final observation (for bootstrap value). Use None for
                      terminal episodes.

        Returns:
            Dict with actor_loss and critic_loss for logging.
        """
        if self._actor is None or len(rollout) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        # --- Compute advantages via GAE ---
        advantages, returns = self._compute_gae(rollout, last_obs)

        # Normalise advantages
        adv_t = torch.FloatTensor(advantages)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        obs_t = torch.FloatTensor(np.stack(rollout.obs))
        acts_t = torch.LongTensor(rollout.actions)
        old_lp_t = torch.FloatTensor(rollout.log_probs)
        ret_t = torch.FloatTensor(returns)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(MINI_EPOCHS):
            # Actor loss (clipped surrogate)
            probs = self._actor(obs_t)
            dist = torch.distributions.Categorical(probs)
            new_lp = dist.log_prob(acts_t)
            ratio = torch.exp(new_lp - old_lp_t)

            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()

            self._opt_actor.zero_grad()
            actor_loss.backward()
            self._opt_actor.step()

            # Critic loss (MSE)
            values = self._critic(obs_t).squeeze(-1)
            critic_loss = nn.functional.mse_loss(values, ret_t)

            self._opt_critic.zero_grad()
            critic_loss.backward()
            self._opt_critic.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        self.is_trained = True
        return {
            "actor_loss": total_actor_loss / MINI_EPOCHS,
            "critic_loss": total_critic_loss / MINI_EPOCHS,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model weights and training flag to disk."""
        if self._actor is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self._actor.state_dict(),
                "critic": self._critic.state_dict(),
                "is_trained": self.is_trained,
            },
            path,
        )

    def load(self, path: str) -> bool:
        """Load model weights from disk. Returns True on success."""
        if self._actor is None:
            return False
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self._actor.load_state_dict(checkpoint["actor"])
            self._critic.load_state_dict(checkpoint["critic"])
            self.is_trained = checkpoint.get("is_trained", True)
            return True
        except Exception:
            return False

    # ── Observation builder ───────────────────────────────────────────────────

    @staticmethod
    def build_obs(rl_state: Dict) -> np.ndarray:
        """Convert get_rl_state() dict into the 8-dim numpy observation.

        All clipping / normalisation lives here, not in protocol_engine.

        Args:
            rl_state: Dict returned by NeurofeedbackProtocol.get_rl_state().

        Returns:
            float32 array of shape (8,).
        """
        obs = np.array([
            float(rl_state.get("avg_score", 0.0)) / 100.0,
            float(rl_state.get("reward_rate", 0.0)),
            min(float(rl_state.get("streak", 0)) / 20.0, 1.0),
            min(float(rl_state.get("total_evaluations", 0)) / 300.0, 1.0),
            min(float(rl_state.get("threshold", 0.5)) / 2.5, 1.0),
            min(float(rl_state.get("band_ratio", 1.0)) / 3.0, 1.0),
            float(np.clip(rl_state.get("trend", 0.0), -1.0, 1.0)),
            float(np.clip(rl_state.get("volatility", 0.0), 0.0, 1.0)),
        ], dtype=np.float32)
        return obs

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_gae(
        self, rollout: Rollout, last_obs: Optional[np.ndarray]
    ) -> Tuple[List[float], List[float]]:
        """Generalised Advantage Estimation."""
        last_value = self.value(last_obs) if last_obs is not None else 0.0

        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(rollout))):
            if rollout.dones[t]:
                next_value = 0.0
                gae = 0.0
            else:
                next_value = (
                    self.value(rollout.obs[t + 1])
                    if t + 1 < len(rollout)
                    else last_value
                )

            delta = rollout.rewards[t] + GAMMA * next_value - rollout.values[t]
            gae = delta + GAMMA * GAE_LAMBDA * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + rollout.values[t])

        return advantages, returns
