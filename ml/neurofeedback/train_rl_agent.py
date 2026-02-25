"""Offline training script for the adaptive neurofeedback PPO agent.

Trains against the synthetic NeurofeedbackEnv simulator and saves
the resulting model to ml/models/saved/rl_nf_agent.pt.

Usage:
    cd ml
    python neurofeedback/train_rl_agent.py
    python neurofeedback/train_rl_agent.py --episodes 50 --protocol alpha_up
    python neurofeedback/train_rl_agent.py --episodes 500 --out models/saved/rl_nf_agent.pt
"""

import argparse
import sys
from pathlib import Path

# Allow running from the ml/ directory directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from neurofeedback.rl_environment import NeurofeedbackEnv
from neurofeedback.adaptive_agent import PPOAgent, Rollout
from neurofeedback.protocol_engine import PROTOCOLS

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "models" / "saved" / "rl_nf_agent.pt"
ROLLOUT_LEN = 300   # one full episode per update


def train(
    protocol_type: str,
    n_episodes: int,
    out_path: Path,
    verbose: bool = True,
) -> PPOAgent:
    """Train a PPO agent on the given protocol.

    Args:
        protocol_type: One of the PROTOCOLS keys.
        n_episodes: Number of training episodes.
        out_path: Where to save the final model.
        verbose: Print episode returns.

    Returns:
        Trained PPOAgent.
    """
    env = NeurofeedbackEnv(protocol_type=protocol_type)
    agent = PPOAgent()

    if verbose:
        print(f"[train] protocol={protocol_type}  episodes={n_episodes}")

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        rollout = Rollout()
        ep_return = 0.0
        done = False

        while not done:
            # Sample action + collect value estimate
            action, log_prob = agent.act(obs)
            value = agent.value(obs)

            next_obs, reward, done, _ = env.step(action)

            rollout.add(obs, action, log_prob, reward, done, value)
            ep_return += reward
            obs = next_obs

        # PPO update at end of episode
        last_obs = None if done else obs
        losses = agent.update(rollout, last_obs=last_obs)

        if verbose and (ep % 50 == 0 or ep == 1):
            print(
                f"  ep {ep:4d}/{n_episodes} | return={ep_return:7.2f} "
                f"| actor_loss={losses['actor_loss']:.4f} "
                f"| critic_loss={losses['critic_loss']:.4f}"
            )

    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Train the adaptive neurofeedback PPO agent offline."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes per protocol (default: 500)",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="all",
        choices=list(PROTOCOLS.keys()) + ["all"],
        help="Protocol to train on, or 'all' to train on all protocols (default: all)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help=f"Output path for the saved model (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    out_path = Path(args.out)

    if args.protocol == "all":
        protocols_to_train = ["alpha_up", "smr_up", "theta_beta_ratio"]
    else:
        protocols_to_train = [args.protocol]

    # Train on each protocol; the agent is shared across protocols so it
    # learns a general policy that transfers across training types.
    agent = PPOAgent()
    for proto in protocols_to_train:
        env = NeurofeedbackEnv(protocol_type=proto)
        print(f"\n=== Training on protocol: {proto} ===")

        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            rollout = Rollout()
            ep_return = 0.0
            done = False

            while not done:
                action, log_prob = agent.act(obs)
                value = agent.value(obs)
                next_obs, reward, done, _ = env.step(action)
                rollout.add(obs, action, log_prob, reward, done, value)
                ep_return += reward
                obs = next_obs

            last_obs = None if done else obs
            losses = agent.update(rollout, last_obs=last_obs)

            if ep % 50 == 0 or ep == 1:
                print(
                    f"  ep {ep:4d}/{args.episodes} | return={ep_return:7.2f} "
                    f"| actor_loss={losses['actor_loss']:.4f} "
                    f"| critic_loss={losses['critic_loss']:.4f}"
                )

    agent.save(str(out_path))
    print(f"\n[done] Model saved to {out_path}")
    print(f"       is_trained = {agent.is_trained}")


if __name__ == "__main__":
    main()
