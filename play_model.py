from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from a2c_agent import ActorCritic
from env.config import EnvConfig
from env.make_env import make_env


def load_policy(model_path: Path, obs_shape, action_low, action_high, device):
    action_dim = int(np.prod(np.asarray(action_low).shape))
    policy = ActorCritic(int(np.prod(obs_shape)), action_dim, action_low=action_low, action_high=action_high).to(device)
    state = torch.load(model_path, map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def play(model_path: Path, episodes: int, render_mode: str):
    env_cfg = EnvConfig(render_mode=render_mode)
    env = make_env(env_cfg, seed=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(
        model_path,
        env.observation_space.shape,
        env.action_space.low.reshape(-1),
        env.action_space.high.reshape(-1),
        device,
    )

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = policy.deterministic_action(obs_t).squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            print(f"Episode {ep + 1}: reward={total_reward:.2f}")
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Play a saved A2C policy.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to saved model (best_model.pt).")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play.")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Render mode for the environment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    play(args.model_path, args.episodes, args.render_mode)


if __name__ == "__main__":
    main()
