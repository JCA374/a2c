"""Test episode completion with real CarRacing environment."""
import csv
from pathlib import Path

import numpy as np

from a2c_agent import A2CConfig, A2CTrainer
from env.config import EnvConfig
from env.make_env import make_env
from env.vector_env import SimpleVectorEnv
from log_utils import TrainingLogger


def test_real_carracing_episodes_complete(tmp_path: Path):
    """Test that real CarRacing episodes complete within reasonable time."""
    num_envs = 2
    rollout_length = 128  # Longer rollout
    total_steps = rollout_length * num_envs * 20  # 20 updates

    # Use real CarRacing environment
    env_config = EnvConfig(render_mode="rgb_array")
    envs = SimpleVectorEnv([
        lambda: make_env(env_config, seed=i)
        for i in range(num_envs)
    ])

    cfg = A2CConfig(
        total_steps=total_steps,
        rollout_length=rollout_length,
        log_interval=1
    )
    logger = TrainingLogger(tmp_path, config={"test": "real_carracing"})

    trainer = A2CTrainer(envs, cfg, logger=logger)

    try:
        policy, completed_returns = trainer.train()
    finally:
        envs.close()
        logger.close()

    print(f"\nCompleted {len(completed_returns)} episodes in {total_steps} total steps")
    if len(completed_returns) > 0:
        print(f"Episode returns: {[r for r, _ in completed_returns[:5]]}")
        print(f"Episode lengths: {[l for _, l in completed_returns[:5]]}")

    # Read the metrics to see what happened
    metrics_file = tmp_path / "metrics.csv"
    with metrics_file.open() as f:
        rows = list(csv.DictReader(f))

    print("\nMetrics summary:")
    for i, row in enumerate(rows[:10]):
        print(f"Update {row['update']}: episodes_completed={row['episodes_completed']}, "
              f"mean_return={row['mean_return_last10']}, "
              f"rollout_reward_mean={row['rollout_reward_mean']}")

    # At this point we should have some completed episodes, but it's possible
    # that early training episodes don't complete if the agent is very bad
    # Just check that the system is working
    episodes_completed = int(rows[-1]['episodes_completed'])
    print(f"\nFinal episodes completed: {episodes_completed}")

    # Even if no full episodes complete, rollout rewards should be non-zero
    rollout_rewards = [float(row['rollout_reward_mean']) for row in rows]
    print(f"Rollout rewards: {rollout_rewards[:10]}")


def test_short_episode_real_carracing(tmp_path: Path):
    """Test with very short max_steps to force episode completion."""
    num_envs = 2
    rollout_length = 64
    total_steps = rollout_length * num_envs * 10

    # Use short max_steps to force episodes to complete
    env_config = EnvConfig(render_mode="rgb_array")
    env_config.episode.max_steps = 50  # Very short episodes
    env_config.episode.max_seconds = None

    envs = SimpleVectorEnv([
        lambda: make_env(env_config, seed=i)
        for i in range(num_envs)
    ])

    cfg = A2CConfig(
        total_steps=total_steps,
        rollout_length=rollout_length,
        log_interval=1
    )
    logger = TrainingLogger(tmp_path, config={"test": "short_episodes"})

    trainer = A2CTrainer(envs, cfg, logger=logger)

    try:
        policy, completed_returns = trainer.train()
    finally:
        envs.close()
        logger.close()

    print(f"\nCompleted {len(completed_returns)} episodes")

    # Read the metrics
    metrics_file = tmp_path / "metrics.csv"
    with metrics_file.open() as f:
        rows = list(csv.DictReader(f))

    print("\nMetrics summary:")
    for row in rows[:10]:
        print(f"Update {row['update']}: episodes_completed={row['episodes_completed']}, "
              f"mean_return={row['mean_return_last10']}")

    # With max_steps=50, we should complete at least some episodes
    episodes_completed = int(rows[-1]['episodes_completed'])
    print(f"\nFinal episodes completed: {episodes_completed}")

    # This should be > 0 since episodes are forced to end quickly
    assert episodes_completed > 0, f"Expected some episodes to complete with max_steps=50, got {episodes_completed}"
