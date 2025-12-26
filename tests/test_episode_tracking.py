"""Tests specifically for episode completion tracking."""
from pathlib import Path

import numpy as np
import torch

from a2c_agent import A2CConfig, A2CTrainer
from env.vector_env import SimpleVectorEnv
from log_utils import TrainingLogger
from tests.dummy_envs import DummyCarRacing


def test_vector_env_returns_done_flags():
    """Test that SimpleVectorEnv properly returns terminated/truncated flags."""
    num_envs = 3
    max_steps = 5
    envs = SimpleVectorEnv([lambda: DummyCarRacing(max_steps=max_steps) for _ in range(num_envs)])

    obs, _ = envs.reset()
    assert obs.shape == (num_envs, 4)

    # Step until episodes should complete
    episodes_completed = 0
    for step in range(max_steps + 2):
        actions = np.zeros((num_envs, 2), dtype=np.float32)
        obs, rewards, terminated, truncated, infos = envs.step(actions)
        dones = np.logical_or(terminated, truncated)

        print(f"Step {step}: terminated={terminated}, truncated={truncated}, dones={dones}")

        if step == max_steps - 1:
            # All episodes should complete on this step
            assert dones.all(), f"Expected all envs to be done at step {step}, got dones={dones}"
            episodes_completed += dones.sum()

    envs.close()
    assert episodes_completed == num_envs, f"Expected {num_envs} episodes to complete, got {episodes_completed}"


def test_trainer_tracks_episode_completions(tmp_path: Path):
    """Test that A2CTrainer properly tracks completed episodes."""
    num_envs = 2
    max_steps = 10  # Short episodes
    rollout_length = 5  # Shorter than episode length

    # We want at least a few full episodes to complete
    total_steps = max_steps * num_envs * 3  # Expect ~3 episodes per env

    envs = SimpleVectorEnv([lambda: DummyCarRacing(max_steps=max_steps) for _ in range(num_envs)])
    cfg = A2CConfig(total_steps=total_steps, rollout_length=rollout_length, log_interval=1)
    logger = TrainingLogger(tmp_path, config={"test": "episode_tracking"})

    trainer = A2CTrainer(envs, cfg, logger=logger)

    try:
        policy, completed_returns = trainer.train()
    finally:
        envs.close()
        logger.close()

    print(f"\nCompleted {len(completed_returns)} episodes")
    print(f"Returns: {[r for r, _ in completed_returns[:10]]}")
    print(f"Lengths: {[l for _, l in completed_returns[:10]]}")

    # We should have completed at least a few episodes
    assert len(completed_returns) > 0, "No episodes completed!"

    # Check that episode lengths make sense (should be around max_steps)
    lengths = [l for _, l in completed_returns]
    assert all(l > 0 for l in lengths), f"Some episode lengths are 0: {lengths}"
    assert all(l <= max_steps for l in lengths), f"Some episodes longer than max_steps: {lengths}"


def test_trainer_mean_return_not_zero(tmp_path: Path):
    """Test that mean_return_last10 is not zero when episodes complete."""
    import csv

    num_envs = 2
    max_steps = 8  # Short episodes so they complete quickly
    rollout_length = 4
    total_steps = max_steps * num_envs * 5  # Ensure multiple episodes complete

    envs = SimpleVectorEnv([lambda: DummyCarRacing(max_steps=max_steps) for _ in range(num_envs)])
    cfg = A2CConfig(
        total_steps=total_steps,
        rollout_length=rollout_length,
        log_interval=1  # Log every update
    )
    logger = TrainingLogger(tmp_path, config={"test": "mean_return_check"})

    trainer = A2CTrainer(envs, cfg, logger=logger)

    try:
        trainer.train()
    finally:
        envs.close()
        logger.close()

    # Read metrics and check that mean_return_last10 becomes non-zero
    metrics_file = tmp_path / "metrics.csv"
    assert metrics_file.exists()

    with metrics_file.open() as f:
        rows = list(csv.DictReader(f))

    print("\nMetrics rows:")
    for row in rows:
        print(f"Update {row['update']}: episodes_completed={row['episodes_completed']}, "
              f"mean_return_last10={row['mean_return_last10']}, "
              f"mean_length_last10={row['mean_length_last10']}")

    # Check that episodes_completed increases
    final_episodes = int(rows[-1]['episodes_completed'])
    assert final_episodes > 0, f"No episodes completed! Final count: {final_episodes}"

    # Once we have episodes, mean_return should be non-zero (DummyCarRacing gives positive rewards)
    mean_returns = [float(row['mean_return_last10']) for row in rows if int(row['episodes_completed']) > 0]
    assert len(mean_returns) > 0, "No metrics with completed episodes"

    # At least some mean returns should be non-zero
    non_zero_returns = [r for r in mean_returns if abs(r) > 1e-6]
    assert len(non_zero_returns) > 0, f"All mean returns are zero: {mean_returns}"


def test_episode_returns_accumulate_correctly():
    """Test that episode returns accumulate across steps."""
    num_envs = 2
    max_steps = 3
    envs = SimpleVectorEnv([lambda: DummyCarRacing(max_steps=max_steps) for _ in range(num_envs)])

    obs, _ = envs.reset()
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    completed_episodes = []

    # Step through one complete episode
    for step in range(max_steps):
        actions = np.zeros((num_envs, 2), dtype=np.float32)
        obs, rewards, terminated, truncated, infos = envs.step(actions)
        dones = np.logical_or(terminated, truncated)

        print(f"Step {step}: rewards={rewards}, dones={dones}, episode_returns={episode_returns}")

        episode_returns += rewards

        for i, done in enumerate(dones):
            if done:
                completed_episodes.append(episode_returns[i])
                print(f"  Episode {i} completed with return {episode_returns[i]}")
                episode_returns[i] = 0.0

    envs.close()

    print(f"\nCompleted episode returns: {completed_episodes}")
    assert len(completed_episodes) == num_envs, f"Expected {num_envs} episodes, got {len(completed_episodes)}"

    # DummyCarRacing gives 0.5 reward per step for max_steps steps
    expected_return = 0.5 * max_steps
    for ret in completed_episodes:
        assert abs(ret - expected_return) < 1e-5, f"Expected return ~{expected_return}, got {ret}"
