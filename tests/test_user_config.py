"""Test with the user's actual configuration that showed zeros."""
import csv
from pathlib import Path

from a2c_agent import A2CConfig, A2CTrainer
from env.config import EnvConfig
from env.make_env import make_env
from env.vector_env import SimpleVectorEnv
from log_utils import TrainingLogger


def test_user_config_episode_completion(tmp_path: Path):
    """Reproduce user's configuration to see why episodes don't complete."""
    num_envs = 4
    rollout_length = 64
    total_steps = 20000  # Same as user's quick_check_v4

    env_config = EnvConfig(render_mode="rgb_array")
    # User's config has max_steps=1000, max_seconds=60.0

    envs = SimpleVectorEnv([
        lambda seed=i: make_env(env_config, seed=seed)
        for i in range(num_envs)
    ])

    cfg = A2CConfig(
        total_steps=total_steps,
        rollout_length=rollout_length,
        log_interval=2  # Log every 2 updates like user
    )
    logger = TrainingLogger(tmp_path, config={"test": "user_config_repro"})

    trainer = A2CTrainer(envs, cfg, logger=logger)

    try:
        policy, completed_returns = trainer.train()
    finally:
        envs.close()
        logger.close()

    print(f"\nCompleted {len(completed_returns)} episodes in {total_steps} total steps")
    if len(completed_returns) > 0:
        print(f"First 5 episode returns: {[f'{r:.2f}' for r, _ in completed_returns[:5]]}")
        print(f"First 5 episode lengths: {[l for _, l in completed_returns[:5]]}")
        print(f"Last 5 episode returns: {[f'{r:.2f}' for r, _ in completed_returns[-5:]]}")
        print(f"Last 5 episode lengths: {[l for _, l in completed_returns[-5:]]}")
    else:
        print("NO EPISODES COMPLETED!")

    # Read the metrics
    metrics_file = tmp_path / "metrics.csv"
    with metrics_file.open() as f:
        rows = list(csv.DictReader(f))

    print("\nFirst 10 metric rows:")
    for row in rows[:10]:
        print(f"Update {row['update']:>3}: "
              f"global_step={row['global_step']:>5}, "
              f"episodes_completed={row['episodes_completed']:>3}, "
              f"mean_return={float(row['mean_return_last10']):>6.2f}, "
              f"rollout_reward={float(row['rollout_reward_mean']):>7.3f}")

    # Check if this matches user's issue
    episodes_completed = int(rows[-1]['episodes_completed'])
    mean_returns = [float(row['mean_return_last10']) for row in rows]

    if episodes_completed == 0:
        print("\n⚠️  REPRODUCED USER'S BUG: No episodes completed!")
        print(f"Total steps: {total_steps}")
        print(f"Rollout length: {rollout_length}")
        print(f"Num envs: {num_envs}")
        print(f"Total rollouts: {total_steps // (rollout_length * num_envs)}")
        print(f"Steps per rollout: {rollout_length * num_envs}")

        # Analyze why
        print("\nPossible reasons:")
        print(f"1. Episodes take > {rollout_length} steps to complete")
        print(f"2. Total training too short ({total_steps} steps)")
        print(f"3. Early training agent is very bad, episodes run to max_steps (1000)")
    else:
        print(f"\n✓ Episodes completed: {episodes_completed}")
        print(f"  Mean returns: {mean_returns}")
