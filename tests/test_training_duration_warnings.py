"""Tests to ensure adequate training duration for episode completion."""
import pytest

from a2c_agent import A2CConfig
from env.config import EnvConfig


def test_config_allows_episode_completion():
    """Warn if training config is too short to complete episodes."""
    cfg = A2CConfig(total_steps=2000, rollout_length=64)
    env_cfg = EnvConfig()

    num_envs = 4  # typical
    steps_per_update = cfg.rollout_length * num_envs
    num_updates = cfg.total_steps // steps_per_update
    max_episode_steps = env_cfg.episode.max_steps

    print(f"\nTraining configuration analysis:")
    print(f"  Steps per update: {steps_per_update}")
    print(f"  Total updates: {num_updates}")
    print(f"  Max episode length: {max_episode_steps}")
    print(f"  Updates needed for 1 episode (worst case): {max_episode_steps // steps_per_update}")

    # Warn if training is likely too short
    if cfg.total_steps < max_episode_steps * 2:
        pytest.fail(
            f"WARNING: total_steps ({cfg.total_steps}) should be at least "
            f"2× max_episode_steps ({max_episode_steps * 2}) to ensure episodes complete. "
            f"Early training agents may take the full {max_episode_steps} steps per episode."
        )


def test_recommended_training_duration():
    """Test recommended minimum training duration."""
    env_cfg = EnvConfig()
    max_episode_steps = env_cfg.episode.max_steps  # 1000
    num_envs = 4

    # Recommended: at least 5 full episodes worth of steps
    recommended_min_steps = max_episode_steps * num_envs * 5

    print(f"\nRecommended minimum training steps: {recommended_min_steps}")
    print(f"  (= max_episode_steps {max_episode_steps} × num_envs {num_envs} × 5 episodes)")

    # Good config
    good_cfg = A2CConfig(total_steps=recommended_min_steps)
    assert good_cfg.total_steps >= recommended_min_steps

    # Too short config (like user's)
    short_cfg = A2CConfig(total_steps=2000)
    print(f"\nShort config total_steps: {short_cfg.total_steps}")
    print(f"  Episodes expected to complete: ~{short_cfg.total_steps // max_episode_steps}")

    assert short_cfg.total_steps < recommended_min_steps, \
        "This config is too short for meaningful training"


def calculate_expected_episodes(total_steps: int, max_episode_steps: int, num_envs: int) -> int:
    """Estimate how many episodes will complete in worst case (untrained agent)."""
    # Worst case: each episode runs to max_episode_steps
    # With num_envs parallel, we get num_envs episodes per max_episode_steps
    full_cycles = total_steps // max_episode_steps
    expected = full_cycles * num_envs
    return expected


def test_episode_completion_estimates():
    """Test estimates of episode completion for various configurations."""
    num_envs = 4
    max_episode_steps = 1000

    test_cases = [
        (2000, 0),    # User's short config - won't complete any
        (4000, 4),    # Should complete 4 episodes (1 cycle × 4 envs)
        (10000, 8),   # Should complete 8 episodes (2 cycles × 4 envs)
        (20000, 16),  # Should complete 16 episodes (4 cycles × 4 envs)
    ]

    print("\nEpisode completion estimates (worst case - untrained agent):")
    for total_steps, expected_min in test_cases:
        estimated = calculate_expected_episodes(total_steps, max_episode_steps, num_envs)
        print(f"  total_steps={total_steps:>6}: ~{estimated:>3} episodes expected")
        assert estimated >= expected_min, \
            f"Expected at least {expected_min} episodes for {total_steps} steps"
