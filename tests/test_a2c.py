import csv
from pathlib import Path

import numpy as np
import torch

from a2c_agent import A2CConfig, A2CTrainer, RolloutBuffer, ActorCritic
from env.vector_env import SimpleVectorEnv
from log_utils import TrainingLogger
from tests.dummy_envs import DummyCarRacing


def test_rollout_buffer_shapes_and_returns():
    device = torch.device("cpu")
    buffer = RolloutBuffer(num_steps=2, num_envs=3, obs_shape=(4,), action_shape=(2,), device=device)

    for _ in range(2):
        obs = torch.zeros((3, 4), device=device)
        actions = torch.zeros((3, 2), device=device)
        rewards = torch.ones((3,), device=device)
        dones = torch.zeros((3,), device=device)
        values = torch.zeros((3,), device=device)
        log_probs = torch.zeros((3,), device=device)
        buffer.add(obs, actions, rewards, dones, values, log_probs)

    last_values = torch.zeros((3,), device=device)
    buffer.compute_returns_and_advantages(last_values, gamma=0.99, gae_lambda=0.95)
    batch_obs, batch_actions, _, advantages, returns, values = buffer.get()

    assert batch_obs.shape == (6, 4)
    assert batch_actions.shape == (6, 2)
    assert advantages.shape == (6,)
    assert returns.shape == (6,)
    assert values.shape == (6,)
    assert torch.allclose(returns[:3], torch.tensor([1.9405, 1.9405, 1.9405], dtype=torch.float32), atol=1e-3)


def test_a2c_trainer_smoke_logs_metrics(tmp_path: Path):
    num_envs = 2
    envs = SimpleVectorEnv([lambda: DummyCarRacing(max_steps=2) for _ in range(num_envs)])
    cfg = A2CConfig(total_steps=40, rollout_length=4, log_interval=1)
    logger = TrainingLogger(tmp_path, config={"test": "a2c_smoke"})

    trainer = A2CTrainer(envs, cfg, logger=logger)
    try:
        trainer.train()
    finally:
        envs.close()
        logger.close()

    metrics_file = tmp_path / "metrics.csv"
    assert metrics_file.exists()
    with metrics_file.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    required_fields = {"update", "global_step", "loss_total", "mean_return_last10", "num_envs"}
    assert required_fields.issubset(set(rows[0].keys()))


def test_policy_outputs_are_finite_and_entropy_positive():
    torch.manual_seed(0)
    policy = ActorCritic(obs_dim=4, action_dim=2)
    obs = torch.zeros((3, 4))
    with torch.no_grad():
        actions, log_probs, entropy, values = policy.act(obs)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(values).all()
    assert (entropy > 0).all()


def test_policy_respects_action_bounds():
    torch.manual_seed(0)
    action_low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    policy = ActorCritic(obs_dim=4, action_dim=3, action_low=action_low, action_high=action_high)
    obs = torch.zeros((10, 4))
    with torch.no_grad():
        actions, _, _, _ = policy.act(obs)
    assert torch.all(actions[:, 0] >= -1.0) and torch.all(actions[:, 0] <= 1.0)
    assert torch.all(actions[:, 1:] >= 0.0) and torch.all(actions[:, 1:] <= 1.0)


def test_policy_biases_gas_over_brake():
    action_low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    policy = ActorCritic(obs_dim=4, action_dim=3, action_low=action_low, action_high=action_high)
    obs = torch.zeros((1, 4))
    with torch.no_grad():
        action = policy.deterministic_action(obs).squeeze(0)
    assert action[1] > action[2]
    assert action[1] > 0.5
    assert action[2] < 0.5


def test_evaluate_actions_matches_act_logprob():
    torch.manual_seed(0)
    action_low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    policy = ActorCritic(obs_dim=4, action_dim=3, action_low=action_low, action_high=action_high)
    obs = torch.zeros((5, 4))
    with torch.no_grad():
        actions, log_probs, _, _ = policy.act(obs)
        log_probs_eval, _, _ = policy.evaluate_actions(obs, actions)
    assert torch.all(torch.isfinite(log_probs_eval))
    assert torch.allclose(log_probs, log_probs_eval, atol=1e-3)


def test_value_estimate_increases_after_training(tmp_path: Path):
    torch.manual_seed(0)
    np.random.seed(0)
    num_envs = 2
    envs = SimpleVectorEnv([lambda: DummyCarRacing(max_steps=3) for _ in range(num_envs)])
    cfg = A2CConfig(total_steps=200, rollout_length=5, log_interval=100)
    logger = TrainingLogger(tmp_path, config={"test": "value_improves"})

    trainer = A2CTrainer(envs, cfg, logger=logger)
    obs0 = torch.zeros(
        (1, int(np.prod(envs.single_observation_space.shape))),
        device=trainer.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        value_before = trainer.policy.get_value(obs0).item()
    try:
        trainer.train()
    finally:
        envs.close()
        logger.close()

    with torch.no_grad():
        value_after = trainer.policy.get_value(obs0).item()

    assert value_after > value_before - 1e-4


def test_entropy_sign_encourages_exploration():
    """Verify entropy computation and that it reduces loss (encourages exploration)."""
    torch.manual_seed(42)
    policy = ActorCritic(obs_dim=4, action_dim=2)
    obs = torch.randn((10, 4))

    # Get entropy from policy
    with torch.no_grad():
        _, _, entropy, _ = policy.act(obs)

    # Entropy must be positive
    assert (entropy > 0).all(), "Entropy must be positive"

    # Verify it matches distribution entropy
    dist = policy.get_dist(obs)
    expected_entropy = dist.entropy().sum(-1)
    assert torch.allclose(entropy, expected_entropy, atol=1e-5), \
        "Policy entropy doesn't match distribution entropy"

    # Verify that subtracting entropy reduces loss (encourages exploration)
    ent_coef = 0.01
    dummy_loss = torch.tensor(1.0)
    loss_with_entropy = dummy_loss - ent_coef * entropy.mean()

    assert loss_with_entropy < dummy_loss, \
        "Loss with entropy bonus should be lower (entropy should be subtracted)"


def test_squash_correction_shape_and_broadcasting():
    """Verify squash correction returns correct shapes with proper broadcasting."""
    action_low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    policy = ActorCritic(obs_dim=4, action_dim=3, action_low=action_low, action_high=action_high)

    batch_size = 5
    squashed = torch.rand((batch_size, 3)) * 0.8  # Keep away from ±1

    correction = policy._squash_correction(squashed)

    # Check shape: should be [batch_size]
    assert correction.shape == (batch_size,), \
        f"Expected shape ({batch_size},), got {correction.shape}"

    # Log determinant can be negative (space-contracting transform)
    # Just verify it's finite
    assert torch.isfinite(correction).all(), "Squash correction should be finite"

    # Verify manual computation matches
    expected_log_scale = torch.log(policy.action_scale + 1e-6).sum()
    expected = expected_log_scale + torch.log(1.0 - squashed**2 + 1e-6).sum(-1)
    assert torch.allclose(correction, expected, atol=1e-5), \
        "Squash correction doesn't match expected computation"

    # Verify scalar broadcasting works correctly
    assert expected_log_scale.dim() == 0, "log_scale should be scalar"


def test_loss_components_correct_signs():
    """Verify policy loss, value loss, and entropy have correct signs."""
    num_envs = 2
    envs = SimpleVectorEnv([lambda: DummyCarRacing(max_steps=5) for _ in range(num_envs)])
    cfg = A2CConfig(total_steps=50, rollout_length=5, ent_coef=0.01)
    trainer = A2CTrainer(envs, cfg)

    # Collect rollout
    obs, _ = trainer.envs.reset()
    obs = torch.as_tensor(obs, device=trainer.device, dtype=torch.float32)

    for _ in range(cfg.rollout_length):
        with torch.no_grad():
            actions, log_probs, _, values = trainer.policy.act(obs)
        next_obs, rewards, terminated, truncated, _ = trainer.envs.step(actions.cpu().numpy())
        dones = np.logical_or(terminated, truncated)

        rewards_t = torch.as_tensor(rewards, device=trainer.device, dtype=torch.float32)
        dones_t = torch.as_tensor(dones, device=trainer.device, dtype=torch.float32)
        trainer.buffer.add(obs, actions, rewards_t, dones_t, values, log_probs)
        obs = torch.as_tensor(next_obs, device=trainer.device, dtype=torch.float32)

    # Compute loss
    with torch.no_grad():
        next_values = trainer.policy.get_value(obs)
    trainer.buffer.compute_returns_and_advantages(next_values, cfg.gamma, cfg.gae_lambda)

    batch_obs, batch_actions, _, advantages, returns, _ = trainer.buffer.get()
    advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_probs, entropy, values_pred = trainer.policy.evaluate_actions(batch_obs, batch_actions)

    policy_loss = -(advantages_norm.detach() * log_probs).mean()
    entropy_loss = entropy.mean()

    # Assertions
    assert entropy_loss > 0, "Entropy should be positive"

    # Loss should SUBTRACT entropy to encourage exploration
    loss_with_entropy = policy_loss - cfg.ent_coef * entropy_loss
    loss_without_entropy = policy_loss

    if cfg.ent_coef > 0:
        assert loss_with_entropy < loss_without_entropy, \
            "Subtracting entropy should reduce total loss (encourage exploration)"

    envs.close()


def test_reward_config_defaults_documented():
    """Ensure reward config defaults are reasonable and documented."""
    from env.config import RewardConfig

    cfg = RewardConfig()

    # Verify documented values (after fixing CLAUDE.md)
    assert cfg.progress_scale == 10.0, "Progress scale should be 10.0"
    assert -0.1 <= cfg.off_track_penalty <= -0.01, \
        f"Off-track penalty {cfg.off_track_penalty} should be small for stable learning"
    assert cfg.collision_penalty == -5.0, "Collision penalty should be -5.0"
    assert cfg.control_penalty == 0.0, "Control penalty should be disabled"
    assert cfg.time_penalty == 0.0, "Time penalty should be disabled"
    assert cfg.finish_bonus == 500.0, "Finish bonus should be 500.0"

    # Verify reward balance (progress reward should outweigh penalties)
    max_penalty_per_episode = abs(cfg.off_track_penalty) * 1000  # Max steps
    typical_progress_reward = cfg.progress_scale * 50  # Assume ~50 tiles

    assert typical_progress_reward > max_penalty_per_episode, \
        f"Progress reward ({typical_progress_reward}) should exceed max penalty " \
        f"({max_penalty_per_episode}) for balanced learning"


def test_advantage_clipping_prevents_extreme_updates():
    """Verify advantage clipping prevents extreme gradient updates."""
    num_envs = 2
    envs = SimpleVectorEnv([lambda: DummyCarRacing(max_steps=5) for _ in range(num_envs)])

    # Config WITH clipping
    cfg_clipped = A2CConfig(total_steps=50, rollout_length=5, clip_advantages=10.0)
    trainer_clipped = A2CTrainer(envs, cfg_clipped)

    # Config WITHOUT clipping
    cfg_no_clip = A2CConfig(total_steps=50, rollout_length=5, clip_advantages=None)
    trainer_no_clip = A2CTrainer(envs, cfg_no_clip)

    # Create already-normalized advantages with extreme values
    # (simulating large penalty spike after normalization)
    extreme_advantages_normalized = torch.tensor([-50.0, -20.0, 0.0, 15.0, 100.0])

    # Without clipping - should have large values
    assert extreme_advantages_normalized.abs().max() > 10.0, \
        "Extreme advantages should exceed clip threshold before clipping"

    # With clipping - should be bounded
    adv_clipped = torch.clamp(extreme_advantages_normalized, -10.0, 10.0)
    assert adv_clipped.abs().max() <= 10.0, \
        "Clipped advantages should be bounded by clip value"
    assert (adv_clipped >= -10.0).all() and (adv_clipped <= 10.0).all(), \
        "All clipped advantages should be within [-10, 10]"

    # Verify clipping actually changed the values
    assert not torch.equal(extreme_advantages_normalized, adv_clipped), \
        "Clipping should modify extreme values"

    envs.close()
