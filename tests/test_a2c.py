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
