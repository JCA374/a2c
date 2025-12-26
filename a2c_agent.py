from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

from log_utils import TrainingLogger

@dataclass
class A2CConfig:
    total_steps: int = 500_000
    rollout_length: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_interval: int = 10
    device: Optional[str] = None
    clip_advantages: Optional[float] = None  # e.g., 10.0 to prevent extreme updates


class RolloutBuffer:
    """Fixed-size rollout storage with GAE computation."""

    def __init__(self, num_steps: int, num_envs: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], device):
        obs_dim = int(np.prod(obs_shape))
        action_dim = int(np.prod(action_shape))
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)

        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        self._step = 0

    def add(self, obs, actions, rewards, dones, values, log_probs):
        if self._step >= self.num_steps:
            raise IndexError("RolloutBuffer is full")
        self.obs[self._step] = obs
        self.actions[self._step] = actions
        self.rewards[self._step] = rewards
        self.dones[self._step] = dones
        self.values[self._step] = values
        self.log_probs[self._step] = log_probs
        self._step += 1

    def compute_returns_and_advantages(self, last_values, gamma: float, gae_lambda: float):
        last_advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_steps)):
            next_values = last_values if step == self.num_steps - 1 else self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            self.advantages[step] = last_advantage
        self.returns = self.advantages + self.values

    def get(self):
        # Flatten rollout dimension
        batch_obs = self.obs.reshape(-1, self.obs.shape[-1])
        batch_actions = self.actions.reshape(-1, self.actions.shape[-1])
        batch_log_probs = self.log_probs.reshape(-1)
        batch_advantages = self.advantages.reshape(-1)
        batch_returns = self.returns.reshape(-1)
        batch_values = self.values.reshape(-1)
        return batch_obs, batch_actions, batch_log_probs, batch_advantages, batch_returns, batch_values

    def reset(self):
        self._step = 0


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_low=None, action_high=None):
        super().__init__()
        hidden = 256
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        if action_low is None or action_high is None:
            action_low = -np.ones(action_dim, dtype=np.float32)
            action_high = np.ones(action_dim, dtype=np.float32)
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        self._init_action_biases()

    def forward(self, obs: torch.Tensor):
        features = self.policy_net(obs)
        mean = self.mean_head(features)
        value = self.value_net(obs).squeeze(-1)
        return mean, value

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean, _ = self.forward(obs)
        std = torch.exp(self.log_std).clamp_min(1e-4)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor):
        dist = self.get_dist(obs)
        raw_actions = dist.sample()
        squashed = torch.tanh(raw_actions)
        actions = squashed * self.action_scale + self.action_bias
        log_probs = dist.log_prob(raw_actions).sum(-1)
        log_probs = log_probs - self._squash_correction(squashed)
        entropy = dist.entropy().sum(-1)
        _, value = self.forward(obs)
        return actions, log_probs, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.get_dist(obs)
        scaled = self._unsquash(actions)
        raw_actions = self._atanh(scaled)
        log_probs = dist.log_prob(raw_actions).sum(-1)
        log_probs = log_probs - self._squash_correction(scaled)
        entropy = dist.entropy().sum(-1)
        _, value = self.forward(obs)
        return log_probs, entropy, value

    def get_value(self, obs: torch.Tensor):
        _, value = self.forward(obs)
        return value

    def deterministic_action(self, obs: torch.Tensor):
        mean, _ = self.forward(obs)
        squashed = torch.tanh(mean)
        return squashed * self.action_scale + self.action_bias

    def _squash_correction(self, squashed: torch.Tensor) -> torch.Tensor:
        log_scale = torch.log(self.action_scale + 1e-6).sum()
        log_det = log_scale + torch.log(1.0 - squashed**2 + 1e-6).sum(-1)
        return log_det

    def _unsquash(self, actions: torch.Tensor) -> torch.Tensor:
        scaled = (actions - self.action_bias) / (self.action_scale + 1e-6)
        return torch.clamp(scaled, -1.0 + 1e-6, 1.0 - 1e-6)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _init_action_biases(self) -> None:
        """Bias gas/brake toward motion for asymmetric action spaces like CarRacing."""
        if self.mean_head.bias is None:
            return
        if self.action_low.numel() >= 3:
            low = self.action_low.cpu().numpy()
            high = self.action_high.cpu().numpy()
            with torch.no_grad():
                if np.isclose(low[1], 0.0) and np.isclose(high[1], 1.0):
                    # Encourage throttle at initialization.
                    self.mean_head.bias[1] = 1.0
                if np.isclose(low[2], 0.0) and np.isclose(high[2], 1.0):
                    # Discourage braking at initialization.
                    self.mean_head.bias[2] = -2.0


class A2CTrainer:
    def __init__(
        self,
        envs,
        config: Optional[A2CConfig] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
        action_shape: Optional[Tuple[int, ...]] = None,
        logger: Optional[TrainingLogger] = None,
        best_model_path: Optional[Path] = None,
    ):
        self.envs = envs
        self.cfg = config or A2CConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        obs_shape = obs_shape or envs.single_observation_space.shape
        action_shape = action_shape or envs.single_action_space.shape
        obs_dim = int(np.prod(obs_shape))
        action_dim = int(np.prod(action_shape))
        action_low = envs.single_action_space.low.reshape(-1)
        action_high = envs.single_action_space.high.reshape(-1)
        self.policy = ActorCritic(obs_dim, action_dim, action_low=action_low, action_high=action_high).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate)
        self.buffer = RolloutBuffer(
            num_steps=self.cfg.rollout_length,
            num_envs=getattr(envs, "num_envs", 1),
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=self.device,
        )
        self.best_model_path = Path(best_model_path) if best_model_path else (logger.log_dir / "best_model.pt" if logger else None)

    def train(self):
        obs, _ = self.envs.reset()
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        global_step = 0
        num_envs = getattr(self.envs, "num_envs", 1)
        episode_returns = np.zeros(num_envs, dtype=np.float32)
        episode_lengths = np.zeros(num_envs, dtype=np.int32)
        completed_returns = deque(maxlen=10)
        episodes_completed_count = 0
        best_return = -np.inf

        total_updates = self.cfg.total_steps // (self.cfg.rollout_length * num_envs)
        pbar = tqdm(range(1, total_updates + 1), desc="Training", unit="update")
        for update in pbar:
            self.buffer.reset()
            for step in range(self.cfg.rollout_length):
                with torch.no_grad():
                    actions, log_probs, _, values = self.policy.act(obs)
                next_obs, rewards, terminated, truncated, infos = self.envs.step(actions.cpu().numpy())
                dones = np.logical_or(terminated, truncated)

                rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
                dones_t = torch.as_tensor(dones, device=self.device, dtype=torch.float32)
                self.buffer.add(obs, actions, rewards_t, dones_t, values, log_probs)

                episode_returns += rewards
                episode_lengths += 1
                for i, done in enumerate(dones):
                    if done:
                        is_new_best = episode_returns[i] > best_return
                        best_return = max(best_return, episode_returns[i])
                        if self.best_model_path and is_new_best:
                            torch.save(self.policy.state_dict(), self.best_model_path)

                        completed_returns.append((episode_returns[i], episode_lengths[i]))
                        episodes_completed_count += 1
                        episode_returns[i] = 0.0
                        episode_lengths[i] = 0

                obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
                global_step += num_envs

            with torch.no_grad():
                next_values = self.policy.get_value(obs)
            self.buffer.compute_returns_and_advantages(next_values, self.cfg.gamma, self.cfg.gae_lambda)

            (
                batch_obs,
                batch_actions,
                old_log_probs,
                advantages,
                returns,
                values,
            ) = self.buffer.get()

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Clip advantages to prevent extreme updates from penalty spikes
            if self.cfg.clip_advantages is not None:
                advantages = torch.clamp(
                    advantages,
                    -self.cfg.clip_advantages,
                    self.cfg.clip_advantages
                )

            log_probs, entropy, values_pred = self.policy.evaluate_actions(batch_obs, batch_actions)

            policy_loss = -(advantages.detach() * log_probs).mean()
            value_loss = F.mse_loss(values_pred, returns)
            entropy_loss = entropy.mean()

            loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.ent_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

            # Compute metrics
            mean_return = float(np.mean([r for r, _ in completed_returns]) if completed_returns else 0.0)
            mean_length = float(np.mean([l for _, l in completed_returns]) if completed_returns else 0.0)
            # Only update best_return when we have actual completed episodes
            if completed_returns:
                # best_return is now updated inside the episode completion loop
                pass
            action_mean = self.buffer.actions.mean(dim=(0, 1)).detach().cpu().numpy()
            action_std = self.buffer.actions.std(dim=(0, 1)).detach().cpu().numpy()
            rollout_reward = float(self.buffer.rewards.mean().item())

            # Update progress bar with key metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'return': f'{mean_return:.1f}',
                'eps': episodes_completed_count,
                'steps': global_step
            })

            if update % self.cfg.log_interval == 0:
                # Detailed console output at log intervals
                tqdm.write(
                    f"Update {update}/{total_updates} | "
                    f"step={global_step} | "
                    f"loss={loss.item():.3f} | "
                    f"return={mean_return:.2f} | "
                    f"eps={episodes_completed_count} | "
                    f"best={best_return:.2f}"
                )
                if self.logger:
                    lr = self.optimizer.param_groups[0].get("lr", self.cfg.learning_rate)
                    metrics = {
                        "update": update,
                        "global_step": global_step,
                        "loss_total": float(loss.item()),
                        "loss_policy": float(policy_loss.item()),
                        "loss_value": float(value_loss.item()),
                        "entropy": float(entropy_loss.item()),
                        "mean_return_last10": mean_return,
                        "mean_length_last10": mean_length,
                        "best_return": best_return,
                        "episodes_completed": episodes_completed_count,
                        "learning_rate": lr,
                        "gamma": self.cfg.gamma,
                        "gae_lambda": self.cfg.gae_lambda,
                        "entropy_coef": self.cfg.ent_coef,
                        "value_coef": self.cfg.value_coef,
                        "max_grad_norm": self.cfg.max_grad_norm,
                        "rollout_length": self.cfg.rollout_length,
                        "num_envs": num_envs,
                        "rollout_reward_mean": rollout_reward,
                    }
                    for idx, (mean_val, std_val) in enumerate(zip(action_mean, action_std)):
                        metrics[f"action_mean_{idx}"] = float(mean_val)
                        metrics[f"action_std_{idx}"] = float(std_val)
                    if action_mean.shape[0] >= 3:
                        gas = self.buffer.actions[..., 1]
                        brake = self.buffer.actions[..., 2]
                        metrics["action_brake_gt_gas"] = float((brake > gas).float().mean().item())
                    self.logger.log_metrics(
                        metrics
                    )
                    # Model saving is now handled in the episode completion loop.
                    # This ensures we save the model that achieved the new best return.

        pbar.close()
        return self.policy, completed_returns
