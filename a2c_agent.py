"""Actor-critic agents for CarRacing: A2C (one gradient step per rollout) and PPO (clipped, several epochs).

Both share the network, the rollout buffer and GAE. PPO is the default because it reuses each rollout ~10 times;
`--algo a2c` gives the original algorithm with the same fixes (normalisation, bootstrapping, working inputs)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class A2CConfig:
    algo: str = "ppo"  # "ppo" or "a2c"
    total_steps: int = 5_000_000
    rollout_length: int = 256  # steps per env between updates
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    lr_final_fraction: float = 0.1  # linear decay to this fraction of learning_rate
    ent_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    ppo_minibatches: int = 8
    ppo_clip: float = 0.2
    target_kl: Optional[float] = 0.03  # stop a PPO update's epochs early above this KL
    hidden: int = 256
    init_log_std: float = -0.5
    normalize_obs: bool = True
    normalize_reward: bool = True
    device: str = "cpu"


class RunningMeanStd:
    """Running mean/variance (Chan et al. parallel update), as in VecNormalize."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64).reshape((-1,) + self.mean.shape)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def state(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load(self, s):
        self.mean, self.var, self.count = np.array(s["mean"]), np.array(s["var"]), s["count"]


def _layer(i, o, gain=math.sqrt(2)):
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


class ActorCritic(nn.Module):
    """Separate policy and value MLPs (tanh). Policy: Gaussian on R^A, squashed by tanh into the action box."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256, init_log_std: float = -0.5,
                 action_low=None, action_high=None):
        super().__init__()
        self.policy_net = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh())
        self.mean_head = _layer(hidden, action_dim, gain=0.01)
        self.value_net = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                                       _layer(hidden, 1, gain=1.0))
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        low = torch.as_tensor(action_low if action_low is not None else -np.ones(action_dim), dtype=torch.float32)
        high = torch.as_tensor(action_high if action_high is not None else np.ones(action_dim), dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def dist(self, obs):
        mean = self.mean_head(self.policy_net(obs))
        return Normal(mean, self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean))

    def value(self, obs):
        return self.value_net(obs).squeeze(-1)

    def _log_prob(self, dist, raw):
        # change of variables for a = scale * tanh(raw) + bias; log(1 - tanh^2) in a stable form
        corr = 2.0 * (math.log(2.0) - raw - nn.functional.softplus(-2.0 * raw))
        return (dist.log_prob(raw) - corr).sum(-1) - torch.log(self.action_scale).sum()

    def act(self, obs):
        """Sample. Returns the env action, the pre-squash sample (stored for the update), log-prob, value."""
        d = self.dist(obs)
        raw = d.sample()
        return torch.tanh(raw) * self.action_scale + self.action_bias, raw, self._log_prob(d, raw), self.value(obs)

    def evaluate(self, obs, raw):
        d = self.dist(obs)
        return self._log_prob(d, raw), d.entropy().sum(-1), self.value(obs)

    def deterministic_action(self, obs):
        return torch.tanh(self.dist(obs).mean) * self.action_scale + self.action_bias


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_dim, action_dim):
        z = lambda *s: torch.zeros((num_steps, num_envs) + s)
        self.obs, self.raw = z(obs_dim), z(action_dim)
        self.rewards, self.dones, self.values, self.log_probs = z(), z(), z(), z()
        self.advantages, self.returns = z(), z()
        self.num_steps, self.step = num_steps, 0

    def add(self, obs, raw, rewards, dones, values, log_probs):
        s = self.step
        self.obs[s], self.raw[s], self.rewards[s] = obs, raw, rewards
        self.dones[s], self.values[s], self.log_probs[s] = dones, values, log_probs
        self.step += 1

    def compute_returns_and_advantages(self, last_values, gamma, lam):
        """GAE. dones[t] = 1 cuts the trace after step t; time-limit cuts were already bootstrapped into rewards."""
        adv = torch.zeros_like(last_values)
        for t in reversed(range(self.num_steps)):
            nxt = last_values if t == self.num_steps - 1 else self.values[t + 1]
            live = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * nxt * live - self.values[t]
            adv = delta + gamma * lam * live * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def flat(self):
        f = lambda x: x.reshape((-1,) + x.shape[2:])
        return f(self.obs), f(self.raw), f(self.log_probs), f(self.advantages), f(self.returns), f(self.values)


class Agent:
    """The network plus observation normalisation: everything needed to drive, saved in one checkpoint."""

    def __init__(self, obs_dim, action_dim, cfg: A2CConfig, action_low=None, action_high=None):
        self.cfg = cfg
        self.net = ActorCritic(obs_dim, action_dim, cfg.hidden, cfg.init_log_std, action_low, action_high)
        self.obs_rms = RunningMeanStd((obs_dim,))

    def norm(self, obs):
        obs = np.asarray(obs, np.float32)
        if self.cfg.normalize_obs:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10, 10)
        return torch.as_tensor(obs, dtype=torch.float32)

    @torch.no_grad()
    def drive(self, obs):
        """Deterministic action for one observation (or a batch)."""
        x = self.norm(obs)
        return self.net.deterministic_action(x).numpy()

    def save(self, path, extra=None):
        torch.save({"net": self.net.state_dict(), "obs_rms": self.obs_rms.state(), "cfg": asdict(self.cfg),
                    "obs_dim": self.net.policy_net[0].in_features, "action_dim": self.net.log_std.numel(),
                    "action_low": (self.net.action_bias - self.net.action_scale).tolist(),
                    "action_high": (self.net.action_bias + self.net.action_scale).tolist(), **(extra or {})}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(ck["obs_dim"], ck["action_dim"], A2CConfig(**ck["cfg"]), ck["action_low"], ck["action_high"])
        agent.net.load_state_dict(ck["net"])
        agent.obs_rms.load(ck["obs_rms"])
        agent.net.eval()
        agent.meta = {k: v for k, v in ck.items() if k not in ("net", "obs_rms")}
        return agent


def update(agent: Agent, opt, buf: RolloutBuffer, cfg: A2CConfig) -> dict:
    """One A2C or PPO update from a full buffer. Returns the losses for logging."""
    obs, raw, old_lp, adv, ret, old_v = buf.flat()
    n = obs.shape[0]
    if cfg.algo == "a2c":
        epochs, mb = 1, n
    else:
        epochs, mb = cfg.ppo_epochs, n // cfg.ppo_minibatches
    stats = {"loss_policy": [], "loss_value": [], "entropy": [], "kl": [], "clip_frac": []}
    for _ in range(epochs):
        perm = torch.randperm(n)
        stop = False
        for s in range(0, n, mb):
            i = perm[s:s + mb]
            a = adv[i]
            a = (a - a.mean()) / (a.std() + 1e-8)
            lp, ent, v = agent.net.evaluate(obs[i], raw[i])
            log_ratio = lp - old_lp[i]
            if cfg.algo == "a2c":
                pl = -(a * lp).mean()
            else:
                ratio = log_ratio.exp()
                pl = -torch.min(ratio * a, ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * a).mean()
            vl = 0.5 * (v - ret[i]).pow(2).mean()
            loss = pl + cfg.value_coef * vl - cfg.ent_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.net.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = ((log_ratio.exp() - 1) - log_ratio).mean().item()
            stats["loss_policy"].append(pl.item())
            stats["loss_value"].append(vl.item())
            stats["entropy"].append(ent.mean().item())
            stats["kl"].append(kl)
            stats["clip_frac"].append(((log_ratio.exp() - 1).abs() > cfg.ppo_clip).float().mean().item())
            if cfg.algo == "ppo" and cfg.target_kl and kl > 1.5 * cfg.target_kl:
                stop = True
                break
        if stop:
            break
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    var_y = ret.var().item()
    out["explained_var"] = float(1 - (ret - old_v).var().item() / var_y) if var_y > 0 else 0.0
    out["std"] = float(agent.net.log_std.exp().mean().item())
    return out
