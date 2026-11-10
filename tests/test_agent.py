import math

import numpy as np
import torch

from a2c_agent import A2CConfig, Agent, ActorCritic, RolloutBuffer, RunningMeanStd, update


def test_log_prob_matches_change_of_variables():
    torch.manual_seed(0)
    net = ActorCritic(5, 2)
    obs = torch.randn(64, 5)
    action, raw, lp, _ = net.act(obs)
    d = net.dist(obs)
    expected = (d.log_prob(raw) - torch.log(1 - torch.tanh(raw) ** 2)).sum(-1)
    assert torch.allclose(lp, expected, atol=1e-4)
    assert action.abs().max() <= 1.0


def test_running_mean_std():
    r = RunningMeanStd((3,))
    x = np.random.default_rng(0).normal(2.0, 3.0, (5000, 3))
    for chunk in np.split(x, 10):
        r.update(chunk)
    assert np.allclose(r.mean, x.mean(0), atol=1e-6) and np.allclose(r.var, x.var(0), rtol=1e-6)


def test_gae_cuts_at_done():
    buf = RolloutBuffer(3, 1, 1, 1)
    for t in range(3):
        buf.add(torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1), torch.tensor([1.0 if t == 1 else 0.0]),
                torch.zeros(1), torch.zeros(1))
    buf.compute_returns_and_advantages(torch.tensor([10.0]), gamma=0.5, lam=1.0)
    assert torch.allclose(buf.advantages[:, 0], torch.tensor([1.5, 1.0, 6.0]))


def test_ppo_and_a2c_update_and_save_load(tmp_path):
    for algo in ["ppo", "a2c"]:
        cfg = A2CConfig(algo=algo, ppo_minibatches=2)
        agent = Agent(4, 2, cfg)
        opt = torch.optim.Adam(agent.net.parameters(), lr=1e-3)
        buf = RolloutBuffer(8, 2, 4, 2)
        for _ in range(8):
            o = torch.randn(2, 4)
            a, raw, lp, v = agent.net.act(o)
            buf.add(o, raw, torch.randn(2), torch.zeros(2), v.detach(), lp.detach())
        buf.compute_returns_and_advantages(torch.zeros(2), 0.99, 0.95)
        out = update(agent, opt, buf, cfg)
        assert all(math.isfinite(v) for v in out.values())
        agent.save(tmp_path / "m.pt")
        again = Agent.load(tmp_path / "m.pt")
        o = np.random.default_rng(1).normal(size=(3, 4)).astype(np.float32)
        assert np.allclose(agent.drive(o), again.drive(o))
