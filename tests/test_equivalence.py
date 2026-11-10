"""Skipping CarRacing's pixel render (training) leaves the world unchanged: same observations, rewards and ends."""
import numpy as np

from env.config import EnvConfig
from env.make_env import make_env


def _roll(render_pixels, impl, seed, n=400):
    cfg = EnvConfig(training=False, render_pixels=render_pixels)
    cfg.obs.lidar_impl = impl
    env = make_env(cfg)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = [obs]
    for _ in range(n):
        a = np.array([np.sin(len(out) / 20.0), 0.4 + 0.3 * rng.random()])
        obs, r, term, trunc, info = env.step(a)
        out.append(np.append(obs, [r, info["score"], term, trunc]))
        if term or trunc:
            break
    return np.concatenate(out)


def test_fast_env_equals_reference():
    for seed in [1, 2]:
        ref = _roll(True, "python", seed)  # the reference: pixels drawn, pure-Python lidar
        fast = _roll(False, "numba", seed)
        assert ref.shape == fast.shape and (ref == fast).all()
