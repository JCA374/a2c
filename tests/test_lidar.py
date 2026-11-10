"""The numba lidar (training) gives the same answers as the pure-Python reference, bit for bit."""
import math

import gymnasium as gym
import numpy as np
import pytest

from env.lidar import cast_numba, cast_python


@pytest.mark.parametrize("seed", [0, 7])
def test_numba_matches_reference_on_real_tracks(seed):
    env = gym.make("CarRacing-v3").unwrapped
    env._render = lambda mode: None
    env.reset(seed=seed)
    quads = np.array([[tuple(v) for v in b.fixtures[0].shape.vertices] for b in env.road], dtype=np.float64)
    tiles, centers = quads.reshape(-1).copy(), quads.mean(1).reshape(-1).copy()
    track = np.array([t[2:4] for t in env.track])
    rng = np.random.default_rng(seed)
    for _ in range(60):
        i = rng.integers(len(track))
        x, y = track[i] + rng.uniform(-10, 10, 2)
        ang = rng.uniform(-math.pi, math.pi) + np.linspace(-0.6 * math.pi, 0.6 * math.pi, 19)
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1).reshape(-1).copy()
        ref = cast_python(tiles, centers, x, y, dirs, 1.0, 60.0)
        assert (cast_numba(tiles, centers, x, y, dirs, 1.0, 60.0) == ref).all()


def test_rays_see_the_road():
    """v1's lidar read 0 on every ray (it tested Box2D bodies as polygons). On the start line it must not."""
    from env.make_env import make_env
    obs, _ = make_env(seed=0).reset(seed=0)
    rays = obs[-19:]
    assert rays.max() > 0.5 and rays.min() >= 0.0 and len(set(np.round(rays, 3))) > 3
