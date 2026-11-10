"""Real CarRacing tracks and car poses for the bake-off: out/data_<seed>.npz."""
import math, sys, numpy as np, gymnasium as gym
for seed in [0, 1, 2]:
    env = gym.make("CarRacing-v3").unwrapped
    env._render = lambda mode: None
    env.reset(seed=seed)
    tiles = np.array([[tuple(v) for v in b.fixtures[0].shape.vertices] for b in env.road], dtype=np.float64)
    track = np.array([t[2:4] for t in env.track]); beta = np.array([t[1] for t in env.track])
    rng = np.random.default_rng(seed)
    P, R = 2000, 19
    idx = rng.integers(0, len(track), P)
    off = rng.uniform(-9, 9, P)                     # some poses on the grass, past the kerb
    head = beta[idx] + math.pi / 2 + rng.normal(0, 0.3, P)
    pos = track[idx] + np.column_stack([np.cos(beta[idx]), np.sin(beta[idx])]) * off[:, None]
    ang = head[:, None] + np.linspace(-0.6 * math.pi, 0.6 * math.pi, R)[None]
    dirs = np.stack([np.cos(ang), np.sin(ang)], -1)
    np.savez(f"out/data_{seed}.npz", tiles=tiles.reshape(-1), centers=tiles.mean(1).reshape(-1),
             pos=np.ascontiguousarray(pos).reshape(-1), dirs=np.ascontiguousarray(dirs).reshape(-1), P=P, R=R)
    print(seed, len(tiles), "tiles")
