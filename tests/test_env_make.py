import numpy as np

from env.config import EnvConfig
from env.make_env import make_env


def test_make_env_real_carracing_short_rollout():
    cfg = EnvConfig(render_mode="rgb_array")
    cfg.episode.max_steps = 5
    env = make_env(cfg, seed=0)
    try:
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
    finally:
        env.close()
