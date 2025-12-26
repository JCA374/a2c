import numpy as np

from env.vector_env import SimpleVectorEnv
from tests.dummy_envs import DummyCarRacing


def test_simple_vector_env_auto_resets_on_done():
    envs = SimpleVectorEnv([lambda: DummyCarRacing(max_steps=1) for _ in range(2)])
    obs, infos = envs.reset()
    assert obs.shape[0] == 2
    actions = np.zeros((2,) + envs.single_action_space.shape, dtype=np.float32)
    next_obs, rewards, terminated, truncated, infos = envs.step(actions)
    assert terminated.all()
    for info in infos:
        assert "terminal_observation" in info
        assert "reset_info" in info
    envs.close()
