from pathlib import Path

import numpy as np
import torch

from a2c_agent import ActorCritic
from play_model import load_policy


def test_play_model_loads_policy(tmp_path: Path):
    action_low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    policy = ActorCritic(obs_dim=4, action_dim=3, action_low=action_low, action_high=action_high)
    model_path = tmp_path / "model.pt"
    torch.save(policy.state_dict(), model_path)

    loaded = load_policy(model_path, obs_shape=(4,), action_low=action_low, action_high=action_high, device="cpu")
    obs = torch.zeros((1, 4))
    with torch.no_grad():
        action = loaded.deterministic_action(obs)
    assert action.shape == (1, 3)
