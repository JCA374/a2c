import numpy as np
import pytest

from env.config import RewardConfig
from env.wrappers import ActionClipWrapper, CarRacingRewardWrapper, CompactObservation, EpisodeTimeLimit
from tests.dummy_envs import DummyCarRacing


def test_action_clip_wrapper_clips_actions():
    env = ActionClipWrapper(DummyCarRacing())
    raw_action = np.array([5.0, -5.0], dtype=np.float32)
    clipped = env.action(raw_action)
    assert np.all(clipped <= env.action_space.high)
    assert np.all(clipped >= env.action_space.low)


def test_episode_time_limit_truncates_after_seconds():
    env = EpisodeTimeLimit(DummyCarRacing(max_steps=100), max_seconds=0.15)
    env.reset()
    _, _, _, truncated, _ = env.step(env.action_space.sample())
    assert truncated is False
    _, _, _, truncated, _ = env.step(env.action_space.sample())
    assert truncated is True


def test_car_racing_reward_wrapper_off_track_penalty():
    env = CarRacingRewardWrapper(DummyCarRacing(max_steps=5), RewardConfig())
    env.reset()
    env.env.set_off_track()
    _, reward, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert terminated is False and truncated is False
    # progress_scale*1 + off_track_penalty + time_penalty
    # With new config: 10.0*1 + (-0.1) + 0.0 = 9.9
    assert pytest.approx(9.9, rel=1e-3) == reward


def test_car_racing_reward_wrapper_finish_bonus():
    env = CarRacingRewardWrapper(DummyCarRacing(max_steps=1), RewardConfig())
    env.reset()
    env.env.set_on_track()
    _, reward, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert terminated is True
    # progress_scale*1 + finish_bonus + time_penalty
    assert reward > 400.0


def test_compact_observation_shape_and_bounds():
    env = CompactObservation(DummyCarRacing())
    obs, _ = env.reset()
    assert obs.shape[0] == 8 + env.lidar_rays
    assert np.all(np.isfinite(obs))
    assert np.all(obs[:8] <= 1.0 + 1e-6)
    assert np.all(obs[:8] >= -1.0 - 1e-6)
    assert np.all(obs[8:] <= 1.0 + 1e-6)
    assert np.all(obs[8:] >= 0.0 - 1e-6)
