from __future__ import annotations

from typing import Optional

import gymnasium as gym

from .config import EnvConfig
from .wrappers import ClipAction, ScoreAndShaping, TrackObservation, TwoAxisAction, skip_state_pixels


def make_env(config: Optional[EnvConfig] = None, seed: Optional[int] = None) -> gym.Env:
    """CarRacing-v3 -> pixels off -> action mapping -> score/shaping -> 1000-step limit -> 49-number observation.

    `seed` seeds the first reset; later resets draw new tracks from the env's own generator."""
    config = config or EnvConfig()
    env = gym.make("CarRacing-v3", render_mode=config.render_mode, domain_randomize=config.domain_randomize,
                   continuous=True, max_episode_steps=None)
    env = env.unwrapped  # drop gym.make's own TimeLimit / order checks; ours follow
    if not config.render_pixels:
        skip_state_pixels(env)
    env = TwoAxisAction(env) if config.action_mode == "2d" else ClipAction(env)
    env = ScoreAndShaping(env, config.reward, config.episode, training=config.training)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.episode.max_steps)
    env = TrackObservation(env, config.obs)
    if seed is not None:
        env.reset(seed=seed)
    return env


def score_config(**overrides) -> EnvConfig:
    """CarRacing's own rules (no early stops): what every reported score uses."""
    cfg = EnvConfig(training=False, **overrides)
    return cfg
