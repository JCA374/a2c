from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardConfig:
    """Training reward. The score is always CarRacing's own reward (+1000/N per new tile, -0.1 per frame)."""

    native_scale: float = 1.0  # weight of CarRacing's own step reward (the score)
    off_road_penalty: float = 0.0  # per step with all four wheels off the road
    finish_bonus: float = 0.0  # extra reward when the lap is completed
    early_stop_penalty: float = -50.0  # reward when a training episode is cut short (see EpisodeConfig)


@dataclass
class EpisodeConfig:
    max_steps: int = 1000  # CarRacing's own limit; the score is defined on 1000 frames
    off_road_patience: Optional[int] = 50  # training only: stop after this many steps with no wheel on the road
    no_progress_patience: Optional[int] = 150  # training only: stop after this many steps without a new tile


@dataclass
class ObservationConfig:
    lidar_rays: int = 19
    lidar_fov: float = 0.6  # rays span +-fov*pi around the car's forward direction
    lidar_max_distance: float = 60.0
    lidar_step: float = 1.0
    lookahead: tuple = (2, 4, 7, 11, 16, 22, 30, 40)  # centre-line points ahead, in tiles
    lidar_impl: str = "numba"  # "numba" (training) or "python" (the reference, same answers bit for bit)


@dataclass
class EnvConfig:
    render_mode: Optional[str] = None  # "rgb_array" for videos, "human" for a window
    render_pixels: bool = False  # draw CarRacing's 96x96 state image each step (99% of the step cost; unused)
    domain_randomize: bool = False  # only changes colours, which the compact observation never sees
    action_mode: str = "2d"  # "2d": [steer, gas(+)/brake(-)]; "3d": CarRacing's own [steer, gas, brake]
    training: bool = True  # False: CarRacing's own rules only (no early stops): the score
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)
