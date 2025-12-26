from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym


class _Joint:
    def __init__(self, angle: float = 0.1):
        self.angle = angle


class _Wheel:
    def __init__(self, ground_contact: bool = True, steer_angle: float = 0.1):
        self.ground_contact = ground_contact
        self.joint = _Joint(angle=steer_angle)


class _Hull:
    def __init__(self):
        self.linearVelocity = np.array([10.0, 0.0], dtype=np.float32)
        self.angularVelocity = 0.1
        self.angle = 0.0
        self.position = np.array([0.0, 0.0], dtype=np.float32)


class _Car:
    def __init__(self, traction: bool = True, steer_angle: float = 0.1):
        self.hull = _Hull()
        self.wheels = [_Wheel(ground_contact=traction, steer_angle=steer_angle) for _ in range(4)]


class DummyCarRacing(gym.Env):
    """Lightweight stand-in for CarRacing for fast tests."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, max_steps: int = 5, traction: bool = True, collision: bool = False):
        super().__init__()
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.ones(4, dtype=np.float32), high=np.ones(4, dtype=np.float32))
        self.dt = 0.1
        self.max_steps = max_steps
        self.collision = collision
        self.car = _Car(traction=traction)
        # Road is a list of tiles; each tile is a tuple/list where index 0 is a polygon.
        self.road = [
            [
                [(-100.0, -100.0), (100.0, -100.0), (100.0, 100.0), (-100.0, 100.0)]
            ]
        ]
        self.tile_visited_count = 0.0
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self.tile_visited_count = 0.0
        self.car = _Car(traction=True)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        self.tile_visited_count += 1.0
        self.car.hull.position += np.array([0.5, 0.0], dtype=np.float32)
        terminated = self._step_count >= self.max_steps
        truncated = False
        reward = 0.5
        info = {
            "track_progress": 1.0,
            "finished_track": terminated,
            "crashed": self.collision,
        }
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def set_off_track(self):
        for wheel in self.car.wheels:
            wheel.ground_contact = False

    def set_on_track(self):
        for wheel in self.car.wheels:
            wheel.ground_contact = True
