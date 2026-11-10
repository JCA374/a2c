from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from .config import EpisodeConfig, ObservationConfig, RewardConfig
from .lidar import CASTERS


def skip_state_pixels(env: gym.Env) -> None:
    """CarRacing draws its 96x96 state image inside every step(); that is ~99% of a step's cost (8 ms vs 0.09 ms)
    and the compact observation never looks at it. Physics is untouched (tests/test_equivalence.py).
    env.render() for videos still works."""
    raw = env.unwrapped
    draw = raw._render
    raw._render = lambda mode, draw=draw: None if mode == "state_pixels" else draw(mode)


class TwoAxisAction(gym.ActionWrapper):
    """[steer, pedal] in [-1, 1]^2: pedal > 0 is gas, pedal < 0 is brake (never both at once)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def action(self, action):
        steer, pedal = float(np.clip(action[0], -1, 1)), float(np.clip(action[1], -1, 1))
        return np.array([steer, max(pedal, 0.0), max(-pedal, 0.0)], dtype=np.float32)


class ClipAction(gym.ActionWrapper):
    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


def wheels_on_road(raw) -> list:
    return [1.0 if len(w.tiles) > 0 else 0.0 for w in raw.car.wheels]


class ScoreAndShaping(gym.Wrapper):
    """Keeps CarRacing's own reward as `info["score"]` (the metric) and turns it into the training reward.

    With `training=True` it also cuts episodes short (a training-time speed-up, not a rule of the task): after
    `off_road_patience` steps with no wheel on the road, or `no_progress_patience` steps without a new tile."""

    def __init__(self, env, reward: RewardConfig, episode: EpisodeConfig, training: bool):
        super().__init__(env)
        self.cfg, self.ep, self.training = reward, episode, training

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.score, self.off_road, self.stale, self.tiles = 0.0, 0, 0, 0
        return obs, info

    def step(self, action):
        obs, native, terminated, truncated, info = self.env.step(action)
        raw = self.env.unwrapped
        self.score += native
        on = wheels_on_road(raw)
        reward = self.cfg.native_scale * native
        if not any(on):
            reward += self.cfg.off_road_penalty
            self.off_road += 1
        else:
            self.off_road = 0
        if raw.tile_visited_count > self.tiles:
            self.tiles, self.stale = raw.tile_visited_count, 0
        else:
            self.stale += 1
        lap = bool(info.get("lap_finished", False))
        if lap:
            reward += self.cfg.finish_bonus
        early = None
        if self.training and not (terminated or truncated):
            if self.ep.off_road_patience and self.off_road >= self.ep.off_road_patience:
                early = "off_road"
            elif self.ep.no_progress_patience and self.stale >= self.ep.no_progress_patience:
                early = "no_progress"
            if early:
                reward += self.cfg.early_stop_penalty
                terminated = True
        info = dict(info)
        info.update(score=self.score, native_reward=native, tiles=raw.tile_visited_count / len(raw.track),
                    lap=lap, early_stop=early, wheels_on_road=sum(on))
        return obs, float(reward), terminated, truncated, info


class TrackObservation(gym.ObservationWrapper):
    """The car's state as 49 numbers instead of pixels (see the site's "How it sees" tab):

    0-2   speed, forward and sideways velocity (/60)       3  yaw rate (/3)        4  front-wheel steer angle (/0.4)
    5-8   each wheel on the road (1) or grass (0)         9-10 front, rear wheel rim speed (/60)
    11    signed distance from the centre line (/ half the road width)
    12-13 heading error against the track direction (sin, cos)
    14-29 centre-line points 2..40 tiles ahead in the car's frame (forward, left; /60)
    30-48 lidar: distance to the road edge along 19 rays over +-108 degrees (/60)"""

    BASE = 14

    def __init__(self, env, cfg: ObservationConfig):
        super().__init__(env)
        self.cfg = cfg
        self.cast = CASTERS[cfg.lidar_impl]
        n = self.BASE + 2 * len(cfg.lookahead) + cfg.lidar_rays
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
        self.ray_angles = np.linspace(-cfg.lidar_fov * math.pi, cfg.lidar_fov * math.pi, cfg.lidar_rays)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw = self.env.unwrapped
        quads = np.array([[tuple(v) for v in b.fixtures[0].shape.vertices] for b in raw.road], dtype=np.float64)
        self.tiles = np.ascontiguousarray(quads.reshape(-1))
        self.centers = np.ascontiguousarray(quads.mean(1).reshape(-1))
        self.track = np.array([t[2:4] for t in raw.track], dtype=np.float64)
        d = np.roll(self.track, -1, 0) - self.track
        self.track_dir = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.idx = None
        return self.observation(obs), info

    def _nearest(self, pos):
        n = len(self.track)
        if self.idx is not None:
            window = (self.idx + np.arange(-10, 31)) % n
            d = np.sum((self.track[window] - pos) ** 2, 1)
            j = int(np.argmin(d))
            if d[j] < 15.0 ** 2:
                self.idx = int(window[j])
                return self.idx
        self.idx = int(np.argmin(np.sum((self.track - pos) ** 2, 1)))
        return self.idx

    def observation(self, _):
        raw = self.env.unwrapped
        hull = raw.car.hull
        a = float(hull.angle)
        fx, fy = -math.sin(a), math.cos(a)  # the car's nose
        lx, ly = -fy, fx  # its left
        vx, vy = float(hull.linearVelocity[0]), float(hull.linearVelocity[1])
        pos = np.array([float(hull.position[0]), float(hull.position[1])])
        wheels = raw.car.wheels
        rim = [w.omega * w.wheel_rad for w in wheels]
        i = self._nearest(pos)
        tdx, tdy = self.track_dir[i]
        off = (pos - self.track[i]) @ np.array([-tdy, tdx])  # + is left of the centre line
        base = [
            math.hypot(vx, vy) / 60.0, (vx * fx + vy * fy) / 60.0, (vx * lx + vy * ly) / 60.0,
            float(hull.angularVelocity) / 3.0, float(wheels[0].joint.angle) / 0.4,
            *wheels_on_road(raw), (rim[0] + rim[1]) / 120.0, (rim[2] + rim[3]) / 120.0,
            off / (40 / 6.0), fx * tdy - fy * tdx, fx * tdx + fy * tdy,
        ]
        n = len(self.track)
        ahead = self.track[[(i + k) % n for k in self.cfg.lookahead]] - pos
        look = np.stack([ahead @ np.array([fx, fy]), ahead @ np.array([lx, ly])], 1).reshape(-1) / 60.0
        ang = a + math.pi / 2 + self.ray_angles
        dirs = np.ascontiguousarray(np.stack([np.cos(ang), np.sin(ang)], 1).reshape(-1))
        rays = self.cast(self.tiles, self.centers, pos[0], pos[1], dirs, self.cfg.lidar_step, self.cfg.lidar_max_distance)
        return np.concatenate([np.array(base), look, rays]).astype(np.float32)
