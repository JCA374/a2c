"""Rendering for replays, the Play tab and films. Drawing only: no policy imports this module."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env.config import EnvConfig
from env.make_env import make_env

try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    FONT_BIG = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
except OSError:  # pragma: no cover
    FONT = FONT_BIG = ImageFont.load_default()


def outcome(info, steps) -> str:
    if info.get("lap"):
        return f"LAP FINISHED in {steps} frames"
    if steps >= 1000:
        return f"time up: {100 * info['tiles']:.0f}% of the track"
    return f"left the playfield at frame {steps}"


def world_to_screen(raw, pts):
    """CarRacing's camera (car_racing.py _render): zoom, rotate by -hull.angle, car at (W/2, H/4) from the bottom."""
    from gymnasium.envs.box2d.car_racing import SCALE, VIDEO_H, VIDEO_W, WINDOW_H, WINDOW_W, ZOOM
    t = raw.t
    zoom = 0.1 * SCALE * max(1 - t, 0) + ZOOM * SCALE * min(t, 1)
    a = -raw.car.hull.angle
    cx, cy = raw.car.hull.position
    p = (np.asarray(pts, float) - [cx, cy]) * zoom
    rx = p[:, 0] * math.cos(a) - p[:, 1] * math.sin(a)
    ry = p[:, 0] * math.sin(a) + p[:, 1] * math.cos(a)
    # drawn on the 1000x800 window, flipped vertically, then scaled to the 600x400 rgb_array frame
    return np.stack([(WINDOW_W / 2 + rx) * VIDEO_W / WINDOW_W, (WINDOW_H - (WINDOW_H / 4 + ry)) * VIDEO_H / WINDOW_H], 1)


def draw_rays(img, env, obs):
    """The lidar the agent sees, drawn from the car to where each ray left the road."""
    w = env  # TrackObservation is the outermost wrapper
    raw = env.unwrapped
    hull = raw.car.hull
    a = hull.angle + math.pi / 2 + w.ray_angles
    d = obs[-len(w.ray_angles):] * w.cfg.lidar_max_distance
    ends = np.stack([hull.position[0] + np.cos(a) * d, hull.position[1] + np.sin(a) * d], 1)
    s = world_to_screen(raw, np.vstack([[hull.position[0], hull.position[1]], ends]))
    dr = ImageDraw.Draw(img, "RGBA")
    for e in s[1:]:
        dr.line([tuple(s[0]), tuple(e)], fill=(80, 220, 255, 120), width=1)
        dr.ellipse([e[0] - 2, e[1] - 2, e[0] + 2, e[1] + 2], fill=(80, 220, 255, 220))
    ahead = env.track[[(env.idx + k) % len(env.track) for k in env.cfg.lookahead]]
    for p in world_to_screen(raw, ahead):
        dr.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], outline=(255, 210, 60, 230), width=2)


def episode_frames(agent, seed, every=1, size=(600, 400), rays=False, label=None, max_frames=None):
    """Drive one episode under the score rules; returns (PIL frames, result dict). Last frame carries the outcome."""
    cfg = EnvConfig(training=False, render_mode="rgb_array")
    env = make_env(cfg)
    obs, _ = env.reset(seed=seed)
    frames, steps, info = [], 0, {"score": 0.0, "tiles": 0.0, "lap": False}
    while True:
        if steps % every == 0:
            img = Image.fromarray(env.render())
            if rays:
                draw_rays(img, env, obs)
            dr = ImageDraw.Draw(img)
            dr.text((10, 8), f"score {info['score']:.0f}   frame {steps}", font=FONT, fill=(255, 255, 255))
            if label:
                dr.text((10, 28), label, font=FONT, fill=(255, 230, 120))
            frames.append(img.resize(size) if img.size != size else img)
        obs, _, term, trunc, info = env.step(agent.drive(obs))
        steps += 1
        if term or trunc or (max_frames and steps >= max_frames):
            break
    img = Image.fromarray(env.render())
    dr = ImageDraw.Draw(img, "RGBA")
    dr.rectangle([0, 150, 600, 250], fill=(0, 0, 0, 170))
    dr.text((300, 180), outcome(info, steps), font=FONT_BIG, fill=(255, 255, 255), anchor="mm")
    dr.text((300, 222), f"score {info['score']:.1f}", font=FONT_BIG, fill=(255, 220, 90), anchor="mm")
    frames.append(img.resize(size) if img.size != size else img)
    env.close()
    return frames, {"score": float(info["score"]), "lap": bool(info["lap"]), "tiles": float(info["tiles"]), "steps": steps}
