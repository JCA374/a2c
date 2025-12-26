# Environment, Rays, and Competition Tracks Summary

This document summarizes the CarRacing environment stack, the compact observation (including lidar-style rays), and the competition track seeds used in this repo.

## Environment stack (core wrapper order)

The environment is built in `env/make_env.py` and applies a fixed wrapper stack:

1. `gym.make("CarRacing-v2", render_mode=..., domain_randomize=...)`
2. `ActionClipWrapper` (clips actions to env bounds)
3. `CarRacingRewardWrapper` (replaces reward with competition shaping)
4. `gym.wrappers.TimeLimit` if `env.episode.max_steps` is set
5. `EpisodeTimeLimit` if `env.episode.max_seconds` is set
6. `CompactObservation` if `env.compact_state.enabled` is true

The wrapper order matters: reward shaping happens on the raw env before compact observations are built.

## Compact observation vector

When `CompactObservation` is enabled, the agent receives a 1D float vector of size `8 + lidar_rays`:

Base dynamics (8 values, all float32):
- `speed`: total speed magnitude, scaled by 50.0 and clipped to [0, 1]
- `local_vx`: x-velocity in car frame, scaled by 50.0 and clipped to [-1, 1]
- `local_vy`: y-velocity in car frame, scaled by 50.0 and clipped to [-1, 1]
- `cos_heading`: cosine of car heading
- `sin_heading`: sine of car heading
- `angular_velocity`: scaled by 5.0 and clipped to [-1, 1]
- `mean_steer`: mean wheel steer angle, normalized by steer limit 0.8 and clipped to [-1, 1]
- `traction_fraction`: mean wheel traction/contact, clipped to [0, 1]

Lidar-style distances (default 32 values):
- Each value is the normalized distance to the road edge along a ray direction.
- Values are clipped to [0, 1] after dividing by `lidar_max_distance`.

## Lidar ray geometry

The ray sampler is implemented in `env/wrappers.py`:

- `lidar_rays` (default: 32)
- `lidar_max_distance` (default: 45.0)
- `lidar_step` (default: 3.0)
- Ray angles are evenly spaced from `-0.9*pi` to `0.9*pi` relative to car heading.
- Each ray steps forward by `lidar_step` until it leaves the road polygon or reaches `lidar_max_distance`.
- If road geometry is missing for a step, the ray distances default to 1.0 (max distance).

## Competition tracks (seeded)

Competition tracks are defined by fixed random seeds that generate deterministic CarRacing tracks:

- Train seeds: 0..19 (20 tracks)
- Validation seeds: 101..105 (5 tracks)
- Test seeds: 2024..2033 (10 tracks)

These are defined in `env/tracks.py` and also reflected in `configs/competition.yaml` under `seeds`.

## Episode limits and pass criteria

Default competition limits in `configs/competition.yaml`:
- `env.episode.max_steps`: 1000
- `env.episode.max_seconds`: 60
- Pass criteria: `require_finished: true`, `max_steps: 1000`

## Reward shaping (competition defaults)

Reward shaping is applied by `CarRacingRewardWrapper` using `env.reward` config:
- `progress_scale`: 2.0
- `off_track_penalty`: -40.0
- `traction_threshold`: 0.95
- `collision_penalty`: -20.0
- `control_penalty`: 0.01
- `finish_bonus`: 400.0
- `time_penalty`: 0.05

Adjust these in `configs/competition.yaml` or another config to change training/validation behavior.
