# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

An actor-critic agent (A2C, and PPO on the same network; PPO is the default) that learns Gymnasium's `CarRacing-v3`
from a 49-number observation instead of pixels. Goal: mean score >= 900 on 100 unseen tracks (seeds 5000-5099),
deterministic, CarRacing's own reward, 1000 frames. The v1 code (git d8ea104, Dec 2025) scored -42.9: its lidar
always read 0. Live site: https://jca374.github.io/a2c/ (GitHub Pages from `main:/docs`).

## Commands

```bash
.venv/bin/python train_a2c.py --run NAME                 # PPO, 6M steps; --algo a2c for A2C; --resume runs/NAME
.venv/bin/python evaluate.py runs/NAME/best.pt           # goal check, reference env -> results/goal_*.json
.venv/bin/python play.py runs/NAME/best.pt               # watch in a window
.venv/bin/python -m pytest -q tests                      # ~30 s
.venv/bin/python publish.py [--push]                     # rebuild the site (publish_loop.sh does this every 15 min)
.venv/bin/python video/build_film.py howto|explainer     # films (needs voice/film-<name> exported)
```
The venv was moved from another path: `.venv/bin/pip` has a stale shebang, use `.venv/bin/python -m pip`.

## Layout

- `env/make_env.py`: CarRacing-v3 (unwrapped) -> `skip_state_pixels` (training) -> `TwoAxisAction` -> `ScoreAndShaping`
  -> `TimeLimit(1000)` -> `TrackObservation`. `score_config()` / `EnvConfig(training=False)` = the task's own rules.
- `env/wrappers.py`: `TrackObservation` (49 numbers; docstring lists them), `ScoreAndShaping` (`info["score"]` is the
  metric; training-only early stops), `TwoAxisAction` (steer, pedal).
- `env/lidar.py`: `cast_python` is the reference; `cast_numba` compiles the same source. Must stay bit-identical
  (`tests/test_lidar.py`). Road tiles come from `body.fixtures[0].shape.vertices` of `env.road`.
- `env/tracks.py`: validation 100-115 (selection), test 2024-2033 (Play tab), goal 5000-5099 (goal checks only).
- `a2c_agent.py`: `ActorCritic`, `RolloutBuffer` (GAE), `update()` (A2C or PPO), `Agent` (net + obs normalisation;
  `save`/`load` hold everything needed to drive).
- `train_a2c.py`: `STAGES` curriculum, AsyncVectorEnv (fork, SAME_STEP autoreset, `final_obs` bootstrap on truncation),
  validation pool, run files in `runs/<run>/` (stats.csv, events.jsonl, curriculum.json, best.pt, champions/).
- `viz/render.py`: drawing for replays/clips/films; no policy imports it.
- `experiments/bakeoff/`: lidar kernel in Python, NumPy, numba, C++, Rust, Go (`./build.sh && python bench.py`).
- Site: `publish.py` (generic, from the ml-project-site skill), `site.json`, `site_hooks.py`, `docs/` (pages; `docs/app.js`
  carries a small project patch for score-valued goals), `podcast.conf` + `podcast/` (project-podcast skill).

## Rules

- The reference env (pixels drawn, `lidar_impl="python"`) is the judge: goal checks run in it (`evaluate.py`).
  The fast path is for training only and is tested equal to it (`tests/test_equivalence.py`).
- Report only CarRacing's own score. Never quote the training return as the score.
- `pkill -f`/`pgrep -f` with a pattern that also appears in the same shell command kills that shell: use `[t]rain_a2c`.
