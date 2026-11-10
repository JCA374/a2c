# Drive A2C: teach a car to race CarRacing-v3 from 49 numbers

**🌐 Live site: https://jca374.github.io/a2c/**

An actor-critic agent (A2C, and PPO on the same network) for Gymnasium's `CarRacing-v3`. Instead of pixels it sees a
compact state: speed and grip, where the centre line goes over the next 40 tiles, and 19 lidar rays to the road edge.

**Goal:** a mean score of **900** on 100 unseen tracks (seeds 5000-5099), deterministic driving, CarRacing's own score
(+1000/N per tile, -0.1 per frame, 1000 frames). The first version (Dec 2025) scored **-42.9**: its lidar read 0 on
every ray. The live site shows the log, the curriculum, how the agent sees, the model, clips of it driving, speed
measurements, lessons and a podcast.

## Use it

```bash
python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train_a2c.py --run my_run            # PPO, 6 M steps by default; runs/my_run/
.venv/bin/python train_a2c.py --algo a2c --run a2c    # plain A2C with the same fixes
.venv/bin/python train_a2c.py --resume runs/my_run    # continue (SIGTERM / Ctrl-C saves first)
.venv/bin/python play.py runs/my_run/best.pt          # watch it drive in a window
.venv/bin/python evaluate.py runs/my_run/best.pt      # the goal check: 100 unseen tracks -> results/
.venv/bin/python -m pytest                            # tests (incl. fast env == reference env, bit for bit)
```

## Layout

| Path | What |
|---|---|
| `env/` | CarRacing wrappers: the 49-number observation, the score and training shaping, the lidar (numba + Python reference) |
| `a2c_agent.py` | actor-critic network, rollout buffer, GAE, the A2C and PPO updates, observation normalisation |
| `train_a2c.py` | training loop: parallel envs, validation on 16 fixed tracks, curriculum, run logs for the site |
| `evaluate.py`, `play.py` | the goal check and a viewer |
| `viz/` | drawing for replays, clips and films (no policy imports it) |
| `experiments/` | the lidar bake-off (Python, NumPy, numba, C++, Rust, Go), the v1 baseline, journal |
| `publish.py`, `site.json`, `site_hooks.py`, `docs/` | the GitHub Pages site, rebuilt every 15 minutes by `publish_loop.sh` |
