"""The goal check: one model on the 100 goal tracks (seeds 5000-5099), deterministic, CarRacing's own rules.

Runs in the reference env (pixels drawn, pure-Python lidar), never the training fast path, and writes
results/goal_<name>.json for the site.   python evaluate.py runs/<run>/best.pt [--name r1_best] [--workers 4]"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from env.tracks import GOAL_SEEDS

_AGENT = {}


def _episode(args):
    path, seed, reference = args
    import torch
    torch.set_num_threads(1)
    from a2c_agent import Agent
    from env.config import EnvConfig
    from env.make_env import make_env
    from train_a2c import run_episode
    if path not in _AGENT:
        _AGENT[path] = Agent.load(path)
    cfg = EnvConfig(training=False, render_pixels=reference)
    cfg.obs.lidar_impl = "python" if reference else "numba"
    return run_episode(_AGENT[path], make_env(cfg), seed)


def evaluate(path, seeds, workers, reference=True):
    with Pool(workers) as p:
        return p.map(_episode, [(path, s, reference) for s in seeds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--name", default=None, help="results/goal_<name>.json (default: <run>_<file>)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fast", action="store_true", help="use the training fast path (for quick looks, not the goal)")
    a = ap.parse_args()
    t0 = time.time()
    rows = evaluate(a.model, GOAL_SEEDS, a.workers, reference=not a.fast)
    s = np.array([r["score"] for r in rows])
    half = 1.96 * s.std(ddof=1) / np.sqrt(len(s))
    parts = os.path.normpath(a.model).split(os.sep)
    name = a.name or f"{parts[-2]}_{os.path.splitext(parts[-1])[0]}"
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "label": a.label or name, "policy": a.model, "n": len(rows),
           "mean": float(s.mean()), "ci": [float(s.mean() - half), float(s.mean() + half)], "median": float(np.median(s)),
           "min": float(s.min()), "max": float(s.max()), "laps": float(np.mean([r["lap"] for r in rows])),
           "tiles": float(np.mean([r["tiles"] for r in rows])), "share_900": float((s >= 900).mean()),
           "env": "reference" if not a.fast else "fast", "seconds": round(time.time() - t0),
           "note": "100 unseen tracks (5000-5099), deterministic; 95% interval of the mean (normal approx.)",
           "rows": rows}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/goal_{name}.json", "w"), indent=1)
    print(f"{name}: mean {out['mean']:.1f} (95% {out['ci'][0]:.1f}-{out['ci'][1]:.1f}), median {out['median']:.1f}, "
          f"laps {out['laps']:.0%}, >=900 {out['share_900']:.0%}, worst {out['min']:.1f}")


if __name__ == "__main__":
    main()
