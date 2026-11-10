"""Describe the Play tab's fixed tracks from their geometry -> experiments/play_tracks.json."""
import json, math, numpy as np, gymnasium as gym
from env.tracks import TEST_SEEDS

out = []
for seed in TEST_SEEDS:
    env = gym.make("CarRacing-v3").unwrapped
    env._render = lambda mode: None
    env.reset(seed=seed)
    beta = np.unwrap(np.array([t[1] for t in env.track]))
    n = len(beta)
    turn10 = np.abs(np.roll(beta, -10) - beta)[: n - 10]            # heading change over 10 tiles (~35 m)
    sharp = float(np.degrees(turn10.max()))
    bends = int(np.sum(np.diff((np.degrees(turn10) > 60).astype(int)) == 1))
    hair = int(np.sum(np.diff((np.degrees(turn10) > 110).astype(int)) == 1))
    straight = int(max(len(s) for s in "".join("s" if d < 8 else "x" for d in np.degrees(turn10)).split("x")))
    out.append(dict(id=f"track{seed}", seed=seed, tiles=n, sharpest=round(sharp), bends=bends, hairpins=hair, straight=straight))
lens = sorted(o["tiles"] for o in out)
for o in out:
    p = []
    p.append("short" if o["tiles"] <= lens[2] else "long" if o["tiles"] >= lens[-3] else "medium-length")
    p[0] += f" ({o['tiles']} tiles)"
    if o["hairpins"]:
        p.append(f"{o['hairpins']} hairpin{'s' if o['hairpins'] > 1 else ''}")
    p.append(f"{o['bends']} sharp bend{'s' if o['bends'] != 1 else ''}, the sharpest turns {o['sharpest']} degrees in 35 m")
    p.append(f"longest straight {o['straight']} tiles")
    o["title"] = f"Track {o['seed']}"
    o["desc"] = "; ".join(p)
json.dump(out, open("experiments/play_tracks.json", "w"), indent=1)
for o in out:
    print(o["title"], "-", o["desc"])
