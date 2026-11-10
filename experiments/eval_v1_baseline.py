"""Run with the v1 code (git d8ea104, before the rewrite): it imports v1 play_model.py.
"""Score the v1 models (Dec 2025) with the v1 code: CarRacing's own reward, deterministic, 100 held-out tracks."""
import json, sys, numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, '.')
torch.set_num_threads(1)
from env.config import EnvConfig
from env.make_env import make_env
from play_model import load_policy
SEEDS = list(range(5000, 5100))
def run(args):
    path, seed = args
    env = make_env(EnvConfig(), seed=None)
    obs, _ = env.reset(seed=seed)
    pol = load_policy(path, env.observation_space.shape, env.action_space.low, env.action_space.high, 'cpu')
    raw = env.unwrapped; score = 0.0; shaped = 0.0; steps = 0
    prev = 0.0
    while True:
        with torch.no_grad():
            a = pol.deterministic_action(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy()
        obs, r, term, trunc, info = env.step(a)
        shaped += r; steps += 1
        if term or trunc: break
    score = raw.reward + (-100 if info.get('lap_finished') is False and term else 0)
    tiles = raw.tile_visited_count / len(raw.track)
    return dict(seed=seed, score=float(score), shaped=float(shaped), steps=steps, tiles=float(tiles),
                lap=bool(info.get('lap_finished', False)))
if __name__ == '__main__':
    out = {}
    with Pool(7) as p:
        for name in ['fixed_reward_balance_fast', 'stable_lr', 'fixed2']:
            rows = p.map(run, [(f'logs/{name}/best_model.pt', s) for s in SEEDS])
            sc = np.array([r['score'] for r in rows])
            out[name] = dict(n=len(rows), mean=float(sc.mean()), std=float(sc.std()), median=float(np.median(sc)),
                             max=float(sc.max()), tiles=float(np.mean([r['tiles'] for r in rows])),
                             laps=sum(r['lap'] for r in rows), rows=rows)
            print(name, out[name]['mean'], out[name]['max'], out[name]['tiles'], flush=True)
    json.dump(out, open('results/baseline_v1_raw.json', 'w'), indent=1)
