"""Train an A2C / PPO agent on CarRacing-v3 from the 49-number track observation.

    python train_a2c.py                                  # PPO, 5M steps, runs/<timestamp>/
    python train_a2c.py --algo a2c --run a2c_test        # plain A2C with the same fixes
    python train_a2c.py --resume runs/<run>              # continue after a stop (SIGTERM saves first)

Writes runs/<run>/: config.json, curriculum.json, stats.csv (one row per update), events.jsonl, latest.pt,
best.pt (best validation score) and champions/ (every new best, for replays)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import time
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from a2c_agent import A2CConfig, Agent, RolloutBuffer, update
from env.config import EnvConfig, EpisodeConfig
from env.make_env import make_env, score_config
from env.tracks import VALIDATION_SEEDS

# Stage 1 stops hopeless training episodes early (off the road for 1 s, or 3 s without a new tile).
# Stage 2 trains under the task's own rules: every episode runs its full 1000 frames.
STAGES = [
    {"name": "Learn the road", "world": "random tracks, early stops on", "reward": "CarRacing's own (+1000/N per tile, -0.1/frame), -50 on an early stop",
     "pass": "validation mean >= 700", "pass_score": 700.0, "cap": 3_000_000, "expect": "300-700 within 1-2M steps",
     "early_stops": True},
    {"name": "Race", "world": "random tracks, the task's own rules (1000 frames, no early stops)", "reward": "CarRacing's own",
     "pass": "validation mean >= 900", "pass_score": 900.0, "cap": None, "expect": "850-920",
     "early_stops": False},
]


def env_config_for(stage: dict) -> EnvConfig:
    if stage["early_stops"]:
        return EnvConfig()
    return EnvConfig(episode=EpisodeConfig(off_road_patience=None, no_progress_patience=None))


def build_envs(num_envs: int, cfg: EnvConfig, seed: int):
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode
    fns = [partial(make_env, cfg, seed * 1000 + i) for i in range(num_envs)]
    return AsyncVectorEnv(fns, context="fork", autoreset_mode=AutoresetMode.SAME_STEP)


# ---- validation: fixed tracks, deterministic policy, CarRacing's own rules -------------------------------------

_EVAL_AGENT = None


def _eval_init():
    torch.set_num_threads(1)


def _eval_episode(args):
    path, seed, impl = args
    global _EVAL_AGENT
    if _EVAL_AGENT is None or _EVAL_AGENT[0] != (path, os.path.getmtime(path)):
        _EVAL_AGENT = ((path, os.path.getmtime(path)), Agent.load(path))
    agent = _EVAL_AGENT[1]
    cfg = score_config()
    cfg.obs.lidar_impl = impl
    return run_episode(agent, make_env(cfg), seed)


def run_episode(agent, env, seed, on_step=None):
    obs, _ = env.reset(seed=seed)
    steps = 0
    while True:
        obs, _, term, trunc, info = env.step(agent.drive(obs))
        steps += 1
        if on_step:
            on_step(env, info)
        if term or trunc:
            break
    return {"seed": seed, "score": float(info["score"]), "tiles": float(info["tiles"]), "lap": bool(info["lap"]),
            "steps": steps}


def evaluate(pool, path, seeds, impl="numba"):
    return pool.map(_eval_episode, [(str(path), s, impl) for s in seeds])


# ---- training ------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    ap.add_argument("--total-steps", type=int, default=5_000_000)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--rollout-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=100_000, help="env steps between validation runs")
    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default=None, help="run name (default: timestamp)")
    ap.add_argument("--resume", default=None, help="runs/<run> to continue")
    args = ap.parse_args()

    if args.resume:
        run_dir = Path(args.resume)
        saved = json.loads((run_dir / "config.json").read_text())
        cfg = A2CConfig(**saved["agent"])
        args.num_envs, args.seed, args.eval_every = saved["num_envs"], saved["seed"], saved["eval_every"]
    else:
        run_dir = Path("runs") / (args.run or datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_dir.mkdir(parents=True, exist_ok=False)
        cfg = A2CConfig(algo=args.algo, total_steps=args.total_steps, rollout_length=args.rollout_length,
                        learning_rate=args.lr)
        (run_dir / "config.json").write_text(json.dumps(
            {"agent": asdict(cfg), "env": asdict(EnvConfig()), "num_envs": args.num_envs, "seed": args.seed,
             "eval_every": args.eval_every, "validation_seeds": VALIDATION_SEEDS}, indent=1))
        (run_dir / "curriculum.json").write_text(json.dumps(
            {"note": "Stages switch when the validation mean (16 fixed tracks, deterministic, CarRacing's own rules) "
                     "passes, or at the cap.", "stages": STAGES}, indent=1))
    (run_dir / "champions").mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)

    def event(kind, **kw):
        with open(run_dir / "events.jsonl", "a") as f:
            f.write(json.dumps({"t": datetime.now().isoformat(timespec="seconds"), "kind": kind, **kw}) + "\n")

    probe = make_env(EnvConfig())
    obs_dim, act_dim = probe.observation_space.shape[0], probe.action_space.shape[0]
    low, high = probe.action_space.low, probe.action_space.high
    probe.close()
    agent = Agent(obs_dim, act_dim, cfg, low, high)
    opt = torch.optim.Adam(agent.net.parameters(), lr=cfg.learning_rate, eps=1e-5)
    ret_rms, state = None, {"step": 0, "update": 0, "stage": 0, "best": -1e9, "wall": 0.0, "episodes": 0}
    from a2c_agent import RunningMeanStd
    ret_rms = RunningMeanStd()
    if args.resume:
        ck = torch.load(run_dir / "latest.pt", weights_only=False)
        agent.net.load_state_dict(ck["net"])
        agent.obs_rms.load(ck["obs_rms"])
        opt.load_state_dict(ck["opt"])
        ret_rms.load(ck["ret_rms"])
        state = ck["state"]
        event("resume", step=state["step"])
    else:
        event("run_start", algo=cfg.algo, total_steps=cfg.total_steps, num_envs=args.num_envs, obs_dim=obs_dim)

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    def save_latest():
        agent.save(run_dir / "latest.pt", {"opt": opt.state_dict(), "ret_rms": ret_rms.state(), "state": state})

    stage = STAGES[state["stage"]]
    if not args.resume or not (run_dir / "stats.csv").exists():
        event("stage_start", stage=stage["name"], expect=stage["expect"], step=state["step"])
    envs = build_envs(args.num_envs, env_config_for(stage), args.seed + state["update"])
    pool = get_context("fork").Pool(args.eval_workers, initializer=_eval_init)
    buf = RolloutBuffer(cfg.rollout_length, args.num_envs, obs_dim, act_dim)
    stats_path = run_dir / "stats.csv"
    writer, stats_file = None, open(stats_path, "a", newline="")
    recent = deque(maxlen=40)
    disc_ret = np.zeros(args.num_envs)
    obs, _ = envs.reset(seed=args.seed + state["update"])
    agent.obs_rms.update(obs) if state["step"] == 0 else None
    next_eval = (state["step"] // args.eval_every + 1) * args.eval_every
    total_updates = cfg.total_steps // (cfg.rollout_length * args.num_envs)
    t_start = time.time() - state["wall"]

    while state["step"] < cfg.total_steps and not stop["now"]:
        t0 = time.time()
        frac = min(1.0, state["step"] / cfg.total_steps)
        lr = cfg.learning_rate * (1 - (1 - cfg.lr_final_fraction) * frac)
        for g in opt.param_groups:
            g["lr"] = lr
        buf.step = 0
        for _ in range(cfg.rollout_length):
            x = agent.norm(obs)
            with torch.no_grad():
                action, raw, lp, v = agent.net.act(x)
            nobs, rew, term, trunc, info = envs.step(action.numpy())
            rew = np.asarray(rew, np.float64)
            # reward scaling by the running std of the discounted return (VecNormalize)
            disc_ret = disc_ret * cfg.gamma + rew
            ret_rms.update(disc_ret)
            disc_ret[term | trunc] = 0.0
            r = rew / np.sqrt(ret_rms.var + 1e-8) if cfg.normalize_reward else rew
            done = term | trunc
            if "final_obs" in info:
                # time-limit cut: the episode did not end, so bootstrap with the value of the state it was cut in
                cut = trunc & ~term & info.get("_final_obs", done)
                if cut.any():
                    fo = np.stack([info["final_obs"][i] for i in np.where(cut)[0]])
                    with torch.no_grad():
                        fv = agent.net.value(agent.norm(fo)).numpy()
                    r[cut] += cfg.gamma * fv
                fi = info["final_info"]
                for i in np.where(done)[0]:
                    recent.append({"score": float(fi["score"][i]), "len": int(fi["episode_steps"][i]) if "episode_steps" in fi else 0,
                                   "lap": bool(fi["lap"][i]), "early": fi["early_stop"][i] is not None,
                                   "tiles": float(fi["tiles"][i])})
                    state["episodes"] += 1
            buf.add(x, raw, torch.as_tensor(r, dtype=torch.float32), torch.as_tensor(done, dtype=torch.float32), v, lp)
            obs = nobs
            agent.obs_rms.update(obs)
            state["step"] += args.num_envs
        t_collect = time.time() - t0
        with torch.no_grad():
            last_v = agent.net.value(agent.norm(obs))
        buf.compute_returns_and_advantages(last_v, cfg.gamma, cfg.gae_lambda)
        losses = update(agent, opt, buf, cfg)
        state["update"] += 1
        dt = time.time() - t0
        state["wall"] = time.time() - t_start

        val = None
        if state["step"] >= next_eval or state["step"] >= cfg.total_steps:
            next_eval += args.eval_every
            save_latest()
            res = evaluate(pool, run_dir / "latest.pt", VALIDATION_SEEDS)
            val = float(np.mean([r["score"] for r in res]))
            check = {"score": val, "laps": float(np.mean([r["lap"] for r in res])),
                     "tiles": float(np.mean([r["tiles"] for r in res])), "worst": float(min(r["score"] for r in res)), "n": len(res)}
            event("val_check", step=state["step"], check=check)
            if val > state["best"]:
                state["best"] = val
                champ = run_dir / "champions" / f"step_{state['step']:09d}.pt"
                agent.save(champ, {"val_score": val, "step": state["step"]})
                agent.save(run_dir / "best.pt", {"val_score": val, "step": state["step"]})
                event("goal_best", step=state["step"], champion=str(champ), check=check)
            hit_cap = stage["cap"] is not None and state["step"] >= stage["cap"]
            if (val >= stage["pass_score"] or hit_cap) and state["stage"] + 1 < len(STAGES):
                event("stage_pass" if val >= stage["pass_score"] else "stage_cap", stage=stage["name"], step=state["step"], check=check)
                state["stage"] += 1
                stage = STAGES[state["stage"]]
                event("stage_start", stage=stage["name"], expect=stage["expect"], step=state["step"])
                envs.close()
                envs = build_envs(args.num_envs, env_config_for(stage), args.seed + state["update"])
                obs, _ = envs.reset(seed=args.seed + state["update"])
                disc_ret[:] = 0

        row = {"update": state["update"], "step": state["step"], "wall_s": round(state["wall"], 1),
               "steps_per_s": round(cfg.rollout_length * args.num_envs / dt, 1),
               "collect_frac": round(t_collect / dt, 3), "stage": stage["name"], "lr": lr,
               "train_score": np.mean([e["score"] for e in recent]) if recent else "",
               "train_tiles": np.mean([e["tiles"] for e in recent]) if recent else "",
               "train_laps": np.mean([e["lap"] for e in recent]) if recent else "",
               "early_stops": np.mean([e["early"] for e in recent]) if recent else "",
               "episodes": state["episodes"], "val_score": "" if val is None else val,
               "best_val": state["best"] if state["best"] > -1e9 else "", **losses}
        if writer is None:
            writer = csv.DictWriter(stats_file, fieldnames=list(row))
            if stats_path.stat().st_size == 0:
                writer.writeheader()
        writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        stats_file.flush()
        if state["update"] % 10 == 0 or val is not None:
            print(f"upd {state['update']}/{total_updates} step {state['step']:,} sps {row['steps_per_s']:.0f} "
                  f"train {row['train_score'] if row['train_score'] == '' else round(row['train_score'], 1)} "
                  f"val {'' if val is None else round(val, 1)} best {state['best']:.1f} std {losses['std']:.3f} "
                  f"kl {losses['kl']:.4f} ev {losses['explained_var']:.2f}", flush=True)

    save_latest()
    event("stopped" if stop["now"] else "run_end", step=state["step"], best_val=state["best"])
    envs.close()
    pool.close()
    stats_file.close()


if __name__ == "__main__":
    main()
