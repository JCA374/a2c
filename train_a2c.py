from __future__ import annotations

import argparse
import os
import random
from functools import partial
from typing import List, Optional

import numpy as np
import torch

from a2c_agent import A2CConfig, A2CTrainer
from env.config import EnvConfig
from env.make_env import make_env
from env.tracks import TRAIN_SEEDS, seeded_cycle
from env.vector_env import SimpleVectorEnv
from log_utils import TrainingLogger
from pathlib import Path
from dataclasses import asdict
from datetime import datetime


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_auto_positive_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise argparse.ArgumentTypeError("Expected a positive integer or 'auto'.")
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a positive integer or 'auto'.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer or 'auto'.")
    return parsed


def detect_cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def resolve_num_envs(num_envs: Optional[int], cpu_count: Optional[int] = None) -> int:
    return num_envs if num_envs is not None else (cpu_count or detect_cpu_count())


def resolve_cpu_threads(cpu_threads: Optional[int], cpu_count: Optional[int] = None) -> int:
    return cpu_threads if cpu_threads is not None else (cpu_count or detect_cpu_count())


def configure_torch_threads(cpu_threads: int) -> int:
    threads = max(1, int(cpu_threads))
    torch.set_num_threads(threads)
    return threads


def estimate_rollout_buffer_bytes(rollout_length: int, num_envs: int, obs_dim: int, action_dim: int) -> int:
    buffer_scalars = 6  # rewards, dones, values, log_probs, advantages, returns
    per_step = obs_dim + action_dim + buffer_scalars
    return int(rollout_length) * int(num_envs) * per_step * 4


def adjust_rollout_length_for_ram(
    rollout_length: int,
    num_envs: int,
    obs_dim: int,
    action_dim: int,
    max_ram_mb: Optional[int],
) -> int:
    if max_ram_mb is None or max_ram_mb <= 0:
        return rollout_length
    max_bytes = int(max_ram_mb) * 1024 * 1024
    per_step_bytes = int(num_envs) * (obs_dim + action_dim + 6) * 4
    if per_step_bytes <= 0:
        return rollout_length
    max_steps = max(1, int(max_bytes // per_step_bytes))
    return min(int(rollout_length), max_steps)


def build_envs(num_envs: int, env_config: EnvConfig, base_seed: int) -> SimpleVectorEnv:
    seed_iter = seeded_cycle(TRAIN_SEEDS)
    # Burn through seeds to get deterministic offset.
    for _ in range(base_seed % len(TRAIN_SEEDS)):
        next(seed_iter)
    env_fns: List = [partial(make_env, env_config, next(seed_iter)) for _ in range(num_envs)]
    return SimpleVectorEnv(env_fns)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an A2C agent on CarRacing with compact observations.")
    parser.add_argument("--num-envs", type=parse_auto_positive_int, default="auto",
                        help="Number of parallel environments or 'auto' (default: auto).")
    parser.add_argument("--cpu-threads", type=parse_auto_positive_int, default="auto",
                        help="Torch CPU threads or 'auto' (default: auto).")
    parser.add_argument("--total-steps", type=int, default=500_000, help="Total environment steps across all envs.")
    parser.add_argument("--rollout-length", type=int, default=128, help="Number of steps per rollout before an update.")
    parser.add_argument("--log-interval", type=int, default=10, help="Updates between console logs.")
    parser.add_argument("--max-ram-mb", type=int, default=None,
                        help="Optional rollout buffer RAM cap in MB (reduces rollout length if needed).")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="CarRacing render_mode (rgb_array recommended).")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for metrics/config logs. Defaults to logs/a2c_<timestamp>.")
    args = parser.parse_args()
    args.num_envs = parse_auto_positive_int(args.num_envs)
    args.cpu_threads = parse_auto_positive_int(args.cpu_threads)
    return args


def main():
    args = parse_args()
    set_global_seed(args.seed)
    cpu_count = detect_cpu_count()
    num_envs = resolve_num_envs(args.num_envs, cpu_count)
    cpu_threads = resolve_cpu_threads(args.cpu_threads, cpu_count)
    cpu_threads = configure_torch_threads(cpu_threads)

    log_dir = Path(args.log_dir) if args.log_dir else Path("logs") / f"a2c_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    env_config = EnvConfig(render_mode=args.render_mode)
    envs = build_envs(num_envs, env_config, base_seed=args.seed)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    rollout_length = adjust_rollout_length_for_ram(
        args.rollout_length, num_envs, obs_dim, action_dim, args.max_ram_mb
    )
    if rollout_length != args.rollout_length:
        before_mb = estimate_rollout_buffer_bytes(args.rollout_length, num_envs, obs_dim, action_dim) / (1024 * 1024)
        after_mb = estimate_rollout_buffer_bytes(rollout_length, num_envs, obs_dim, action_dim) / (1024 * 1024)
        print(
            f"Adjusted rollout_length from {args.rollout_length} to {rollout_length} "
            f"to fit ~{args.max_ram_mb} MB buffer cap ({before_mb:.1f} -> {after_mb:.1f} MB)."
        )

    training_cfg = A2CConfig(
        total_steps=args.total_steps,
        rollout_length=rollout_length,
        log_interval=args.log_interval,
    )

    logger = None
    try:
        logger = TrainingLogger(
            log_dir,
            config={
                "seed": args.seed,
                "num_envs": num_envs,
                "cpu_threads": cpu_threads,
                "max_ram_mb": args.max_ram_mb,
                "env_config": asdict(env_config),
                "training_config": asdict(training_cfg),
            },
        )

        trainer = A2CTrainer(envs, training_cfg, logger=logger)
        trainer.train()
    finally:
        envs.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
