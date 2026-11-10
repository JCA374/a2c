"""Watch a model drive in a window:  python play.py runs/<run>/best.pt [--seed 2024] [--episodes 3]"""
import argparse

from a2c_agent import Agent
from env.config import EnvConfig
from env.make_env import make_env
from train_a2c import run_episode

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--seed", type=int, default=2024)
ap.add_argument("--episodes", type=int, default=3)
a = ap.parse_args()
agent = Agent.load(a.model)
env = make_env(EnvConfig(training=False, render_mode="human"))
for i in range(a.episodes):
    r = run_episode(agent, env, a.seed + i)
    print(f"track {a.seed + i}: score {r['score']:.1f}, {'lap finished' if r['lap'] else 'time up'}, {r['tiles']:.0%} of the track, in {r['steps']} frames")
