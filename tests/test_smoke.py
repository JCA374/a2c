import sys
from pathlib import Path

from env.vector_env import SimpleVectorEnv
from tests.dummy_envs import DummyCarRacing


def test_train_script_smoke(monkeypatch, tmp_path: Path):
    import train_a2c

    def fake_build_envs(num_envs, env_config, base_seed):
        return SimpleVectorEnv([lambda: DummyCarRacing(max_steps=1) for _ in range(num_envs)])

    log_dir = tmp_path / "logs"
    argv = [
        "train_a2c.py",
        "--num-envs",
        "2",
        "--total-steps",
        "20",
        "--rollout-length",
        "2",
        "--log-interval",
        "1",
        "--log-dir",
        str(log_dir),
    ]

    monkeypatch.setattr(train_a2c, "build_envs", fake_build_envs)
    monkeypatch.setattr(sys, "argv", argv)

    train_a2c.main()

    metrics_file = log_dir / "metrics.csv"
    assert metrics_file.exists()
