from pathlib import Path

from a2c_agent import A2CConfig, A2CTrainer
from env.vector_env import SimpleVectorEnv
from log_utils import TrainingLogger
from tests.dummy_envs import DummyCarRacing


def test_best_model_is_saved(tmp_path: Path):
    envs = SimpleVectorEnv([lambda: DummyCarRacing(max_steps=2) for _ in range(2)])
    cfg = A2CConfig(total_steps=40, rollout_length=4, log_interval=1)
    logger = TrainingLogger(tmp_path, config={"test": "best_model"})
    trainer = A2CTrainer(envs, cfg, logger=logger)
    try:
        trainer.train()
    finally:
        envs.close()
        logger.close()

    assert (tmp_path / "best_model.pt").exists()
