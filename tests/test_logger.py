import json

from log_utils import TrainingLogger


def test_training_logger_writes_config_and_metrics(tmp_path):
    logger = TrainingLogger(tmp_path, config={"foo": "bar"})
    logger.log_metrics({"step": 1, "loss": 0.5})
    logger.close()

    config_path = tmp_path / "config.json"
    metrics_path = tmp_path / "metrics.csv"
    assert config_path.exists()
    assert metrics_path.exists()

    payload = json.loads(config_path.read_text())
    assert payload["foo"] == "bar"
    assert "created_at" in payload
