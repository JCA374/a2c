from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _to_serializable(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


class TrainingLogger:
    """CSV-based logger that stores configs and metrics."""

    def __init__(self, log_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.log_dir / "config.json"
        self.metrics_path = self.log_dir / "metrics.csv"
        self._metrics_file = self.metrics_path.open("w", newline="")
        self._writer: Optional[csv.DictWriter] = None

        if config:
            self.log_config(config)

    def log_config(self, config: Dict[str, Any]):
        payload = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            **_to_serializable(config),
        }
        self.config_path.write_text(json.dumps(payload, indent=2))

    def log_metrics(self, metrics: Dict[str, Any]):
        if self._writer is None:
            fieldnames = list(metrics.keys())
            self._writer = csv.DictWriter(self._metrics_file, fieldnames=fieldnames)
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._metrics_file.flush()

    def close(self):
        if self._metrics_file and not self._metrics_file.closed:
            self._metrics_file.close()

