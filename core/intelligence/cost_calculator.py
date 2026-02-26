from __future__ import annotations

import yaml
from pathlib import Path

from config.settings import settings
from models.enums import ExecutionTarget


class CostCalculator:
    """Calculate estimated cost per request given token count and execution target."""

    def __init__(self) -> None:
        self._config: dict = {}
        self._load_config()

    def _load_config(self) -> None:
        try:
            path = Path(settings.COST_CONFIG_PATH)
            with path.open() as f:
                self._config = yaml.safe_load(f)
        except Exception:
            self._config = {
                "cost_per_1k_tokens": {
                    "CLOUD": {"gemini-2.0-flash": 0.00025},
                    "GPU": {},
                    "CPU": {},
                    "QUANTIZED": {},
                },
                "overhead_per_request": {
                    "CLOUD": 0.00001,
                    "GPU": 0.000001,
                    "CPU": 0.000001,
                    "QUANTIZED": 0.000001,
                },
                "default_latency_ms": {
                    "CLOUD": {"gemini-2.0-flash": 1800},
                    "GPU": {"mistral:7b-instruct-q4_0": 1500},
                    "CPU": {"phi3:mini": 8000},
                    "QUANTIZED": {"phi3:mini": 6000},
                },
            }

    def estimate_cost(
        self,
        target: ExecutionTarget,
        model_name: str,
        estimated_tokens: int,
    ) -> float:
        target_str = target.value
        cost_table = self._config.get("cost_per_1k_tokens", {}).get(target_str, {})
        overhead = self._config.get("overhead_per_request", {}).get(target_str, 0.0)

        # Find rate for this model or fallback
        rate = cost_table.get(model_name, 0.0)

        token_cost = (estimated_tokens / 1000.0) * rate
        return round(token_cost + overhead, 8)

    def estimate_latency(
        self,
        target: ExecutionTarget,
        model_name: str,
        estimated_tokens: int,
    ) -> int:
        target_str = target.value
        latency_table = self._config.get("default_latency_ms", {}).get(target_str, {})
        base_latency = latency_table.get(model_name, 5000)

        # Scale latency with token count (rough estimate: +1ms per 4 tokens over 512)
        extra_tokens = max(0, estimated_tokens - 512)
        extra_latency = extra_tokens // 4  # 1ms per 4 tokens

        return base_latency + extra_latency

    def reload(self) -> None:
        self._load_config()
