from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config.settings import settings
from models.enums import ExecutionTarget, Priority
from models.schemas import ExecutionProfile, ResourceSnapshot


class RuleEngine:
    """
    Stage 1 routing: evaluate YAML-defined rules against ExecutionProfile + ResourceSnapshot.
    First matching rule wins.
    """

    _PRIORITY_ORDER = {
        Priority.LOW: 0,
        Priority.NORMAL: 1,
        Priority.HIGH: 2,
        Priority.URGENT: 3,
    }

    def __init__(self) -> None:
        self._rules: list[dict] = []
        self._fallback_chain: dict[str, str] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        try:
            path = Path(settings.ROUTING_POLICY_PATH)
            with path.open() as f:
                data = yaml.safe_load(f)
            self._rules = sorted(
                data.get("rules", []),
                key=lambda r: r.get("priority_order", 999),
            )
            self._fallback_chain = data.get("fallback_chain", {})
        except Exception:
            # Default minimal rule set
            self._rules = [
                {"name": "default_cpu", "conditions": {}, "target": "CPU", "model": "phi3:mini"}
            ]
            self._fallback_chain = {"GPU": "CPU", "CPU": "QUANTIZED", "QUANTIZED": "CLOUD", "CLOUD": "CPU"}

    def evaluate(
        self,
        profile: ExecutionProfile,
        resource: ResourceSnapshot,
        priority: Priority,
    ) -> tuple[ExecutionTarget, str, str]:
        """
        Returns (target, model_name, rule_name) for first matching rule.
        """
        context = self._build_context(profile, resource, priority)

        for rule in self._rules:
            conditions = rule.get("conditions", {})
            if self._matches(conditions, context):
                target_str = rule.get("target", "CPU")
                model = rule.get("model", settings.CPU_MODEL)
                return ExecutionTarget(target_str), model, rule.get("name", "unknown")

        # Should never reach here due to default_cpu catch-all, but safety fallback
        return ExecutionTarget.CPU, settings.CPU_MODEL, "hardcoded_default"

    def get_fallback(self, target: ExecutionTarget) -> ExecutionTarget:
        fallback_str = self._fallback_chain.get(target.value, "CLOUD")
        return ExecutionTarget(fallback_str)

    def reload(self) -> None:
        self._load_rules()

    def _build_context(
        self,
        profile: ExecutionProfile,
        resource: ResourceSnapshot,
        priority: Priority,
    ) -> dict[str, Any]:
        return {
            "priority": priority.value,
            "priority_level": self._PRIORITY_ORDER[priority],
            "complexity_score": profile.complexity_score,
            "latency_sensitivity": profile.latency_sensitivity,
            "cost_sensitivity": profile.cost_sensitivity,
            "estimated_tokens": profile.estimated_tokens,
            "is_batch": profile.is_batch,
            "requires_reasoning": profile.requires_reasoning,
            "gpu_available": resource.gpu_available,
            "gpu_utilization": resource.gpu_percent,
            "cpu_utilization": resource.cpu_percent,
            "ram_percent": resource.ram_percent,
            "gpu_vram_used_mb": resource.gpu_vram_used_mb,
        }

    def _matches(self, conditions: dict, context: dict) -> bool:
        for key, value in conditions.items():
            # Boolean / exact match conditions
            if key == "priority":
                if context["priority"] != value:
                    return False
            elif key == "is_batch":
                if context["is_batch"] != value:
                    return False
            elif key == "requires_reasoning":
                if context["requires_reasoning"] != value:
                    return False
            elif key == "gpu_available":
                if context["gpu_available"] != value:
                    return False
            # Priority >= condition
            elif key == "priority_gte":
                levels = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "URGENT": 3}
                if context["priority_level"] < levels.get(value, 0):
                    return False
            # Numeric greater-than conditions
            elif key.endswith("_gt"):
                field = key[:-3]
                if field not in context or context[field] <= value:
                    return False
            # Numeric greater-than-or-equal conditions
            elif key.endswith("_gte"):
                field = key[:-4]
                if field not in context or context[field] < value:
                    return False
            # Numeric less-than-or-equal conditions
            elif key.endswith("_lte"):
                field = key[:-4]
                if field not in context or context[field] > value:
                    return False
            # Numeric less-than conditions
            elif key.endswith("_lt"):
                field = key[:-3]
                if field not in context or context[field] >= value:
                    return False

        return True
