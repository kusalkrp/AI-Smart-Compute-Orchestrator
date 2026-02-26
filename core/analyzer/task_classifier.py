from __future__ import annotations

from models.enums import Priority, TaskType


class TaskClassifier:
    """Map task type and priority to latency and cost sensitivity defaults."""

    # (latency_sensitivity, cost_sensitivity) per task type
    _TYPE_DEFAULTS: dict[TaskType, tuple[float, float]] = {
        TaskType.CHAT: (0.8, 0.4),
        TaskType.SUMMARIZATION: (0.5, 0.5),
        TaskType.CLASSIFICATION: (0.6, 0.6),
        TaskType.EMBEDDING: (0.4, 0.8),
        TaskType.REASONING: (0.6, 0.3),
        TaskType.BATCH_SUMMARIZATION: (0.2, 0.9),
    }

    _PRIORITY_LATENCY_BOOST: dict[Priority, float] = {
        Priority.LOW: -0.2,
        Priority.NORMAL: 0.0,
        Priority.HIGH: 0.2,
        Priority.URGENT: 0.4,
    }

    _PRIORITY_COST_REDUCTION: dict[Priority, float] = {
        Priority.LOW: 0.2,
        Priority.NORMAL: 0.0,
        Priority.HIGH: -0.2,
        Priority.URGENT: -0.4,
    }

    def classify(
        self,
        task_type: TaskType,
        priority: Priority,
        is_batch: bool = False,
    ) -> tuple[float, float]:
        """Return (latency_sensitivity, cost_sensitivity) clamped to [0, 1]."""
        base_latency, base_cost = self._TYPE_DEFAULTS.get(task_type, (0.5, 0.5))

        latency = base_latency + self._PRIORITY_LATENCY_BOOST[priority]
        cost = base_cost + self._PRIORITY_COST_REDUCTION[priority]

        if is_batch:
            latency = max(0.0, latency - 0.3)
            cost = min(1.0, cost + 0.2)

        return (
            max(0.0, min(1.0, latency)),
            max(0.0, min(1.0, cost)),
        )

    def requires_reasoning(self, task_type: TaskType, metadata: dict) -> bool:
        if task_type == TaskType.REASONING:
            return True
        if metadata.get("chain_of_thought") or metadata.get("requires_reasoning"):
            return True
        return False
