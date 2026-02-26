from __future__ import annotations

from models.enums import TaskType


class ComplexityScorer:
    """Score task complexity from 0.0 (trivial) to 1.0 (maximal complexity)."""

    # Base complexity per task type
    _TYPE_BASE: dict[TaskType, float] = {
        TaskType.CHAT: 0.3,
        TaskType.SUMMARIZATION: 0.4,
        TaskType.CLASSIFICATION: 0.25,
        TaskType.EMBEDDING: 0.1,
        TaskType.REASONING: 0.7,
        TaskType.BATCH_SUMMARIZATION: 0.5,
    }

    # Token count thresholds for scaling
    _TOKEN_TIERS = [
        (256, 0.0),
        (512, 0.1),
        (1024, 0.2),
        (2048, 0.3),
        (4096, 0.4),
        (8192, 0.6),
        (16384, 0.8),
        (float("inf"), 1.0),
    ]

    def score(
        self,
        task_type: TaskType,
        estimated_tokens: int,
        requires_reasoning: bool = False,
        is_batch: bool = False,
        metadata: dict | None = None,
    ) -> float:
        base = self._TYPE_BASE.get(task_type, 0.4)

        # Token complexity bonus
        token_bonus = 0.0
        for threshold, bonus in self._TOKEN_TIERS:
            if estimated_tokens <= threshold:
                token_bonus = bonus
                break

        # Reasoning bonus
        reasoning_bonus = 0.15 if requires_reasoning else 0.0

        # Batch slight increase (more tokens overall)
        batch_bonus = 0.1 if is_batch else 0.0

        # Metadata hints
        meta_bonus = 0.0
        if metadata:
            if metadata.get("multi_hop"):
                meta_bonus += 0.1
            if metadata.get("code_generation"):
                meta_bonus += 0.1
            if metadata.get("long_form"):
                meta_bonus += 0.05

        score = base + token_bonus * 0.5 + reasoning_bonus + batch_bonus + meta_bonus
        return max(0.0, min(1.0, score))
