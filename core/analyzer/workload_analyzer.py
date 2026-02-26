from __future__ import annotations

import uuid

from core.analyzer.complexity_scorer import ComplexityScorer
from core.analyzer.task_classifier import TaskClassifier
from core.analyzer.token_estimator import TokenEstimator
from models.schemas import ExecutionProfile, TaskRequest


class WorkloadAnalyzer:
    """
    Pipeline: TaskRequest → ExecutionProfile

    Steps:
    1. TaskClassifier → latency_sensitivity, cost_sensitivity
    2. TokenEstimator → estimated_tokens
    3. ComplexityScorer → complexity_score
    """

    def __init__(self) -> None:
        self._classifier = TaskClassifier()
        self._estimator = TokenEstimator()
        self._scorer = ComplexityScorer()

    def analyze(self, task_id: uuid.UUID, request: TaskRequest) -> ExecutionProfile:
        latency_sensitivity, cost_sensitivity = self._classifier.classify(
            task_type=request.task_type,
            priority=request.priority,
            is_batch=request.is_batch,
        )

        requires_reasoning = self._classifier.requires_reasoning(
            task_type=request.task_type,
            metadata=request.metadata,
        )

        estimated_tokens = self._estimator.estimate(request.input_text)

        complexity_score = self._scorer.score(
            task_type=request.task_type,
            estimated_tokens=estimated_tokens,
            requires_reasoning=requires_reasoning,
            is_batch=request.is_batch,
            metadata=request.metadata,
        )

        # Override sensitivities with user-provided constraints
        if request.max_latency_ms is not None:
            # Very tight latency constraint → boost latency sensitivity
            if request.max_latency_ms < 5000:
                latency_sensitivity = min(1.0, latency_sensitivity + 0.3)

        if request.max_cost_usd is not None:
            # Tight cost constraint → boost cost sensitivity
            if request.max_cost_usd < 0.001:
                cost_sensitivity = min(1.0, cost_sensitivity + 0.3)

        return ExecutionProfile(
            task_id=task_id,
            task_type=request.task_type,
            complexity_score=complexity_score,
            latency_sensitivity=latency_sensitivity,
            cost_sensitivity=cost_sensitivity,
            estimated_tokens=estimated_tokens,
            is_batch=request.is_batch,
            requires_reasoning=requires_reasoning,
        )
