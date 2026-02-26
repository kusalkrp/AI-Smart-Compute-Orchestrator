"""Unit tests for WorkloadAnalyzer pipeline."""
from __future__ import annotations

import uuid

import pytest

from core.analyzer.workload_analyzer import WorkloadAnalyzer
from models.enums import Priority, TaskType
from models.schemas import TaskRequest


@pytest.fixture
def analyzer() -> WorkloadAnalyzer:
    return WorkloadAnalyzer()


def make_request(
    task_type: TaskType = TaskType.CHAT,
    priority: Priority = Priority.NORMAL,
    text: str = "Hello world",
    is_batch: bool = False,
    metadata: dict | None = None,
) -> TaskRequest:
    return TaskRequest(
        task_type=task_type,
        input_text=text,
        priority=priority,
        is_batch=is_batch,
        metadata=metadata or {},
    )


class TestWorkloadAnalyzer:
    def test_basic_chat_profile(self, analyzer: WorkloadAnalyzer) -> None:
        task_id = uuid.uuid4()
        request = make_request(TaskType.CHAT, Priority.NORMAL, "Hi")
        profile = analyzer.analyze(task_id, request)

        assert profile.task_id == task_id
        assert profile.task_type == TaskType.CHAT
        assert 0.0 <= profile.complexity_score <= 1.0
        assert 0.0 <= profile.latency_sensitivity <= 1.0
        assert 0.0 <= profile.cost_sensitivity <= 1.0
        assert profile.estimated_tokens >= 1
        assert not profile.is_batch
        assert not profile.requires_reasoning

    def test_urgent_high_latency_sensitivity(self, analyzer: WorkloadAnalyzer) -> None:
        request = make_request(TaskType.CHAT, Priority.URGENT)
        profile = analyzer.analyze(uuid.uuid4(), request)
        # URGENT should push latency_sensitivity up
        assert profile.latency_sensitivity > 0.5

    def test_low_priority_cost_sensitive(self, analyzer: WorkloadAnalyzer) -> None:
        request = make_request(TaskType.CHAT, Priority.LOW)
        profile = analyzer.analyze(uuid.uuid4(), request)
        # LOW priority should have higher cost sensitivity
        assert profile.cost_sensitivity > 0.3

    def test_batch_task(self, analyzer: WorkloadAnalyzer) -> None:
        request = make_request(TaskType.BATCH_SUMMARIZATION, Priority.LOW, is_batch=True)
        profile = analyzer.analyze(uuid.uuid4(), request)
        assert profile.is_batch is True
        # Batch tasks should be more cost-sensitive
        assert profile.cost_sensitivity > 0.5

    def test_reasoning_task(self, analyzer: WorkloadAnalyzer) -> None:
        request = make_request(
            TaskType.REASONING,
            Priority.NORMAL,
            metadata={"chain_of_thought": True},
        )
        profile = analyzer.analyze(uuid.uuid4(), request)
        assert profile.requires_reasoning is True

    def test_large_input_higher_complexity(self, analyzer: WorkloadAnalyzer) -> None:
        short_request = make_request(text="Hi")
        long_request = make_request(text="word " * 2000)

        short_profile = analyzer.analyze(uuid.uuid4(), short_request)
        long_profile = analyzer.analyze(uuid.uuid4(), long_request)

        assert long_profile.complexity_score >= short_profile.complexity_score
        assert long_profile.estimated_tokens > short_profile.estimated_tokens

    def test_max_latency_constraint_boosts_sensitivity(self, analyzer: WorkloadAnalyzer) -> None:
        base = TaskRequest(task_type=TaskType.CHAT, input_text="test", priority=Priority.NORMAL)
        tight = TaskRequest(task_type=TaskType.CHAT, input_text="test", priority=Priority.NORMAL, max_latency_ms=2000)

        base_profile = analyzer.analyze(uuid.uuid4(), base)
        tight_profile = analyzer.analyze(uuid.uuid4(), tight)

        assert tight_profile.latency_sensitivity >= base_profile.latency_sensitivity

    def test_complexity_scores_in_range(self, analyzer: WorkloadAnalyzer) -> None:
        for task_type in TaskType:
            request = make_request(task_type=task_type)
            profile = analyzer.analyze(uuid.uuid4(), request)
            assert 0.0 <= profile.complexity_score <= 1.0, f"Out of range for {task_type}"
