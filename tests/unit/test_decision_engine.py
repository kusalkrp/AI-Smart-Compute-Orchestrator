"""Unit tests for DecisionEngine."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.intelligence.decision_engine import DecisionEngine
from models.enums import ExecutionTarget, Priority, TaskType
from models.schemas import ResourceSnapshot, TaskRequest


@pytest.fixture
def gpu_resource() -> ResourceSnapshot:
    return ResourceSnapshot(
        cpu_percent=30.0,
        ram_percent=40.0,
        gpu_percent=50.0,
        gpu_vram_used_mb=2000.0,
        gpu_vram_total_mb=6144.0,
        gpu_available=True,
    )


@pytest.fixture
def no_gpu_resource() -> ResourceSnapshot:
    return ResourceSnapshot(
        cpu_percent=30.0,
        ram_percent=40.0,
        gpu_percent=0.0,
        gpu_vram_used_mb=0.0,
        gpu_vram_total_mb=0.0,
        gpu_available=False,
    )


def make_request(
    task_type: TaskType = TaskType.CHAT,
    priority: Priority = Priority.NORMAL,
    text: str = "Hello",
) -> TaskRequest:
    return TaskRequest(task_type=task_type, input_text=text, priority=priority)


class TestDecisionEngine:
    @pytest.mark.asyncio
    async def test_returns_routing_decision(self, gpu_resource: ResourceSnapshot) -> None:
        with patch.object(DecisionEngine, "_load_performance_cache", new_callable=AsyncMock, return_value={}):
            engine = DecisionEngine()
            task_id = uuid.uuid4()
            request = make_request()
            decision = await engine.decide(task_id, request, gpu_resource)

            assert decision.task_id == task_id
            assert decision.target in list(ExecutionTarget)
            assert decision.model_name
            assert 0.0 <= decision.confidence <= 1.0
            assert decision.estimated_cost_usd >= 0.0
            assert decision.estimated_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_urgent_complex_routes_to_cloud_or_gpu(self, gpu_resource: ResourceSnapshot) -> None:
        with patch.object(DecisionEngine, "_load_performance_cache", new_callable=AsyncMock, return_value={}):
            engine = DecisionEngine()
            request = make_request(
                task_type=TaskType.REASONING,
                priority=Priority.URGENT,
                text="Complex reasoning: " + "x" * 500,
            )
            decision = await engine.decide(uuid.uuid4(), request, gpu_resource)

            # URGENT + complex should prefer cloud or fast GPU
            assert decision.target in (ExecutionTarget.CLOUD, ExecutionTarget.GPU)

    @pytest.mark.asyncio
    async def test_no_gpu_does_not_route_to_gpu(self, no_gpu_resource: ResourceSnapshot) -> None:
        with patch.object(DecisionEngine, "_load_performance_cache", new_callable=AsyncMock, return_value={}):
            engine = DecisionEngine()
            request = make_request()
            decision = await engine.decide(uuid.uuid4(), request, no_gpu_resource)

            assert decision.target != ExecutionTarget.GPU

    @pytest.mark.asyncio
    async def test_decision_has_fallback(self, gpu_resource: ResourceSnapshot) -> None:
        with patch.object(DecisionEngine, "_load_performance_cache", new_callable=AsyncMock, return_value={}):
            engine = DecisionEngine()
            request = make_request()
            decision = await engine.decide(uuid.uuid4(), request, gpu_resource)

            assert decision.fallback_target in list(ExecutionTarget)
            assert decision.fallback_target != decision.target or decision.target == ExecutionTarget.CPU

    @pytest.mark.asyncio
    async def test_reasoning_field_non_empty(self, gpu_resource: ResourceSnapshot) -> None:
        with patch.object(DecisionEngine, "_load_performance_cache", new_callable=AsyncMock, return_value={}):
            engine = DecisionEngine()
            request = make_request()
            decision = await engine.decide(uuid.uuid4(), request, gpu_resource)

            assert len(decision.reasoning) > 0

    @pytest.mark.asyncio
    async def test_batch_low_cost_route(self, gpu_resource: ResourceSnapshot) -> None:
        with patch.object(DecisionEngine, "_load_performance_cache", new_callable=AsyncMock, return_value={}):
            engine = DecisionEngine()
            request = TaskRequest(
                task_type=TaskType.BATCH_SUMMARIZATION,
                input_text="process batch " + "x" * 200,
                priority=Priority.LOW,
                is_batch=True,
            )
            decision = await engine.decide(uuid.uuid4(), request, gpu_resource)
            # Batch + LOW should prefer cost-efficient targets
            assert decision.target in (ExecutionTarget.QUANTIZED, ExecutionTarget.CPU, ExecutionTarget.GPU)
