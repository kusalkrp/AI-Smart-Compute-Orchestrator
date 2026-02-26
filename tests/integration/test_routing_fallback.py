"""Integration test: routing fallback when target unavailable."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from core.intelligence.decision_engine import DecisionEngine
from core.router.fallback_handler import FallbackHandler
from models.enums import ExecutionTarget, Priority, TaskType
from models.schemas import ResourceSnapshot, TaskRequest


def gpu_overloaded_resource() -> ResourceSnapshot:
    return ResourceSnapshot(
        cpu_percent=30.0,
        ram_percent=40.0,
        gpu_percent=95.0,  # GPU overloaded
        gpu_vram_used_mb=5800.0,
        gpu_vram_total_mb=6144.0,
        gpu_available=True,
    )


def no_resources() -> ResourceSnapshot:
    return ResourceSnapshot(
        cpu_percent=92.0,  # CPU overloaded too
        ram_percent=80.0,
        gpu_percent=0.0,
        gpu_vram_used_mb=0.0,
        gpu_vram_total_mb=0.0,
        gpu_available=False,
    )


class TestFallbackHandler:
    def test_gpu_falls_back_to_cpu(self) -> None:
        handler = FallbackHandler()
        assert handler.get_fallback(ExecutionTarget.GPU) == ExecutionTarget.CPU

    def test_cpu_falls_back_to_quantized(self) -> None:
        handler = FallbackHandler()
        assert handler.get_fallback(ExecutionTarget.CPU) == ExecutionTarget.QUANTIZED

    def test_quantized_falls_back_to_cloud(self) -> None:
        handler = FallbackHandler()
        assert handler.get_fallback(ExecutionTarget.QUANTIZED) == ExecutionTarget.CLOUD

    def test_chain_from_gpu(self) -> None:
        handler = FallbackHandler()
        chain = handler.get_chain(ExecutionTarget.GPU, max_depth=3)
        assert ExecutionTarget.GPU in chain
        assert len(chain) > 1

    @pytest.mark.asyncio
    async def test_overloaded_gpu_not_selected(self) -> None:
        resource = gpu_overloaded_resource()
        with patch.object(DecisionEngine, "_load_performance_cache", new_callable=AsyncMock, return_value={}):
            engine = DecisionEngine()
            request = TaskRequest(
                task_type=TaskType.CHAT,
                input_text="Hello",
                priority=Priority.NORMAL,
            )
            decision = await engine.decide(uuid.uuid4(), request, resource)
            # GPU is 95% utilized (> 85% threshold) — should not select GPU
            assert decision.target != ExecutionTarget.GPU

    @pytest.mark.asyncio
    async def test_no_gpu_routes_to_local_or_cloud(self) -> None:
        resource = ResourceSnapshot(
            cpu_percent=30.0,
            ram_percent=40.0,
            gpu_available=False,
            gpu_percent=0.0,
            gpu_vram_used_mb=0.0,
            gpu_vram_total_mb=0.0,
        )
        with patch.object(DecisionEngine, "_load_performance_cache", new_callable=AsyncMock, return_value={}):
            engine = DecisionEngine()
            request = TaskRequest(
                task_type=TaskType.CHAT,
                input_text="Hello",
                priority=Priority.NORMAL,
            )
            decision = await engine.decide(uuid.uuid4(), request, resource)
            assert decision.target in (ExecutionTarget.CPU, ExecutionTarget.QUANTIZED, ExecutionTarget.CLOUD)
