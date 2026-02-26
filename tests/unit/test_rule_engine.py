"""Unit tests for RuleEngine."""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from core.intelligence.rule_engine import RuleEngine
from models.enums import ExecutionTarget, Priority, TaskType
from models.schemas import ExecutionProfile, ResourceSnapshot


@pytest.fixture
def engine() -> RuleEngine:
    return RuleEngine()


def make_profile(
    complexity: float = 0.5,
    latency_sens: float = 0.5,
    cost_sens: float = 0.5,
    tokens: int = 512,
    is_batch: bool = False,
    requires_reasoning: bool = False,
) -> ExecutionProfile:
    return ExecutionProfile(
        task_id=uuid.uuid4(),
        task_type=TaskType.CHAT,
        complexity_score=complexity,
        latency_sensitivity=latency_sens,
        cost_sensitivity=cost_sens,
        estimated_tokens=tokens,
        is_batch=is_batch,
        requires_reasoning=requires_reasoning,
    )


def make_resource(
    gpu_available: bool = True,
    gpu_percent: float = 50.0,
    cpu_percent: float = 30.0,
    gpu_vram_used: float = 2000.0,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        cpu_percent=cpu_percent,
        ram_percent=40.0,
        gpu_percent=gpu_percent,
        gpu_vram_used_mb=gpu_vram_used,
        gpu_vram_total_mb=6144.0,
        gpu_available=gpu_available,
    )


class TestRuleEngine:
    def test_urgent_complex_goes_to_cloud(self, engine: RuleEngine) -> None:
        profile = make_profile(complexity=0.8)
        resource = make_resource(gpu_available=True, gpu_percent=50.0)
        target, model, rule = engine.evaluate(profile, resource, Priority.URGENT)
        assert target == ExecutionTarget.CLOUD

    def test_batch_low_latency_goes_to_quantized(self, engine: RuleEngine) -> None:
        profile = make_profile(is_batch=True, latency_sens=0.2)
        resource = make_resource()
        target, model, rule = engine.evaluate(profile, resource, Priority.LOW)
        assert target == ExecutionTarget.QUANTIZED

    def test_gpu_available_medium_task(self, engine: RuleEngine) -> None:
        profile = make_profile(complexity=0.5, tokens=1024)
        resource = make_resource(gpu_available=True, gpu_percent=40.0)
        target, model, rule = engine.evaluate(profile, resource, Priority.NORMAL)
        assert target == ExecutionTarget.GPU

    def test_no_gpu_falls_back(self, engine: RuleEngine) -> None:
        profile = make_profile(complexity=0.4)
        resource = make_resource(gpu_available=False)
        target, model, rule = engine.evaluate(profile, resource, Priority.NORMAL)
        # Without GPU, should not route to GPU
        assert target != ExecutionTarget.GPU

    def test_fallback_chain(self, engine: RuleEngine) -> None:
        assert engine.get_fallback(ExecutionTarget.GPU) == ExecutionTarget.CPU
        assert engine.get_fallback(ExecutionTarget.CPU) == ExecutionTarget.QUANTIZED
        assert engine.get_fallback(ExecutionTarget.QUANTIZED) == ExecutionTarget.CLOUD

    def test_returns_valid_model_name(self, engine: RuleEngine) -> None:
        profile = make_profile()
        resource = make_resource()
        target, model, rule = engine.evaluate(profile, resource, Priority.NORMAL)
        assert isinstance(model, str)
        assert len(model) > 0

    def test_rule_name_returned(self, engine: RuleEngine) -> None:
        profile = make_profile()
        resource = make_resource()
        target, model, rule = engine.evaluate(profile, resource, Priority.NORMAL)
        assert isinstance(rule, str)
