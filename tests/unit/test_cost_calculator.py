"""Unit tests for CostCalculator."""
from __future__ import annotations

import pytest

from core.intelligence.cost_calculator import CostCalculator
from models.enums import ExecutionTarget


@pytest.fixture
def calc() -> CostCalculator:
    return CostCalculator()


class TestCostCalculator:
    def test_local_targets_zero_cost(self, calc: CostCalculator) -> None:
        for target in (ExecutionTarget.GPU, ExecutionTarget.CPU, ExecutionTarget.QUANTIZED):
            model = "phi3:mini" if target != ExecutionTarget.GPU else "mistral:7b-instruct-q4_0"
            cost = calc.estimate_cost(target, model, 1000)
            assert cost >= 0.0
            # Local targets should be near-zero (just overhead)
            assert cost < 0.001

    def test_cloud_cost_scales_with_tokens(self, calc: CostCalculator) -> None:
        cost_500 = calc.estimate_cost(ExecutionTarget.CLOUD, "gemini-2.0-flash", 500)
        cost_1000 = calc.estimate_cost(ExecutionTarget.CLOUD, "gemini-2.0-flash", 1000)
        cost_2000 = calc.estimate_cost(ExecutionTarget.CLOUD, "gemini-2.0-flash", 2000)

        assert cost_500 <= cost_1000 <= cost_2000

    def test_cost_formula_accuracy(self, calc: CostCalculator) -> None:
        # gemini-2.0-flash: $0.00025 per 1k tokens
        tokens = 1000
        expected_min = 0.00025  # approximately
        cost = calc.estimate_cost(ExecutionTarget.CLOUD, "gemini-2.0-flash", tokens)
        assert cost >= expected_min * 0.9  # allow 10% tolerance

    def test_latency_scales_with_tokens(self, calc: CostCalculator) -> None:
        lat_short = calc.estimate_latency(ExecutionTarget.GPU, "mistral:7b-instruct-q4_0", 256)
        lat_long = calc.estimate_latency(ExecutionTarget.GPU, "mistral:7b-instruct-q4_0", 4096)
        assert lat_long >= lat_short

    def test_cloud_faster_than_cpu_by_default(self, calc: CostCalculator) -> None:
        cloud_lat = calc.estimate_latency(ExecutionTarget.CLOUD, "gemini-2.0-flash", 512)
        cpu_lat = calc.estimate_latency(ExecutionTarget.CPU, "phi3:mini", 512)
        # Cloud should be faster than CPU for typical prompts
        assert cloud_lat < cpu_lat

    def test_all_targets_return_positive_latency(self, calc: CostCalculator) -> None:
        models = {
            ExecutionTarget.GPU: "mistral:7b-instruct-q4_0",
            ExecutionTarget.CPU: "phi3:mini",
            ExecutionTarget.QUANTIZED: "phi3:mini",
            ExecutionTarget.CLOUD: "gemini-2.0-flash",
        }
        for target, model in models.items():
            latency = calc.estimate_latency(target, model, 512)
            assert latency > 0, f"Expected positive latency for {target}"
