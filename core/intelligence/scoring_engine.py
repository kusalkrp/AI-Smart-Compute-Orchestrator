from __future__ import annotations

from pathlib import Path

import yaml

from config.settings import settings
from core.intelligence.cost_calculator import CostCalculator
from models.enums import ExecutionTarget
from models.schemas import ExecutionProfile, ResourceSnapshot


class ScoringEngine:
    """
    Stage 2: Weighted multi-factor scoring to select best execution target.

    Score formula (lower is better):
        score = w1*cost_norm + w2*latency_norm + w3*(1-availability) + w4*(1-success_rate)
    """

    _TARGETS = [ExecutionTarget.GPU, ExecutionTarget.CPU, ExecutionTarget.QUANTIZED, ExecutionTarget.CLOUD]

    _TARGET_MODELS = {
        ExecutionTarget.GPU: settings.GPU_MODEL,
        ExecutionTarget.CPU: settings.CPU_MODEL,
        ExecutionTarget.QUANTIZED: settings.CPU_MODEL,
        ExecutionTarget.CLOUD: settings.CLOUD_MODEL,
    }

    def __init__(self) -> None:
        self._cost_calc = CostCalculator()
        self._weights = self._load_weights()

    def _load_weights(self) -> dict[str, float]:
        try:
            path = Path(settings.ROUTING_POLICY_PATH)
            with path.open() as f:
                data = yaml.safe_load(f)
            w = data.get("scoring_weights", {})
            return {
                "cost": w.get("cost_weight", 0.3),
                "availability": w.get("availability_weight", 0.3),
                "success_rate": w.get("success_rate_weight", 0.2),
                "latency": w.get("latency_weight", 0.2),
            }
        except Exception:
            return {"cost": 0.3, "availability": 0.3, "success_rate": 0.2, "latency": 0.2}

    def score(
        self,
        profile: ExecutionProfile,
        resource: ResourceSnapshot,
        performance_cache: dict[str, dict] | None = None,
    ) -> tuple[ExecutionTarget, str, float]:
        """
        Returns (best_target, model_name, confidence) where confidence ∈ [0, 1].
        """
        scores: dict[ExecutionTarget, float] = {}

        for target in self._targets_available(resource):
            model = self._TARGET_MODELS.get(target, settings.CPU_MODEL)
            score = self._compute_score(target, model, profile, resource, performance_cache)
            scores[target] = score

        if not scores:
            return ExecutionTarget.CPU, settings.CPU_MODEL, 0.5

        best = min(scores, key=lambda t: scores[t])
        model = self._TARGET_MODELS.get(best, settings.CPU_MODEL)

        # Confidence = how much better best is over second best
        sorted_scores = sorted(scores.values())
        if len(sorted_scores) >= 2:
            gap = sorted_scores[1] - sorted_scores[0]
            confidence = min(1.0, 0.5 + gap * 2.0)
        else:
            confidence = 0.7

        return best, model, confidence

    def _targets_available(self, resource: ResourceSnapshot) -> list[ExecutionTarget]:
        available = []
        if resource.gpu_available and resource.gpu_percent < settings.GPU_OVERLOAD_PERCENT:
            available.append(ExecutionTarget.GPU)
        if resource.cpu_percent < settings.CPU_OVERLOAD_PERCENT:
            available.append(ExecutionTarget.CPU)
        available.append(ExecutionTarget.QUANTIZED)
        available.append(ExecutionTarget.CLOUD)
        return available or [ExecutionTarget.CLOUD]

    def _compute_score(
        self,
        target: ExecutionTarget,
        model: str,
        profile: ExecutionProfile,
        resource: ResourceSnapshot,
        performance_cache: dict | None,
    ) -> float:
        w = self._weights

        # Normalize cost (cloud is most expensive, local is 0)
        raw_cost = self._cost_calc.estimate_cost(target, model, profile.estimated_tokens)
        cost_norm = min(1.0, raw_cost / 0.001) if raw_cost > 0 else 0.0

        # Availability score (0 = fully available, 1 = overloaded)
        availability_penalty = self._availability_penalty(target, resource)

        # Success rate (from cache or default)
        success_rate = 1.0
        if performance_cache:
            key = f"{target.value}:{model}"
            perf = performance_cache.get(key, {})
            success_rate = perf.get("success_rate", 1.0)
        success_penalty = 1.0 - success_rate

        # Latency score: blend of raw latency and latency_sensitivity
        raw_latency = self._cost_calc.estimate_latency(target, model, profile.estimated_tokens)
        latency_norm = min(1.0, raw_latency / 20000.0)
        # If high latency sensitivity, penalize slow targets more
        latency_penalty = latency_norm * (0.5 + 0.5 * profile.latency_sensitivity)

        total = (
            w["cost"] * cost_norm
            + w["availability"] * availability_penalty
            + w["success_rate"] * success_penalty
            + w["latency"] * latency_penalty
        )

        # Bonus for cost-sensitive profiles preferring local
        if profile.cost_sensitivity > 0.7 and target != ExecutionTarget.CLOUD:
            total -= 0.1

        return max(0.0, total)

    def _availability_penalty(self, target: ExecutionTarget, resource: ResourceSnapshot) -> float:
        if target == ExecutionTarget.GPU:
            return resource.gpu_percent / 100.0
        elif target == ExecutionTarget.CPU:
            queue_pressure = min(1.0, resource.cpu_queue_depth / 10.0)
            return (resource.cpu_percent / 100.0) * 0.5 + queue_pressure * 0.5
        elif target == ExecutionTarget.QUANTIZED:
            return min(1.0, resource.quantized_queue_depth / 10.0)
        else:  # CLOUD
            return min(1.0, resource.cloud_queue_depth / 50.0)

    def update_weights(self, weights: dict[str, float]) -> None:
        self._weights.update(weights)
