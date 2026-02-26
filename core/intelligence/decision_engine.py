from __future__ import annotations

import uuid
from typing import Optional

import structlog

from config.settings import settings
from core.analyzer.workload_analyzer import WorkloadAnalyzer
from core.intelligence.cost_calculator import CostCalculator
from core.intelligence.ml_engine import MLEngine
from core.intelligence.rule_engine import RuleEngine
from core.intelligence.scoring_engine import ScoringEngine
from models.enums import DecisionStage, ExecutionTarget, Priority
from models.schemas import ExecutionProfile, ResourceSnapshot, RoutingDecision, TaskRequest

logger = structlog.get_logger(__name__)


class DecisionEngine:
    """
    Core orchestration brain.

    Three-stage cascade (selectable via ROUTING_STAGE env var):
    - Stage 1 (rule): YAML-driven rule evaluation
    - Stage 2 (scored): Weighted multi-factor scoring
    - Stage 3 (ml): XGBoost classifier, falls back to Stage 2 if confidence < threshold
    """

    def __init__(self) -> None:
        self._analyzer = WorkloadAnalyzer()
        self._rule_engine = RuleEngine()
        self._scoring_engine = ScoringEngine()
        self._ml_engine = MLEngine()
        self._cost_calc = CostCalculator()

    async def decide(
        self,
        task_id: uuid.UUID,
        request: TaskRequest,
        resource: ResourceSnapshot,
    ) -> RoutingDecision:
        # Step 1: Analyze workload
        profile = self._analyzer.analyze(task_id, request)

        logger.debug(
            "decision_engine.profile",
            task_id=str(task_id),
            complexity=profile.complexity_score,
            tokens=profile.estimated_tokens,
            latency_sens=profile.latency_sensitivity,
        )

        # Step 2: Route through configured stage
        stage = settings.ROUTING_STAGE
        target, model_name, confidence, reasoning, decision_stage = await self._route(
            profile, resource, request.priority, stage
        )

        # Step 3: Determine fallback
        fallback = self._rule_engine.get_fallback(target)

        # Step 4: Estimate cost and latency
        estimated_cost = self._cost_calc.estimate_cost(target, model_name, profile.estimated_tokens)
        estimated_latency = self._cost_calc.estimate_latency(target, model_name, profile.estimated_tokens)

        decision = RoutingDecision(
            task_id=task_id,
            target=target,
            model_name=model_name,
            estimated_cost_usd=estimated_cost,
            estimated_latency_ms=estimated_latency,
            confidence=confidence,
            reasoning=reasoning,
            fallback_target=fallback,
            decision_stage=decision_stage,
        )

        logger.info(
            "decision_engine.decided",
            task_id=str(task_id),
            target=target.value,
            model=model_name,
            stage=decision_stage.value,
            confidence=confidence,
            estimated_cost=estimated_cost,
        )

        return decision

    async def _route(
        self,
        profile: ExecutionProfile,
        resource: ResourceSnapshot,
        priority: Priority,
        stage: str,
    ) -> tuple[ExecutionTarget, str, float, str, DecisionStage]:
        if stage == "ml" and self._ml_engine.is_available():
            try:
                target, model, confidence = self._ml_engine.predict(profile, resource)
                if confidence >= settings.ML_CONFIDENCE_THRESHOLD:
                    reasoning = (
                        f"ML model selected {target.value} with {confidence:.0%} confidence. "
                        f"Complexity: {profile.complexity_score:.2f}, "
                        f"Tokens: {profile.estimated_tokens}"
                    )
                    return target, model, confidence, reasoning, DecisionStage.ML_MODEL
                else:
                    logger.info(
                        "ml_engine.low_confidence_fallback",
                        confidence=confidence,
                        threshold=settings.ML_CONFIDENCE_THRESHOLD,
                    )
                    # Fall through to Stage 2
            except Exception as e:
                logger.warning("ml_engine.failed", error=str(e))

        if stage in ("scored", "ml"):
            # Load performance cache from Redis
            performance_cache = await self._load_performance_cache()
            target, model, confidence = self._scoring_engine.score(profile, resource, performance_cache)
            reasoning = (
                f"Scoring engine selected {target.value} (score-based). "
                f"Cost sensitivity: {profile.cost_sensitivity:.2f}, "
                f"Latency sensitivity: {profile.latency_sensitivity:.2f}, "
                f"GPU available: {resource.gpu_available}"
            )
            return target, model, confidence, reasoning, DecisionStage.SCORED

        # Stage 1: Rule-based (default)
        target, model, rule_name = self._rule_engine.evaluate(profile, resource, priority)
        reasoning = (
            f"Rule '{rule_name}' matched: target={target.value}, "
            f"priority={priority.value}, complexity={profile.complexity_score:.2f}, "
            f"gpu_available={resource.gpu_available}, gpu_util={resource.gpu_percent:.1f}%"
        )
        return target, model, 1.0, reasoning, DecisionStage.RULE_BASED

    async def _load_performance_cache(self) -> dict:
        try:
            from infrastructure.redis_client import hash_get_all
            from config.settings import settings as cfg
            return await hash_get_all(cfg.MODEL_PERFORMANCE_KEY)
        except Exception:
            return {}
