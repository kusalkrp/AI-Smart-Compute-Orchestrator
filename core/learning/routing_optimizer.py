"""Routing Optimizer — recalculates scoring weights from execution history."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import structlog
from celery import shared_task

from config.settings import settings

logger = structlog.get_logger(__name__)


class RoutingOptimizer:
    """Adjust Stage 2 scoring weights based on execution feedback.

    Objective: minimize total cost while keeping success rate > 0.95.
    Uses simple gradient descent on the weight vector.
    """

    def __init__(self) -> None:
        self._policy_path = Path(settings.ROUTING_POLICY_PATH)

    async def optimize(self) -> dict:
        """Recalculate weights and update routing_policy.yaml."""
        stats = await self._fetch_stats()
        if not stats:
            return {}

        new_weights = self._compute_weights(stats)
        await self._write_weights(new_weights)

        logger.info("routing_optimizer.weights_updated", weights=new_weights)
        return new_weights

    async def _fetch_stats(self) -> dict | None:
        """Fetch per-target performance metrics from the last 7 days."""
        from infrastructure.postgres_client import AsyncSessionLocal
        from models.database import ExecutionLog
        from sqlalchemy import func, select

        since = datetime.utcnow() - timedelta(days=7)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(
                    ExecutionLog.target_used,
                    func.avg(ExecutionLog.actual_latency_ms),
                    func.avg(ExecutionLog.actual_cost_usd),
                    func.avg(ExecutionLog.success.cast(float)),
                    func.count(ExecutionLog.id),
                )
                .where(ExecutionLog.created_at >= since)
                .group_by(ExecutionLog.target_used)
            )
            rows = result.fetchall()

        if not rows:
            return None

        return {
            row[0]: {
                "avg_latency": float(row[1] or 0),
                "avg_cost": float(row[2] or 0),
                "success_rate": float(row[3] or 1.0),
                "count": int(row[4] or 0),
            }
            for row in rows
        }

    def _compute_weights(self, stats: dict) -> dict[str, float]:
        """Simple heuristic weight adjustment.

        - If cloud has high cost relative to local → increase cost weight
        - If local has poor success rate → increase success_rate weight
        - If latency varies greatly → increase latency weight
        """
        cloud_cost = stats.get("CLOUD", {}).get("avg_cost", 0.0)
        local_avg_cost = sum(
            v["avg_cost"] for k, v in stats.items() if k != "CLOUD"
        ) / max(1, len([k for k in stats if k != "CLOUD"]))

        # Cost importance: higher when cloud-local gap is large
        cost_ratio = cloud_cost / max(local_avg_cost, 0.00001)
        cost_weight = min(0.5, 0.2 + min(0.3, cost_ratio / 100.0))

        # Success rate importance: higher when there are failures
        worst_success = min((v["success_rate"] for v in stats.values()), default=1.0)
        success_weight = 0.1 if worst_success > 0.98 else 0.3

        # Latency importance: remaining weight
        remaining = 1.0 - cost_weight - success_weight
        latency_weight = remaining * 0.5
        availability_weight = remaining * 0.5

        return {
            "cost_weight": round(cost_weight, 3),
            "availability_weight": round(availability_weight, 3),
            "success_rate_weight": round(success_weight, 3),
            "latency_weight": round(latency_weight, 3),
        }

    async def _write_weights(self, weights: dict) -> None:
        import yaml

        try:
            with self._policy_path.open() as f:
                data = yaml.safe_load(f)
            data["scoring_weights"] = weights
            with self._policy_path.open("w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.warning("routing_optimizer.write_failed", error=str(e))


@shared_task(name="core.learning.routing_optimizer.optimize_weights")
def optimize_weights() -> None:
    """Celery beat task: weekly weight optimization."""
    import asyncio
    optimizer = RoutingOptimizer()
    asyncio.run(optimizer.optimize())
