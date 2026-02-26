"""Analytics — query patterns: cost savings, latency trends, routing stats."""
from __future__ import annotations

from datetime import datetime, timedelta

import structlog
from celery import shared_task

logger = structlog.get_logger(__name__)


async def get_cost_savings(hours: int = 24) -> dict:
    """Compare actual hybrid cost vs hypothetical all-cloud cost."""
    from infrastructure.postgres_client import AsyncSessionLocal
    from models.database import ExecutionLog
    from sqlalchemy import func, select
    from config.settings import settings

    since = datetime.utcnow() - timedelta(hours=hours)

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(
                func.sum(ExecutionLog.actual_cost_usd),
                func.sum(ExecutionLog.tokens_used),
                func.count(ExecutionLog.id),
                func.avg(ExecutionLog.actual_latency_ms),
            ).where(ExecutionLog.created_at >= since)
        )
        row = result.one()
        total_actual_cost = float(row[0] or 0.0)
        total_tokens = int(row[1] or 0)
        total_tasks = int(row[2] or 0)
        avg_latency = float(row[3] or 0.0)

        cloud_cost = (total_tokens / 1000.0) * settings.CLOUD_COST_PER_1K_TOKENS
        savings = max(0.0, cloud_cost - total_actual_cost)

    return {
        "period_hours": hours,
        "total_tasks": total_tasks,
        "actual_cost_usd": total_actual_cost,
        "estimated_cloud_cost_usd": cloud_cost,
        "cost_saved_usd": savings,
        "savings_percent": (savings / cloud_cost * 100) if cloud_cost > 0 else 0.0,
        "avg_latency_ms": avg_latency,
    }


async def get_latency_trends(hours: int = 24, bucket_minutes: int = 30) -> list[dict]:
    """Return time-bucketed latency data for charts."""
    from infrastructure.postgres_client import AsyncSessionLocal
    from models.database import ExecutionLog
    from sqlalchemy import func, select, text

    since = datetime.utcnow() - timedelta(hours=hours)

    async with AsyncSessionLocal() as session:
        # Use date_trunc for PostgreSQL bucketing
        result = await session.execute(
            text("""
                SELECT
                    date_trunc('hour', created_at) +
                    (EXTRACT(MINUTE FROM created_at)::int / :bucket * :bucket || ' minutes')::interval AS bucket,
                    target_used,
                    AVG(actual_latency_ms) AS avg_latency,
                    COUNT(*) AS count
                FROM execution_logs
                WHERE created_at >= :since
                GROUP BY bucket, target_used
                ORDER BY bucket, target_used
            """),
            {"since": since, "bucket": bucket_minutes},
        )
        rows = result.fetchall()

    return [
        {
            "bucket": row[0].isoformat() if row[0] else None,
            "target": row[1],
            "avg_latency_ms": float(row[2] or 0),
            "count": int(row[3] or 0),
        }
        for row in rows
    ]


async def get_routing_distribution(hours: int = 24) -> dict[str, int]:
    """Return routing decisions by target for the given period."""
    from infrastructure.postgres_client import AsyncSessionLocal
    from models.database import RoutingDecision
    from sqlalchemy import func, select

    since = datetime.utcnow() - timedelta(hours=hours)

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(RoutingDecision.target, func.count(RoutingDecision.id))
            .where(RoutingDecision.created_at >= since)
            .group_by(RoutingDecision.target)
        )
        return {row[0]: row[1] for row in result}


@shared_task(name="core.learning.analytics.update_performance_cache")
def update_performance_cache() -> None:
    """Celery task: refresh model performance cache from DB."""
    import asyncio

    async def _run():
        try:
            from infrastructure.postgres_client import AsyncSessionLocal
            from infrastructure.redis_client import hash_set
            from models.database import ModelPerformance
            from sqlalchemy import select
            from config.settings import settings

            async with AsyncSessionLocal() as session:
                result = await session.execute(select(ModelPerformance))
                perfs = result.scalars().all()

            for perf in perfs:
                key = settings.MODEL_PERFORMANCE_KEY
                field = f"{perf.target}:{perf.model_name}"
                await hash_set(key, field, {
                    "avg_latency_ms": perf.avg_latency_ms,
                    "avg_cost_usd": perf.avg_cost_usd,
                    "success_rate": perf.success_rate,
                    "sample_count": perf.sample_count,
                })
        except Exception as e:
            logger.warning("analytics.cache_update_failed", error=str(e))

    asyncio.run(_run())
