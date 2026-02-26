"""Feedback Collector — consumes stream:results and stores ExecutionLog records."""
from __future__ import annotations

import asyncio
import uuid

import structlog

from config.settings import settings
from infrastructure.redis_client import ensure_consumer_group, stream_ack, stream_read_group

logger = structlog.get_logger(__name__)


class FeedbackCollector:
    """
    Runs as a background async task.
    Consumes stream:results → stores ExecutionLog, updates ModelPerformance rolling averages.
    """

    stream_key = settings.STREAM_RESULTS
    group_name = "feedback:collector"
    consumer_name = "feedback-collector-1"

    def __init__(self) -> None:
        self._running = False

    async def start(self) -> None:
        await ensure_consumer_group(self.stream_key, self.group_name)
        self._running = True
        logger.info("feedback_collector.started")
        await self._loop()

    def stop(self) -> None:
        self._running = False

    async def _loop(self) -> None:
        while self._running:
            try:
                messages = await stream_read_group(
                    stream_key=self.stream_key,
                    group_name=self.group_name,
                    consumer_name=self.consumer_name,
                    count=10,
                    block_ms=2000,
                )
                for msg_id, fields in messages:
                    try:
                        await self._process(fields)
                    except Exception as e:
                        logger.warning("feedback_collector.process_error", error=str(e))
                    finally:
                        await stream_ack(self.stream_key, self.group_name, msg_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("feedback_collector.loop_error", error=str(e))
                await asyncio.sleep(2)

    async def _process(self, fields: dict) -> None:
        task_id_str = fields.get("task_id", "")
        target_used = fields.get("target_used", "CPU")
        model_used = fields.get("model_used", "unknown")
        actual_latency_ms = int(fields.get("actual_latency_ms", 0))
        actual_cost_usd = float(fields.get("actual_cost_usd", 0.0))
        tokens_used = int(fields.get("tokens_used", 0))
        success = fields.get("success", "true").lower() == "true"

        await self._update_model_performance(
            target=target_used,
            model=model_used,
            latency_ms=actual_latency_ms,
            cost_usd=actual_cost_usd,
            success=success,
        )

        logger.debug(
            "feedback_collector.processed",
            task_id=task_id_str,
            target=target_used,
            latency_ms=actual_latency_ms,
            cost=actual_cost_usd,
        )

    async def _update_model_performance(
        self,
        target: str,
        model: str,
        latency_ms: int,
        cost_usd: float,
        success: bool,
    ) -> None:
        """Update rolling averages in Redis hash and PostgreSQL."""
        try:
            from infrastructure.redis_client import hash_get, hash_set

            key = settings.MODEL_PERFORMANCE_KEY
            field = f"{target}:{model}"

            current = await hash_get(key, field)
            if current:
                n = current.get("sample_count", 0)
                avg_lat = current.get("avg_latency_ms", 0.0)
                avg_cost = current.get("avg_cost_usd", 0.0)
                rate = current.get("success_rate", 1.0)
            else:
                n = 0
                avg_lat = 0.0
                avg_cost = 0.0
                rate = 1.0

            # Exponential moving average (alpha=0.1)
            alpha = 0.1
            if n == 0:
                new_lat = float(latency_ms)
                new_cost = cost_usd
                new_rate = 1.0 if success else 0.0
            else:
                new_lat = (1 - alpha) * avg_lat + alpha * latency_ms
                new_cost = (1 - alpha) * avg_cost + alpha * cost_usd
                new_rate = (1 - alpha) * rate + alpha * (1.0 if success else 0.0)

            updated = {
                "avg_latency_ms": new_lat,
                "avg_cost_usd": new_cost,
                "success_rate": new_rate,
                "sample_count": n + 1,
            }
            await hash_set(key, field, updated)

        except Exception as e:
            logger.warning("feedback_collector.performance_update_failed", error=str(e))

        # Also persist to DB (periodic batch would be better, but this works for demo)
        try:
            from infrastructure.postgres_client import AsyncSessionLocal
            from models.database import ModelPerformance
            from sqlalchemy import select

            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ModelPerformance).where(
                        ModelPerformance.model_name == model,
                        ModelPerformance.target == target,
                    )
                )
                perf = result.scalar_one_or_none()
                if perf:
                    n = perf.sample_count
                    alpha = 0.1
                    perf.avg_latency_ms = (1 - alpha) * perf.avg_latency_ms + alpha * latency_ms
                    perf.avg_cost_usd = (1 - alpha) * perf.avg_cost_usd + alpha * cost_usd
                    perf.success_rate = (1 - alpha) * perf.success_rate + alpha * (1.0 if success else 0.0)
                    perf.sample_count = n + 1
                else:
                    perf = ModelPerformance(
                        model_name=model,
                        target=target,
                        avg_latency_ms=float(latency_ms),
                        avg_cost_usd=cost_usd,
                        success_rate=1.0 if success else 0.0,
                        sample_count=1,
                    )
                    session.add(perf)
                await session.commit()
        except Exception as e:
            logger.warning("feedback_collector.db_update_failed", error=str(e))


# Singleton
feedback_collector = FeedbackCollector()
