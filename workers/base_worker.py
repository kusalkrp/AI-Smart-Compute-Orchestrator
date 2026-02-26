from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional

import structlog

from config.settings import settings
from infrastructure.redis_client import (
    ensure_consumer_group,
    get_redis,
    stream_ack,
    stream_add,
    stream_read_group,
)
from models.enums import ExecutionTarget, TaskStatus
from models.schemas import ExecutionResult

logger = structlog.get_logger(__name__)


class BaseWorker(ABC):
    """Abstract worker base class.

    Subclasses implement `execute()` to process a task message.
    This base handles:
    - Redis Streams consumer group management
    - Message polling loop
    - Result publishing
    - DB task status updates
    - Error handling and fallback
    """

    stream_key: str
    group_name: str
    target: ExecutionTarget
    consumer_name: str

    def __init__(self) -> None:
        self._running = False

    async def start(self) -> None:
        await ensure_consumer_group(self.stream_key, self.group_name)
        self._running = True
        logger.info("worker.started", target=self.target.value, stream=self.stream_key)
        await self._loop()

    def stop(self) -> None:
        self._running = False
        logger.info("worker.stopping", target=self.target.value)

    async def _loop(self) -> None:
        while self._running:
            try:
                messages = await stream_read_group(
                    stream_key=self.stream_key,
                    group_name=self.group_name,
                    consumer_name=self.consumer_name,
                    count=1,
                    block_ms=2000,
                )
                for msg_id, fields in messages:
                    await self._handle_message(msg_id, fields)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("worker.loop_error", target=self.target.value, error=str(e))
                await asyncio.sleep(1)

    async def _handle_message(self, msg_id: str, fields: dict) -> None:
        task_id_str = fields.get("task_id", "")
        try:
            task_id = uuid.UUID(task_id_str)
        except ValueError:
            logger.warning("worker.invalid_task_id", msg_id=msg_id, raw=task_id_str)
            await stream_ack(self.stream_key, self.group_name, msg_id)
            return

        logger.info("worker.processing", task_id=task_id_str, target=self.target.value)
        start_time = time.time()

        try:
            result = await self.execute(task_id, fields)
            elapsed_ms = int((time.time() - start_time) * 1000)
            result.actual_latency_ms = elapsed_ms

            await self._publish_result(result)
            await self._update_task_complete(task_id, result)
            await stream_ack(self.stream_key, self.group_name, msg_id)

            logger.info(
                "worker.completed",
                task_id=task_id_str,
                target=self.target.value,
                latency_ms=elapsed_ms,
                cost=result.actual_cost_usd,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.exception("worker.execution_failed", task_id=task_id_str, error=str(e))

            # Try fallback
            fallback_str = fields.get("fallback", "CLOUD")
            await self._handle_fallback(task_id, fields, fallback_str, str(e))
            await stream_ack(self.stream_key, self.group_name, msg_id)

    async def _publish_result(self, result: ExecutionResult) -> None:
        try:
            payload = {
                "task_id": str(result.task_id),
                "target_used": result.target_used.value,
                "model_used": result.model_used,
                "actual_latency_ms": str(result.actual_latency_ms),
                "actual_cost_usd": str(result.actual_cost_usd),
                "tokens_used": str(result.tokens_used),
                "success": "true" if result.success else "false",
                "error_message": result.error_message or "",
                "output_text": result.output_text[:10000],  # Truncate large outputs
            }
            await stream_add(settings.STREAM_RESULTS, payload)
        except Exception as e:
            logger.warning("worker.publish_result_failed", error=str(e))

    async def _update_task_complete(self, task_id: uuid.UUID, result: ExecutionResult) -> None:
        try:
            from infrastructure.postgres_client import AsyncSessionLocal
            from models.database import ExecutionLog, Task as DBTask
            from sqlalchemy import select

            async with AsyncSessionLocal() as session:
                task_result = await session.execute(select(DBTask).where(DBTask.id == task_id))
                db_task = task_result.scalar_one_or_none()
                if db_task:
                    db_task.status = TaskStatus.COMPLETED.value if result.success else TaskStatus.FAILED.value
                    db_task.result = result.output_text if result.success else None

                    log = ExecutionLog(
                        task_id=task_id,
                        target_used=result.target_used.value,
                        model_used=result.model_used,
                        actual_latency_ms=result.actual_latency_ms,
                        actual_cost_usd=result.actual_cost_usd,
                        tokens_used=result.tokens_used,
                        success=result.success,
                        error_message=result.error_message,
                    )
                    session.add(log)
                    await session.commit()

                    # Trigger callback webhook if set
                    if db_task.callback_url and result.success:
                        await self._deliver_callback(db_task.callback_url, task_id, result)

        except Exception as e:
            logger.warning("worker.db_update_failed", task_id=str(task_id), error=str(e))

    async def _deliver_callback(
        self,
        url: str,
        task_id: uuid.UUID,
        result: ExecutionResult,
    ) -> None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(url, json={
                    "task_id": str(task_id),
                    "status": "COMPLETED",
                    "target_used": result.target_used.value,
                    "actual_latency_ms": result.actual_latency_ms,
                    "actual_cost_usd": result.actual_cost_usd,
                })
        except Exception as e:
            logger.warning("worker.callback_failed", url=url, error=str(e))

    async def _handle_fallback(
        self,
        task_id: uuid.UUID,
        original_fields: dict,
        fallback_target_str: str,
        error_msg: str,
    ) -> None:
        from config.settings import settings as cfg
        from models.enums import ExecutionTarget
        from infrastructure.redis_client import stream_add

        _stream_map = {
            "CPU": cfg.STREAM_TASKS_CPU,
            "GPU": cfg.STREAM_TASKS_GPU,
            "QUANTIZED": cfg.STREAM_TASKS_QUANTIZED,
            "CLOUD": cfg.STREAM_TASKS_CLOUD,
        }

        # Don't loop back to ourselves
        if fallback_target_str == self.target.value:
            fallback_target_str = "CLOUD"

        stream_key = _stream_map.get(fallback_target_str, cfg.STREAM_TASKS_CLOUD)

        logger.info(
            "worker.falling_back",
            task_id=str(task_id),
            from_target=self.target.value,
            to_target=fallback_target_str,
            reason=error_msg,
        )

        new_fields = {**original_fields, "fallback": "CLOUD", "fallback_reason": error_msg[:500]}
        await stream_add(stream_key, new_fields)

    @abstractmethod
    async def execute(self, task_id: uuid.UUID, fields: dict) -> ExecutionResult:
        """Process the task and return an ExecutionResult."""
        ...
