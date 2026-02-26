from __future__ import annotations

import structlog

from config.settings import settings
from core.router.fallback_handler import FallbackHandler
from core.router.load_balancer import LoadBalancer
from infrastructure.redis_client import stream_add
from models.enums import ExecutionTarget
from models.schemas import RoutingDecision, TaskRequest

logger = structlog.get_logger(__name__)

_STREAM_MAP = {
    ExecutionTarget.CPU: settings.STREAM_TASKS_CPU,
    ExecutionTarget.GPU: settings.STREAM_TASKS_GPU,
    ExecutionTarget.QUANTIZED: settings.STREAM_TASKS_QUANTIZED,
    ExecutionTarget.CLOUD: settings.STREAM_TASKS_CLOUD,
}


class ExecutionRouter:
    def __init__(self) -> None:
        self._fallback = FallbackHandler()
        self._balancer = LoadBalancer()

    async def dispatch(
        self,
        decision: RoutingDecision,
        request: TaskRequest,
    ) -> None:
        """Add task to the appropriate Redis Stream worker queue."""
        stream_key = _STREAM_MAP.get(decision.target, settings.STREAM_TASKS_CPU)

        payload = {
            "task_id": str(decision.task_id),
            "model": decision.model_name,
            "input": request.input_text,
            "task_type": request.task_type.value,
            "priority": request.priority.value,
            "fallback": decision.fallback_target.value,
            "max_cost_usd": str(request.max_cost_usd or ""),
            "max_latency_ms": str(request.max_latency_ms or ""),
            "callback_url": request.callback_url or "",
        }

        msg_id = await stream_add(stream_key, payload)

        logger.info(
            "execution_router.dispatched",
            task_id=str(decision.task_id),
            target=decision.target.value,
            stream=stream_key,
            msg_id=msg_id,
        )
