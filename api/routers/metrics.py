from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_session
from models.database import ExecutionLog, RoutingDecision, Task

router = APIRouter(prefix="/v1", tags=["Metrics"])


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(session: AsyncSession = Depends(get_session)) -> str:
    lines = []
    since_1h = datetime.utcnow() - timedelta(hours=1)

    # Task counts by status
    status_result = await session.execute(
        select(Task.status, func.count(Task.id)).group_by(Task.status)
    )
    lines.append("# HELP orchestrator_tasks_total Total number of tasks by status")
    lines.append("# TYPE orchestrator_tasks_total counter")
    for status, count in status_result:
        lines.append(f'orchestrator_tasks_total{{status="{status}"}} {count}')

    # Task counts by target (from routing decisions)
    target_result = await session.execute(
        select(RoutingDecision.target, func.count(RoutingDecision.id)).group_by(RoutingDecision.target)
    )
    lines.append("# HELP orchestrator_tasks_by_target Tasks routed per execution target")
    lines.append("# TYPE orchestrator_tasks_by_target counter")
    for target, count in target_result:
        lines.append(f'orchestrator_tasks_by_target{{target="{target}"}} {count}')

    # Avg latency by target (last 1 hour)
    latency_result = await session.execute(
        select(
            ExecutionLog.target_used,
            func.avg(ExecutionLog.actual_latency_ms),
            func.count(ExecutionLog.id),
        )
        .where(ExecutionLog.created_at >= since_1h)
        .group_by(ExecutionLog.target_used)
    )
    lines.append("# HELP orchestrator_avg_latency_ms Average execution latency in milliseconds")
    lines.append("# TYPE orchestrator_avg_latency_ms gauge")
    for target, avg_lat, count in latency_result:
        if avg_lat is not None:
            lines.append(f'orchestrator_avg_latency_ms{{target="{target}"}} {avg_lat:.2f}')

    # Total cost saved
    cost_result = await session.execute(
        select(func.sum(ExecutionLog.actual_cost_usd), func.sum(ExecutionLog.tokens_used))
    )
    total_cost_row = cost_result.one()
    total_actual_cost = float(total_cost_row[0] or 0.0)
    total_tokens = int(total_cost_row[1] or 0)

    from config.settings import settings
    hypothetical_cloud_cost = (total_tokens / 1000) * settings.CLOUD_COST_PER_1K_TOKENS
    cost_saved = max(0.0, hypothetical_cloud_cost - total_actual_cost)

    lines.append("# HELP orchestrator_cost_saved_usd_total Total estimated cost savings in USD")
    lines.append("# TYPE orchestrator_cost_saved_usd_total counter")
    lines.append(f"orchestrator_cost_saved_usd_total {cost_saved:.6f}")

    lines.append("# HELP orchestrator_total_cost_usd_total Total actual execution cost in USD")
    lines.append("# TYPE orchestrator_total_cost_usd_total counter")
    lines.append(f"orchestrator_total_cost_usd_total {total_actual_cost:.6f}")

    # Resource metrics from Redis
    try:
        from core.monitor.resource_monitor import resource_monitor
        snapshot = await resource_monitor.get_current()
        lines.append("# HELP orchestrator_cpu_percent Current CPU utilization percent")
        lines.append("# TYPE orchestrator_cpu_percent gauge")
        lines.append(f"orchestrator_cpu_percent {snapshot.cpu_percent:.1f}")

        lines.append("# HELP orchestrator_gpu_percent Current GPU utilization percent")
        lines.append("# TYPE orchestrator_gpu_percent gauge")
        lines.append(f"orchestrator_gpu_percent {snapshot.gpu_percent:.1f}")

        lines.append("# HELP orchestrator_ram_percent Current RAM utilization percent")
        lines.append("# TYPE orchestrator_ram_percent gauge")
        lines.append(f"orchestrator_ram_percent {snapshot.ram_percent:.1f}")
    except Exception:
        pass

    return "\n".join(lines) + "\n"
