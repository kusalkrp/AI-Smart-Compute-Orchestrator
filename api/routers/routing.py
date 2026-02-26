from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_session, verify_api_key
from models.database import ExecutionLog, RoutingDecision, RoutingPolicy
from models.schemas import RoutingPolicyUpdate, RoutingStatsResponse

router = APIRouter(prefix="/v1/routing", tags=["Routing"])


@router.get("/stats", response_model=RoutingStatsResponse)
async def get_routing_stats(
    hours: int = Query(default=24, ge=1, le=168),
    session: AsyncSession = Depends(get_session),
    _api_key: str = Depends(verify_api_key),
) -> RoutingStatsResponse:
    since = datetime.utcnow() - timedelta(hours=hours)

    # Routing decisions by target
    target_result = await session.execute(
        select(RoutingDecision.target, func.count(RoutingDecision.id))
        .where(RoutingDecision.created_at >= since)
        .group_by(RoutingDecision.target)
    )
    by_target = {row[0]: row[1] for row in target_result}

    # Execution logs for cost and latency
    logs_result = await session.execute(
        select(
            func.count(ExecutionLog.id),
            func.sum(ExecutionLog.actual_cost_usd),
            func.avg(ExecutionLog.actual_latency_ms),
        ).where(ExecutionLog.created_at >= since)
    )
    log_row = logs_result.one()
    total_tasks = log_row[0] or 0
    total_cost = float(log_row[1] or 0.0)
    avg_latency = float(log_row[2] or 0.0)

    # Estimate what cloud would have cost
    tokens_result = await session.execute(
        select(func.sum(ExecutionLog.tokens_used)).where(ExecutionLog.created_at >= since)
    )
    total_tokens = tokens_result.scalar() or 0
    from config.settings import settings
    estimated_cloud_cost = (total_tokens / 1000) * settings.CLOUD_COST_PER_1K_TOKENS
    cost_saved = max(0.0, estimated_cloud_cost - total_cost)

    return RoutingStatsResponse(
        total_tasks=total_tasks,
        by_target=by_target,
        by_status={},
        total_cost_usd=total_cost,
        estimated_cloud_cost_usd=estimated_cloud_cost,
        cost_saved_usd=cost_saved,
        avg_latency_ms=avg_latency,
        period_hours=hours,
    )


@router.get("/decision-log")
async def get_decision_log(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
    _api_key: str = Depends(verify_api_key),
) -> dict:
    offset = (page - 1) * page_size
    result = await session.execute(
        select(RoutingDecision)
        .order_by(RoutingDecision.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    decisions = result.scalars().all()

    count_result = await session.execute(select(func.count(RoutingDecision.id)))
    total = count_result.scalar() or 0

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "decisions": [
            {
                "id": str(d.id),
                "task_id": str(d.task_id),
                "target": d.target,
                "model_name": d.model_name,
                "estimated_cost_usd": d.estimated_cost_usd,
                "confidence": d.confidence,
                "decision_stage": d.decision_stage,
                "reasoning": d.reasoning,
                "created_at": d.created_at.isoformat(),
            }
            for d in decisions
        ],
    }


@router.post("/policy")
async def create_or_update_policy(
    policy: RoutingPolicyUpdate,
    session: AsyncSession = Depends(get_session),
    _api_key: str = Depends(verify_api_key),
) -> dict:
    result = await session.execute(
        select(RoutingPolicy).where(RoutingPolicy.name == policy.name)
    )
    existing = result.scalar_one_or_none()

    if existing:
        existing.description = policy.description
        existing.conditions = policy.conditions
        existing.target = policy.target.value
        existing.priority_order = policy.priority_order
        existing.is_active = policy.is_active
        db_policy = existing
    else:
        db_policy = RoutingPolicy(
            name=policy.name,
            description=policy.description,
            conditions=policy.conditions,
            target=policy.target.value,
            priority_order=policy.priority_order,
            is_active=policy.is_active,
        )
        session.add(db_policy)

    await session.commit()
    return {"status": "ok", "policy_id": str(db_policy.id), "name": db_policy.name}


@router.get("/policy")
async def list_policies(
    session: AsyncSession = Depends(get_session),
    _api_key: str = Depends(verify_api_key),
) -> dict:
    result = await session.execute(
        select(RoutingPolicy)
        .where(RoutingPolicy.is_active.is_(True))
        .order_by(RoutingPolicy.priority_order)
    )
    policies = result.scalars().all()
    return {
        "policies": [
            {
                "id": str(p.id),
                "name": p.name,
                "description": p.description,
                "conditions": p.conditions,
                "target": p.target,
                "priority_order": p.priority_order,
                "is_active": p.is_active,
            }
            for p in policies
        ]
    }
