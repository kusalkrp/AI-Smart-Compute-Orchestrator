from __future__ import annotations

import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.dependencies import get_session, verify_api_key
from models.database import RoutingDecision as DBRoutingDecision
from models.database import Task as DBTask
from models.enums import TaskStatus
from models.schemas import RoutingDecision, TaskRequest, TaskResponse

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1/tasks", tags=["Tasks"])


async def _process_task(task_id: uuid.UUID, task_request: TaskRequest) -> None:
    """Background processing: analyze → route → dispatch."""
    try:
        from core.intelligence.decision_engine import DecisionEngine
        from core.monitor.resource_monitor import resource_monitor
        from core.router.execution_router import ExecutionRouter
        from infrastructure.postgres_client import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            # Update status to ROUTING
            result = await session.execute(select(DBTask).where(DBTask.id == task_id))
            db_task = result.scalar_one_or_none()
            if not db_task:
                return

            db_task.status = TaskStatus.ROUTING.value
            await session.commit()

            # Get resource snapshot
            snapshot = await resource_monitor.get_current()

            # Get routing decision
            engine = DecisionEngine()
            decision = await engine.decide(task_id, task_request, snapshot)

            # Store decision in DB
            db_decision = DBRoutingDecision(
                task_id=task_id,
                target=decision.target.value,
                model_name=decision.model_name,
                estimated_cost_usd=decision.estimated_cost_usd,
                estimated_latency_ms=decision.estimated_latency_ms,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                fallback_target=decision.fallback_target.value,
                decision_stage=decision.decision_stage.value,
            )
            session.add(db_decision)
            db_task.status = TaskStatus.EXECUTING.value
            await session.commit()

            logger.info(
                "task.routed",
                task_id=str(task_id),
                target=decision.target.value,
                model=decision.model_name,
                stage=decision.decision_stage.value,
            )

        # Dispatch to worker queue
        from core.router.execution_router import ExecutionRouter
        exec_router = ExecutionRouter()
        await exec_router.dispatch(decision, task_request)

    except Exception as exc:
        logger.exception("task.routing_failed", task_id=str(task_id), error=str(exc))
        from infrastructure.postgres_client import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(DBTask).where(DBTask.id == task_id))
            db_task = result.scalar_one_or_none()
            if db_task:
                db_task.status = TaskStatus.FAILED.value
                await session.commit()


@router.post("", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_task(
    task_request: TaskRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    _api_key: str = Depends(verify_api_key),
) -> TaskResponse:
    task_id = uuid.uuid4()

    db_task = DBTask(
        id=task_id,
        task_type=task_request.task_type.value,
        priority=task_request.priority.value,
        input_text=task_request.input_text,
        status=TaskStatus.PENDING.value,
        is_batch=task_request.is_batch,
        metadata_=task_request.metadata,
        callback_url=task_request.callback_url,
    )
    session.add(db_task)
    await session.commit()

    background_tasks.add_task(_process_task, task_id, task_request)

    logger.info(
        "task.submitted",
        task_id=str(task_id),
        task_type=task_request.task_type.value,
        priority=task_request.priority.value,
    )

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=db_task.created_at,
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _api_key: str = Depends(verify_api_key),
) -> TaskResponse:
    result = await session.execute(
        select(DBTask)
        .options(
            selectinload(DBTask.routing_decision),
            selectinload(DBTask.execution_log),
        )
        .where(DBTask.id == task_id)
    )
    db_task = result.scalar_one_or_none()
    if not db_task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    routing_decision = None
    if db_task.routing_decision:
        rd = db_task.routing_decision
        routing_decision = RoutingDecision(
            task_id=task_id,
            target=rd.target,
            model_name=rd.model_name,
            estimated_cost_usd=rd.estimated_cost_usd,
            estimated_latency_ms=rd.estimated_latency_ms,
            confidence=rd.confidence,
            reasoning=rd.reasoning,
            fallback_target=rd.fallback_target,
            decision_stage=rd.decision_stage,
        )

    actual_cost = None
    actual_latency = None
    if db_task.execution_log:
        actual_cost = db_task.execution_log.actual_cost_usd
        actual_latency = db_task.execution_log.actual_latency_ms

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus(db_task.status),
        routing_decision=routing_decision,
        result=db_task.result,
        actual_cost_usd=actual_cost,
        actual_latency_ms=actual_latency,
        created_at=db_task.created_at,
        completed_at=db_task.updated_at if db_task.status == TaskStatus.COMPLETED.value else None,
    )


@router.post("/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(
    task_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _api_key: str = Depends(verify_api_key),
) -> TaskResponse:
    result = await session.execute(select(DBTask).where(DBTask.id == task_id))
    db_task = result.scalar_one_or_none()
    if not db_task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if db_task.status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task in status: {db_task.status}",
        )

    db_task.status = TaskStatus.CANCELLED.value
    await session.commit()

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.CANCELLED,
        created_at=db_task.created_at,
    )
