from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.dependencies import get_session, verify_api_key
from models.database import Task as DBTask
from models.enums import TaskStatus
from models.schemas import TaskResponse

router = APIRouter(prefix="/v1/tasks", tags=["Results"])


@router.get("/{task_id}/result", response_model=TaskResponse)
async def get_task_result(
    task_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _api_key: str = Depends(verify_api_key),
) -> TaskResponse:
    result = await session.execute(
        select(DBTask)
        .options(selectinload(DBTask.execution_log))
        .where(DBTask.id == task_id)
    )
    db_task = result.scalar_one_or_none()
    if not db_task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if db_task.status != TaskStatus.COMPLETED.value:
        raise HTTPException(
            status_code=202,
            detail=f"Task not yet completed. Current status: {db_task.status}",
        )

    actual_cost = None
    actual_latency = None
    if db_task.execution_log:
        actual_cost = db_task.execution_log.actual_cost_usd
        actual_latency = db_task.execution_log.actual_latency_ms

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus(db_task.status),
        result=db_task.result,
        actual_cost_usd=actual_cost,
        actual_latency_ms=actual_latency,
        created_at=db_task.created_at,
        completed_at=db_task.updated_at,
    )
