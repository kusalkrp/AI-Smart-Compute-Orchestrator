from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from infrastructure.postgres_client import ping_db
from infrastructure.redis_client import ping_redis
from models.schemas import HealthResponse, ReadinessResponse

router = APIRouter(prefix="/v1/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
async def readiness() -> ReadinessResponse:
    redis_ok = await ping_redis()
    db_ok = await ping_db()

    checks = {"redis": redis_ok, "postgres": db_ok}
    status = "ready" if all(checks.values()) else "degraded"

    return ReadinessResponse(status=status, checks=checks)
