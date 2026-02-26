from __future__ import annotations

import structlog
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.auth import APIKeyMiddleware
from api.middleware.rate_limiter import SlidingWindowRateLimiter
from api.middleware.request_logger import RequestLoggerMiddleware
from api.routers import health, metrics, results, routing, tasks
from config.settings import settings
from infrastructure.postgres_client import create_tables
from infrastructure.redis_client import close_redis, ensure_consumer_group

logger = structlog.get_logger(__name__)

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("orchestrator.starting", version=settings.VERSION)

    # Initialize database tables
    try:
        await create_tables()
        logger.info("database.tables_ready")
    except Exception as e:
        logger.warning("database.init_failed", error=str(e))

    # Initialize Redis consumer groups
    try:
        stream_groups = [
            (settings.STREAM_TASKS_CPU, "workers:cpu"),
            (settings.STREAM_TASKS_GPU, "workers:gpu"),
            (settings.STREAM_TASKS_QUANTIZED, "workers:quantized"),
            (settings.STREAM_TASKS_CLOUD, "workers:cloud"),
            (settings.STREAM_RESULTS, "feedback:collector"),
        ]
        for stream, group in stream_groups:
            await ensure_consumer_group(stream, group)
        logger.info("redis.streams_ready")
    except Exception as e:
        logger.warning("redis.stream_init_failed", error=str(e))

    # Start background resource monitor
    try:
        from core.monitor.resource_monitor import resource_monitor
        import asyncio
        asyncio.create_task(resource_monitor.run())
        logger.info("resource_monitor.started")
    except Exception as e:
        logger.warning("resource_monitor.start_failed", error=str(e))

    logger.info("orchestrator.ready", host=settings.API_HOST, port=settings.API_PORT)
    yield

    logger.info("orchestrator.shutting_down")
    await close_redis()
    logger.info("orchestrator.stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Smart Compute Orchestrator",
        description="Intelligent AI workload routing across CPU, GPU, quantized models, and cloud APIs",
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Middleware (order matters: outermost runs first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggerMiddleware)
    app.add_middleware(SlidingWindowRateLimiter)
    app.add_middleware(APIKeyMiddleware)

    # Routers
    app.include_router(health.router)
    app.include_router(tasks.router)
    app.include_router(results.router)
    app.include_router(routing.router)
    app.include_router(metrics.router)

    return app


app = create_app()
