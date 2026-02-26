from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Bind request context to logger
        bound_logger = logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        bound_logger.info("request.started")

        try:
            response = await call_next(request)
            elapsed_ms = int((time.time() - start_time) * 1000)

            bound_logger.info(
                "request.completed",
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
            )

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
            return response

        except Exception as exc:
            elapsed_ms = int((time.time() - start_time) * 1000)
            bound_logger.exception("request.failed", elapsed_ms=elapsed_ms, error=str(exc))
            raise
