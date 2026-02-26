from __future__ import annotations

import time

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from infrastructure.redis_client import get_redis

RATE_LIMIT_REQUESTS = 100  # per minute
RATE_LIMIT_WINDOW_SEC = 60
PUBLIC_PATHS = {"/v1/health", "/v1/health/ready", "/v1/metrics"}


class SlidingWindowRateLimiter(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "anonymous")
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW_SEC

        try:
            redis = await get_redis()
            key = f"rate_limit:{api_key}"

            # Remove old entries outside window
            await redis.zremrangebyscore(key, 0, window_start)

            # Count current requests in window
            count = await redis.zcard(key)
            if count >= RATE_LIMIT_REQUESTS:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per minute"
                    },
                    headers={"Retry-After": str(RATE_LIMIT_WINDOW_SEC)},
                )

            # Add this request timestamp
            await redis.zadd(key, {str(now): now})
            await redis.expire(key, RATE_LIMIT_WINDOW_SEC * 2)
        except Exception:
            # If Redis is down, allow the request through
            pass

        response = await call_next(request)
        return response
