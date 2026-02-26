from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import redis.asyncio as aioredis
from redis.asyncio import Redis

from config.settings import settings

_redis_pool: Redis | None = None


async def get_redis() -> Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
    return _redis_pool


async def close_redis() -> None:
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None


@asynccontextmanager
async def redis_client() -> AsyncIterator[Redis]:
    client = await get_redis()
    try:
        yield client
    finally:
        pass  # Pool handles connection lifecycle


async def ping_redis() -> bool:
    try:
        client = await get_redis()
        return await client.ping()
    except Exception:
        return False


# ─── Stream Helpers ───────────────────────────────────────────────────────────

async def stream_add(stream_key: str, data: dict[str, Any]) -> str:
    client = await get_redis()
    # Redis streams require string values
    serialized = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in data.items()}
    msg_id = await client.xadd(stream_key, serialized)
    return msg_id


async def stream_read_group(
    stream_key: str,
    group_name: str,
    consumer_name: str,
    count: int = 1,
    block_ms: int = 1000,
) -> list[tuple[str, dict]]:
    client = await get_redis()
    try:
        messages = await client.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams={stream_key: ">"},
            count=count,
            block=block_ms,
        )
        if not messages:
            return []
        result = []
        for _stream, msgs in messages:
            for msg_id, fields in msgs:
                result.append((msg_id, fields))
        return result
    except Exception:
        return []


async def stream_ack(stream_key: str, group_name: str, msg_id: str) -> None:
    client = await get_redis()
    await client.xack(stream_key, group_name, msg_id)


async def ensure_consumer_group(stream_key: str, group_name: str) -> None:
    client = await get_redis()
    try:
        await client.xgroup_create(stream_key, group_name, id="0", mkstream=True)
    except aioredis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


async def get_stream_length(stream_key: str) -> int:
    client = await get_redis()
    try:
        return await client.xlen(stream_key)
    except Exception:
        return 0


# ─── Hash Helpers ─────────────────────────────────────────────────────────────

async def hash_set(key: str, field: str, value: Any) -> None:
    client = await get_redis()
    await client.hset(key, field, json.dumps(value) if not isinstance(value, str) else value)


async def hash_get(key: str, field: str) -> Any:
    client = await get_redis()
    val = await client.hget(key, field)
    if val is None:
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return val


async def hash_get_all(key: str) -> dict[str, Any]:
    client = await get_redis()
    raw = await client.hgetall(key)
    result = {}
    for k, v in raw.items():
        try:
            result[k] = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            result[k] = v
    return result


# ─── Key/Value Helpers ────────────────────────────────────────────────────────

async def set_with_ttl(key: str, value: Any, ttl_seconds: int) -> None:
    client = await get_redis()
    serialized = json.dumps(value) if not isinstance(value, str) else value
    await client.setex(key, ttl_seconds, serialized)


async def get_value(key: str) -> Any:
    client = await get_redis()
    val = await client.get(key)
    if val is None:
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return val
