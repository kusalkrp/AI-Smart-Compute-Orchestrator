from __future__ import annotations

import asyncio
import json
from datetime import datetime

import structlog

from config.settings import settings
from core.monitor.cpu_monitor import CPUMonitor
from core.monitor.gpu_monitor import GPUMonitor
from core.monitor.queue_monitor import QueueMonitor
from models.schemas import ResourceSnapshot

logger = structlog.get_logger(__name__)

_DEFAULT_SNAPSHOT = ResourceSnapshot(
    cpu_percent=0.0,
    ram_percent=0.0,
    gpu_percent=0.0,
    gpu_vram_used_mb=0.0,
    gpu_vram_total_mb=0.0,
    gpu_available=False,
)


class ResourceMonitor:
    """
    Aggregates CPU, GPU, and queue state into a ResourceSnapshot.
    Runs as a background async task, publishing to Redis every N seconds.
    Stores to DB every 30 seconds for historical trending.
    """

    def __init__(self) -> None:
        self._cpu = CPUMonitor()
        self._gpu = GPUMonitor()
        self._queue = QueueMonitor()
        self._current: ResourceSnapshot = _DEFAULT_SNAPSHOT
        self._db_counter = 0
        self._running = False

    async def get_current(self) -> ResourceSnapshot:
        """Return the latest cached snapshot, or fetch fresh if not started."""
        try:
            from infrastructure.redis_client import get_value
            cached = await get_value(settings.RESOURCE_CURRENT_KEY)
            if cached:
                return ResourceSnapshot(**cached)
        except Exception:
            pass
        return self._current

    async def _collect(self) -> ResourceSnapshot:
        cpu_data = self._cpu.snapshot()
        gpu_data = self._gpu.snapshot()
        queue_data = await self._queue.get_depths()

        return ResourceSnapshot(
            cpu_percent=cpu_data["cpu_percent"],
            ram_percent=cpu_data["ram_percent"],
            gpu_percent=gpu_data["gpu_percent"],
            gpu_vram_used_mb=gpu_data["gpu_vram_used_mb"],
            gpu_vram_total_mb=gpu_data["gpu_vram_total_mb"],
            gpu_available=gpu_data["gpu_available"],
            cpu_queue_depth=queue_data["cpu_queue_depth"],
            gpu_queue_depth=queue_data["gpu_queue_depth"],
            quantized_queue_depth=queue_data["quantized_queue_depth"],
            cloud_queue_depth=queue_data["cloud_queue_depth"],
            captured_at=datetime.utcnow(),
        )

    async def _publish_to_redis(self, snapshot: ResourceSnapshot) -> None:
        try:
            from infrastructure.redis_client import set_with_ttl
            data = snapshot.model_dump()
            data["captured_at"] = data["captured_at"].isoformat()
            await set_with_ttl(settings.RESOURCE_CURRENT_KEY, data, ttl_seconds=10)
        except Exception as e:
            logger.warning("resource_monitor.redis_publish_failed", error=str(e))

    async def _store_to_db(self, snapshot: ResourceSnapshot) -> None:
        try:
            from infrastructure.postgres_client import AsyncSessionLocal
            from models.database import ResourceSnapshot as DBSnapshot

            async with AsyncSessionLocal() as session:
                db_snap = DBSnapshot(
                    cpu_percent=snapshot.cpu_percent,
                    ram_percent=snapshot.ram_percent,
                    gpu_percent=snapshot.gpu_percent,
                    gpu_vram_used_mb=snapshot.gpu_vram_used_mb,
                    gpu_vram_total_mb=snapshot.gpu_vram_total_mb,
                    cpu_queue_depth=snapshot.cpu_queue_depth,
                    gpu_queue_depth=snapshot.gpu_queue_depth,
                    quantized_queue_depth=snapshot.quantized_queue_depth,
                    cloud_queue_depth=snapshot.cloud_queue_depth,
                )
                session.add(db_snap)
                await session.commit()
        except Exception as e:
            logger.warning("resource_monitor.db_store_failed", error=str(e))

    async def run(self) -> None:
        self._running = True
        interval = settings.RESOURCE_MONITOR_INTERVAL_SEC
        db_interval_ticks = settings.RESOURCE_SNAPSHOT_DB_INTERVAL_SEC // interval

        logger.info("resource_monitor.running", interval_sec=interval)

        while self._running:
            try:
                snapshot = await self._collect()
                self._current = snapshot
                await self._publish_to_redis(snapshot)

                self._db_counter += 1
                if self._db_counter >= db_interval_ticks:
                    await self._store_to_db(snapshot)
                    self._db_counter = 0

            except Exception as e:
                logger.warning("resource_monitor.collect_failed", error=str(e))

            await asyncio.sleep(interval)

    def stop(self) -> None:
        self._running = False


# Singleton
resource_monitor = ResourceMonitor()
