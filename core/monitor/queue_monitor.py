from __future__ import annotations

from config.settings import settings
from infrastructure.redis_client import get_stream_length


class QueueMonitor:
    async def get_depths(self) -> dict[str, int]:
        cpu_depth = await get_stream_length(settings.STREAM_TASKS_CPU)
        gpu_depth = await get_stream_length(settings.STREAM_TASKS_GPU)
        quantized_depth = await get_stream_length(settings.STREAM_TASKS_QUANTIZED)
        cloud_depth = await get_stream_length(settings.STREAM_TASKS_CLOUD)

        return {
            "cpu_queue_depth": cpu_depth,
            "gpu_queue_depth": gpu_depth,
            "quantized_queue_depth": quantized_depth,
            "cloud_queue_depth": cloud_depth,
        }
