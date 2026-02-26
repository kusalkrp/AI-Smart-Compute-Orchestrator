"""Cloud Worker — calls Google Gemini API with async client and exponential backoff retry."""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

import structlog

from config.settings import settings
from models.enums import ExecutionTarget
from models.schemas import ExecutionResult
from workers.base_worker import BaseWorker

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0


class CloudWorker(BaseWorker):
    stream_key = settings.STREAM_TASKS_CLOUD
    group_name = "workers:cloud"
    target = ExecutionTarget.CLOUD
    consumer_name = "cloud-worker-1"

    def __init__(self) -> None:
        super().__init__()
        self._client: Optional[object] = None
        self._setup_client()

    def _setup_client(self) -> None:
        if not settings.GEMINI_API_KEY:
            logger.warning("cloud_worker.no_api_key", provider="gemini")
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
            logger.info("cloud_worker.client_ready", provider="gemini")
        except ImportError:
            logger.warning("cloud_worker.google_genai_not_installed")

    async def execute(self, task_id: uuid.UUID, fields: dict) -> ExecutionResult:
        if self._client is None:
            raise RuntimeError("Gemini client not configured — check GEMINI_API_KEY")

        model = fields.get("model", settings.CLOUD_MODEL)
        input_text = fields.get("input", "")

        start = time.time()

        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.aio.models.generate_content(
                    model=model,
                    contents=input_text,
                    config={
                        "temperature": 0.7,
                        "max_output_tokens": 512,
                    },
                )
                elapsed_ms = int((time.time() - start) * 1000)

                output = response.text or ""

                # Extract token usage from response metadata
                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                total_tokens = input_tokens + output_tokens

                cost = (total_tokens / 1000.0) * settings.CLOUD_COST_PER_1K_TOKENS

                return ExecutionResult(
                    task_id=task_id,
                    target_used=ExecutionTarget.CLOUD,
                    model_used=model,
                    output_text=output,
                    actual_latency_ms=elapsed_ms,
                    actual_cost_usd=cost,
                    tokens_used=total_tokens,
                    success=True,
                )

            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "cloud_worker.retry",
                        attempt=attempt + 1,
                        delay_sec=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
                else:
                    raise


async def main() -> None:
    import structlog
    structlog.configure(
        processors=[structlog.processors.JSONRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(20),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    worker = CloudWorker()
    await worker.start()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
