"""CPU Worker — runs Ollama inference in CPU-only mode."""
from __future__ import annotations

import uuid

from config.settings import settings
from infrastructure.ollama_client import ollama
from models.enums import ExecutionTarget
from models.schemas import ExecutionResult
from workers.base_worker import BaseWorker


class CPUWorker(BaseWorker):
    stream_key = settings.STREAM_TASKS_CPU
    group_name = "workers:cpu"
    target = ExecutionTarget.CPU
    consumer_name = "cpu-worker-1"

    async def execute(self, task_id: uuid.UUID, fields: dict) -> ExecutionResult:
        model = fields.get("model", settings.CPU_MODEL)
        input_text = fields.get("input", "")

        response = await ollama.generate(
            model=model,
            prompt=input_text,
            options={"temperature": 0.7, "num_predict": 512},
            cpu_only=True,  # Force CPU inference
        )

        output = response.get("response", "")
        elapsed_ms = response.get("_elapsed_ms", 0)
        tokens = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)

        return ExecutionResult(
            task_id=task_id,
            target_used=ExecutionTarget.CPU,
            model_used=model,
            output_text=output,
            actual_latency_ms=elapsed_ms,
            actual_cost_usd=0.0,
            tokens_used=tokens,
            success=True,
        )


async def main() -> None:
    import structlog
    structlog.configure(
        processors=[structlog.processors.JSONRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(20),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    worker = CPUWorker()
    await worker.start()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
