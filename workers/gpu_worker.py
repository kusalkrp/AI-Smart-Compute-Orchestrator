"""GPU Worker — runs Ollama inference with GPU acceleration."""
from __future__ import annotations

import uuid

from config.settings import settings
from infrastructure.ollama_client import ollama
from models.enums import ExecutionTarget
from models.schemas import ExecutionResult
from workers.base_worker import BaseWorker


class GPUWorker(BaseWorker):
    stream_key = settings.STREAM_TASKS_GPU
    group_name = "workers:gpu"
    target = ExecutionTarget.GPU
    consumer_name = "gpu-worker-1"

    async def execute(self, task_id: uuid.UUID, fields: dict) -> ExecutionResult:
        model = fields.get("model", settings.GPU_MODEL)
        input_text = fields.get("input", "")

        response = await ollama.generate(
            model=model,
            prompt=input_text,
            options={"temperature": 0.7, "num_predict": 512},
            cpu_only=False,
        )

        output = response.get("response", "")
        elapsed_ms = response.get("_elapsed_ms", 0)
        tokens = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)

        return ExecutionResult(
            task_id=task_id,
            target_used=ExecutionTarget.GPU,
            model_used=model,
            output_text=output,
            actual_latency_ms=elapsed_ms,
            actual_cost_usd=0.0,  # Local inference = no cost
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
    worker = GPUWorker()
    await worker.start()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
