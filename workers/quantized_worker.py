"""Quantized Worker — runs llama-cpp-python for CPU-based quantized inference."""
from __future__ import annotations

import uuid
from pathlib import Path

import structlog

from config.settings import settings
from models.enums import ExecutionTarget
from models.schemas import ExecutionResult
from workers.base_worker import BaseWorker

logger = structlog.get_logger(__name__)


class QuantizedWorker(BaseWorker):
    stream_key = settings.STREAM_TASKS_QUANTIZED
    group_name = "workers:quantized"
    target = ExecutionTarget.QUANTIZED
    consumer_name = "quantized-worker-1"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._model_path = Path(settings.QUANTIZED_MODEL_PATH)
        self._llama_available = False
        self._try_load_model()

    def _try_load_model(self) -> None:
        if not self._model_path.exists():
            logger.warning(
                "quantized_worker.model_not_found",
                path=str(self._model_path),
            )
            return
        try:
            from llama_cpp import Llama
            self._model = Llama(
                model_path=str(self._model_path),
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=0,  # CPU only
                verbose=False,
            )
            self._llama_available = True
            logger.info("quantized_worker.model_loaded", path=str(self._model_path))
        except ImportError:
            logger.warning("quantized_worker.llama_cpp_not_installed")
        except Exception as e:
            logger.warning("quantized_worker.load_failed", error=str(e))

    async def execute(self, task_id: uuid.UUID, fields: dict) -> ExecutionResult:
        input_text = fields.get("input", "")
        model_name = fields.get("model", settings.CPU_MODEL)

        if not self._llama_available or self._model is None:
            # Fallback to Ollama CPU if llama.cpp not available
            from infrastructure.ollama_client import ollama
            response = await ollama.generate(
                model=model_name,
                prompt=input_text,
                cpu_only=True,
            )
            output = response.get("response", "")
            elapsed_ms = response.get("_elapsed_ms", 0)
            tokens = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
        else:
            import asyncio
            import time

            start = time.time()
            # Run synchronous llama.cpp inference in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._model(
                    input_text,
                    max_tokens=512,
                    temperature=0.7,
                    stop=["</s>", "[INST]"],
                ),
            )
            elapsed_ms = int((time.time() - start) * 1000)

            output = response["choices"][0]["text"]
            usage = response.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            model_name = self._model_path.stem

        return ExecutionResult(
            task_id=task_id,
            target_used=ExecutionTarget.QUANTIZED,
            model_used=model_name,
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
    worker = QuantizedWorker()
    await worker.start()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
