from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

_pynvml_available = False
_initialized = False


def _ensure_init() -> bool:
    global _pynvml_available, _initialized
    if _initialized:
        return _pynvml_available
    try:
        import pynvml
        pynvml.nvmlInit()
        _pynvml_available = True
        logger.info("gpu_monitor.pynvml_initialized")
    except Exception as e:
        logger.info("gpu_monitor.no_gpu", reason=str(e))
        _pynvml_available = False
    _initialized = True
    return _pynvml_available


class GPUMonitor:
    def __init__(self) -> None:
        self.available = _ensure_init()

    def snapshot(self) -> dict[str, float]:
        if not self.available:
            return {
                "gpu_available": False,
                "gpu_percent": 0.0,
                "gpu_vram_used_mb": 0.0,
                "gpu_vram_total_mb": 0.0,
                "gpu_vram_free_mb": 0.0,
            }

        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            return {
                "gpu_available": True,
                "gpu_percent": float(util.gpu),
                "gpu_vram_used_mb": mem_info.used / (1024 * 1024),
                "gpu_vram_total_mb": mem_info.total / (1024 * 1024),
                "gpu_vram_free_mb": mem_info.free / (1024 * 1024),
            }
        except Exception as e:
            logger.warning("gpu_monitor.read_failed", error=str(e))
            return {
                "gpu_available": False,
                "gpu_percent": 0.0,
                "gpu_vram_used_mb": 0.0,
                "gpu_vram_total_mb": 0.0,
                "gpu_vram_free_mb": 0.0,
            }

    def is_available(self) -> bool:
        return self.available

    def get_vram_free_mb(self) -> float:
        data = self.snapshot()
        return data.get("gpu_vram_free_mb", 0.0)
