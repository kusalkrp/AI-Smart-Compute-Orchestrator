from __future__ import annotations

import psutil


class CPUMonitor:
    def get_cpu_percent(self, interval: float = 0.1) -> float:
        return psutil.cpu_percent(interval=interval)

    def get_ram_percent(self) -> float:
        return psutil.virtual_memory().percent

    def get_ram_available_mb(self) -> float:
        return psutil.virtual_memory().available / (1024 * 1024)

    def snapshot(self) -> dict[str, float]:
        mem = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": mem.percent,
            "ram_available_mb": mem.available / (1024 * 1024),
        }
