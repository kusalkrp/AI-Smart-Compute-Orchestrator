from __future__ import annotations

from models.enums import ExecutionTarget
from models.schemas import ResourceSnapshot


class LoadBalancer:
    """Balance load across same-type workers based on queue depths."""

    def select_target(
        self,
        preferred: ExecutionTarget,
        resource: ResourceSnapshot,
    ) -> ExecutionTarget:
        """Return the least-loaded available target of the same class."""
        # For now we have one worker per type; this is the hook for future scaling
        # where multiple GPU/CPU workers could exist with different queue depths
        return preferred

    def is_overloaded(self, target: ExecutionTarget, resource: ResourceSnapshot) -> bool:
        from config.settings import settings

        if target == ExecutionTarget.GPU:
            return (
                resource.gpu_percent > settings.GPU_OVERLOAD_PERCENT
                or not resource.gpu_available
            )
        elif target == ExecutionTarget.CPU:
            return resource.cpu_percent > settings.CPU_OVERLOAD_PERCENT
        elif target == ExecutionTarget.QUANTIZED:
            return resource.quantized_queue_depth > 20
        else:  # CLOUD
            return resource.cloud_queue_depth > 100
