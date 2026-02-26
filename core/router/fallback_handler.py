from __future__ import annotations

import structlog

from core.intelligence.rule_engine import RuleEngine
from models.enums import ExecutionTarget

logger = structlog.get_logger(__name__)

_FALLBACK_CHAIN = {
    ExecutionTarget.GPU: ExecutionTarget.CPU,
    ExecutionTarget.CPU: ExecutionTarget.QUANTIZED,
    ExecutionTarget.QUANTIZED: ExecutionTarget.CLOUD,
    ExecutionTarget.CLOUD: ExecutionTarget.CPU,
}


class FallbackHandler:
    """Manage target unavailability and fallback chain traversal."""

    def get_fallback(self, failed_target: ExecutionTarget) -> ExecutionTarget:
        fallback = _FALLBACK_CHAIN.get(failed_target, ExecutionTarget.CLOUD)
        logger.info("fallback_handler.target_switched", from_=failed_target.value, to=fallback.value)
        return fallback

    def get_chain(self, start_target: ExecutionTarget, max_depth: int = 3) -> list[ExecutionTarget]:
        chain = [start_target]
        current = start_target
        for _ in range(max_depth):
            next_target = _FALLBACK_CHAIN.get(current, ExecutionTarget.CLOUD)
            if next_target in chain:
                break
            chain.append(next_target)
            current = next_target
        return chain
