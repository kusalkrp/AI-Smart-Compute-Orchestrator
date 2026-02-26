"""Benchmark routing: compare Stage 1 vs Stage 2 vs Stage 3 decisions."""
from __future__ import annotations

import asyncio
import sys
import os
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


SAMPLE_TASKS = [
    {"task_type": "CHAT", "input_text": "Hello, how are you?", "priority": "NORMAL"},
    {"task_type": "SUMMARIZATION", "input_text": "Summarize this long document: " + "A" * 2000, "priority": "LOW"},
    {"task_type": "REASONING", "input_text": "Solve this complex logic puzzle: " + "B" * 1000, "priority": "HIGH"},
    {"task_type": "BATCH_SUMMARIZATION", "input_text": "Process batch: " + "C" * 500, "priority": "LOW", "is_batch": True},
    {"task_type": "CLASSIFICATION", "input_text": "Classify urgent feedback.", "priority": "URGENT"},
    {"task_type": "EMBEDDING", "input_text": "Generate embedding for: " + "D" * 300, "priority": "LOW"},
]


async def benchmark_stage(stage: str) -> list[dict]:
    """Run all sample tasks through a given routing stage."""
    import os
    os.environ["ROUTING_STAGE"] = stage

    # Reload settings after env change
    import importlib
    import config.settings
    importlib.reload(config.settings)
    from config.settings import settings
    settings.ROUTING_STAGE = stage

    from core.intelligence.decision_engine import DecisionEngine
    from core.monitor.resource_monitor import resource_monitor
    from models.schemas import TaskRequest
    from models.enums import TaskType, Priority

    engine = DecisionEngine()
    snapshot = await resource_monitor.get_current()

    results = []
    for task_dict in SAMPLE_TASKS:
        task_id = uuid.uuid4()
        request = TaskRequest(
            task_type=TaskType(task_dict["task_type"]),
            input_text=task_dict["input_text"],
            priority=Priority(task_dict.get("priority", "NORMAL")),
            is_batch=task_dict.get("is_batch", False),
        )

        start = time.time()
        decision = await engine.decide(task_id, request, snapshot)
        elapsed_ms = int((time.time() - start) * 1000)

        results.append({
            "task_type": task_dict["task_type"],
            "priority": task_dict.get("priority", "NORMAL"),
            "target": decision.target.value,
            "model": decision.model_name,
            "estimated_cost": decision.estimated_cost_usd,
            "estimated_latency": decision.estimated_latency_ms,
            "confidence": decision.confidence,
            "routing_time_ms": elapsed_ms,
            "stage": stage,
        })

    return results


async def main() -> None:
    print("=" * 70)
    print("AI Smart Compute Orchestrator — Routing Benchmark")
    print("=" * 70)

    stages = ["rule", "scored"]

    all_results = {}
    for stage in stages:
        print(f"\nBenchmarking Stage: {stage.upper()}...")
        results = await benchmark_stage(stage)
        all_results[stage] = results

        for r in results:
            print(
                f"  {r['task_type']:20} {r['priority']:8} → "
                f"{r['target']:10} {r['model']:30} "
                f"cost=${r['estimated_cost']:.6f} "
                f"lat={r['estimated_latency']}ms "
                f"conf={r['confidence']:.2f} "
                f"route={r['routing_time_ms']}ms"
            )

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON: Rule vs Scored")
    print("=" * 70)
    rule_results = {r["task_type"]: r for r in all_results.get("rule", [])}
    scored_results = {r["task_type"]: r for r in all_results.get("scored", [])}

    for task_type in rule_results:
        rule = rule_results[task_type]
        scored = scored_results.get(task_type, {})
        same = "✓ SAME" if rule["target"] == scored.get("target") else "✗ DIFF"
        print(
            f"  {task_type:20} Rule:{rule['target']:10} "
            f"Scored:{scored.get('target', 'N/A'):10} {same}"
        )

    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
