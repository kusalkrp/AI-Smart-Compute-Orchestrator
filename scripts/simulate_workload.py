"""Demo script: simulate varied task types and urgency scenarios."""
from __future__ import annotations

import asyncio
import argparse
import sys
import os
import time
from typing import Optional

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BASE_URL = "http://localhost:8000"
API_KEY = "kusal1234567890"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


async def submit_task(client: httpx.AsyncClient, payload: dict) -> dict:
    response = await client.post(f"{BASE_URL}/v1/tasks", json=payload, headers=HEADERS)
    response.raise_for_status()
    return response.json()


async def wait_for_result(client: httpx.AsyncClient, task_id: str, timeout: int = 120) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        response = await client.get(f"{BASE_URL}/v1/tasks/{task_id}", headers=HEADERS)
        data = response.json()
        status = data.get("status")
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            return data
        await asyncio.sleep(2)
    return {"status": "TIMEOUT", "task_id": task_id}


def print_result(task_data: dict, label: str = "") -> None:
    routing = task_data.get("routing_decision") or {}
    print(f"\n{'-' * 60}")
    if label:
        print(f"  Scenario: {label}")
    print(f"  Task ID:   {task_data.get('task_id', 'N/A')[:8]}...")
    print(f"  Status:    {task_data.get('status', 'N/A')}")
    print(f"  Target:    {routing.get('target', 'N/A')}")
    print(f"  Model:     {routing.get('model_name', 'N/A')}")
    print(f"  Stage:     {routing.get('decision_stage', 'N/A')}")
    print(f"  Est. Cost: ${routing.get('estimated_cost_usd', 0):.6f}")
    print(f"  Est. Lat:  {routing.get('estimated_latency_ms', 0)}ms")
    print(f"  Actual Lat:{task_data.get('actual_latency_ms', 'N/A')}ms")
    reasoning = routing.get('reasoning', 'N/A') or 'N/A'
    print(f"  Reasoning: {reasoning[:80]}...")


async def scenario_urgency(client: httpx.AsyncClient) -> None:
    """Scenario 1: Same task, different priority -> different routing."""
    print("\n" + "=" * 60)
    print("SCENARIO 1: Urgency-Based Routing")
    print("Same task, different priority -> different execution path")
    print("=" * 60)

    base_prompt = "Summarize the following article about machine learning advances in 2025: " + "A" * 500

    # LOW priority -> should route to local
    result_low = await submit_task(client, {
        "task_type": "SUMMARIZATION",
        "input_text": base_prompt,
        "priority": "LOW",
    })
    print(f"\n-> LOW priority submitted: {result_low['task_id'][:8]}...")

    # URGENT priority -> should route to cloud
    result_urgent = await submit_task(client, {
        "task_type": "SUMMARIZATION",
        "input_text": base_prompt,
        "priority": "URGENT",
    })
    print(f"-> URGENT priority submitted: {result_urgent['task_id'][:8]}...")

    # Wait for routing decisions (don't wait for full execution in demo)
    await asyncio.sleep(3)

    low_status = await client.get(f"{BASE_URL}/v1/tasks/{result_low['task_id']}", headers=HEADERS)
    urgent_status = await client.get(f"{BASE_URL}/v1/tasks/{result_urgent['task_id']}", headers=HEADERS)

    print_result(low_status.json(), "LOW Priority")
    print_result(urgent_status.json(), "URGENT Priority")


async def scenario_cost(client: httpx.AsyncClient) -> None:
    """Scenario 2: Cost vs latency trade-off."""
    print("\n" + "=" * 60)
    print("SCENARIO 2: Cost vs Latency Trade-off")
    print("Batch background tasks -> quantized; HIGH priority -> GPU")
    print("=" * 60)

    tasks = []

    # Submit 5 LOW priority batch tasks
    for i in range(5):
        result = await submit_task(client, {
            "task_type": "BATCH_SUMMARIZATION",
            "input_text": f"Classify document {i}: " + "X" * 300,
            "priority": "LOW",
            "is_batch": True,
        })
        tasks.append(("LOW/BATCH", result["task_id"]))
        print(f"  -> Batch task {i+1} submitted: {result['task_id'][:8]}...")

    # Submit HIGH priority single task
    result = await submit_task(client, {
        "task_type": "CLASSIFICATION",
        "input_text": "Urgent: Classify this critical customer complaint immediately.",
        "priority": "HIGH",
    })
    tasks.append(("HIGH/SINGLE", result["task_id"]))
    print(f"  -> HIGH priority task submitted: {result['task_id'][:8]}...")

    await asyncio.sleep(3)

    print("\nRouting decisions:")
    for label, task_id in tasks:
        status = await client.get(f"{BASE_URL}/v1/tasks/{task_id}", headers=HEADERS)
        data = status.json()
        routing = data.get("routing_decision") or {}
        print(f"  [{label}] -> Target: {routing.get('target', 'PENDING')}, Stage: {routing.get('decision_stage', '?')}")


async def scenario_resource(client: httpx.AsyncClient) -> None:
    """Scenario 3: Resource pressure adaptation."""
    print("\n" + "=" * 60)
    print("SCENARIO 3: Resource Pressure Adaptation")
    print("Submitting burst of tasks -> watch routing adapt")
    print("=" * 60)

    tasks = []
    for i in range(10):
        result = await submit_task(client, {
            "task_type": "CHAT",
            "input_text": f"Task {i}: Explain quantum computing in simple terms.",
            "priority": "NORMAL",
        })
        tasks.append(result["task_id"])
        print(f"  -> Task {i+1} submitted: {result['task_id'][:8]}...")
        await asyncio.sleep(0.2)

    await asyncio.sleep(5)

    target_counts: dict[str, int] = {}
    for task_id in tasks:
        status = await client.get(f"{BASE_URL}/v1/tasks/{task_id}", headers=HEADERS)
        data = status.json()
        routing = data.get("routing_decision") or {}
        target = routing.get("target", "UNKNOWN")
        target_counts[target] = target_counts.get(target, 0) + 1

    print("\nRouting distribution under load:")
    for target, count in sorted(target_counts.items()):
        bar = "#" * count
        print(f"  {target:10} {bar} ({count})")


async def scenario_reasoning(client: httpx.AsyncClient) -> None:
    """Scenario 4: Reasoning task routing."""
    print("\n" + "=" * 60)
    print("SCENARIO 4: Reasoning Task Routing")
    print("Complex reasoning -> cloud; Simple reasoning -> GPU")
    print("=" * 60)

    # Complex reasoning
    result = await submit_task(client, {
        "task_type": "REASONING",
        "input_text": "Solve this multi-step problem: If a train travels at 120km/h for 2.5 hours, then slows to 80km/h for 1.5 hours, calculate total distance, average speed, and fuel efficiency assuming 0.8L/km at high speed and 0.5L/km at low speed. Show all working.",
        "priority": "HIGH",
        "metadata": {"chain_of_thought": True},
    })
    print(f"  -> Complex reasoning submitted: {result['task_id'][:8]}...")

    # Simple chat
    result2 = await submit_task(client, {
        "task_type": "CHAT",
        "input_text": "What is 2+2?",
        "priority": "NORMAL",
    })
    print(f"  -> Simple chat submitted: {result2['task_id'][:8]}...")

    await asyncio.sleep(3)

    for task_id, label in [(result["task_id"], "Complex Reasoning"), (result2["task_id"], "Simple Chat")]:
        status = await client.get(f"{BASE_URL}/v1/tasks/{task_id}", headers=HEADERS)
        print_result(status.json(), label)


async def run_all(client: httpx.AsyncClient) -> None:
    await scenario_urgency(client)
    await scenario_cost(client)
    await scenario_resource(client)
    await scenario_reasoning(client)

    print("\n" + "=" * 60)
    print("All scenarios complete!")
    print(f"View results at: http://localhost:8501")
    print(f"View API docs at: http://localhost:8000/docs")
    stats_resp = await client.get(f"{BASE_URL}/v1/routing/stats", headers=HEADERS)
    if stats_resp.status_code == 200:
        stats = stats_resp.json()
        print(f"\nRouting Stats Summary:")
        print(f"  Total tasks routed: {stats.get('total_tasks', 0)}")
        print(f"  Cost saved: ${stats.get('cost_saved_usd', 0):.6f}")
        print(f"  By target: {stats.get('by_target', {})}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Orchestrator Demo Scenarios")
    parser.add_argument(
        "--scenario",
        choices=["all", "urgency", "cost", "resource", "reasoning"],
        default="all",
        help="Which scenario to run",
    )
    parser.add_argument("--base-url", default=BASE_URL, help="API base URL")
    args = parser.parse_args()

    async def _run():
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Verify API is up
            try:
                health = await client.get(f"{args.base_url}/v1/health", headers=HEADERS)
                health.raise_for_status()
                print(f"[OK] API is running at {args.base_url}")
            except Exception as e:
                print(f"[ERR] API not available at {args.base_url}: {e}")
                print("  Start the API with: make dev-api")
                return

            if args.scenario == "all":
                await run_all(client)
            elif args.scenario == "urgency":
                await scenario_urgency(client)
            elif args.scenario == "cost":
                await scenario_cost(client)
            elif args.scenario == "resource":
                await scenario_resource(client)
            elif args.scenario == "reasoning":
                await scenario_reasoning(client)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
