"""Seed the database with sample routing policies and cost config."""
from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


async def main() -> None:
    from infrastructure.postgres_client import create_tables, AsyncSessionLocal
    from models.database import RoutingPolicy, ModelPerformance

    print("Creating database tables...")
    await create_tables()

    async with AsyncSessionLocal() as session:
        # Sample routing policies
        policies = [
            RoutingPolicy(
                name="urgent_cloud",
                description="Route urgent tasks with high complexity to cloud",
                conditions={"priority": "URGENT", "complexity_score_gt": 0.6},
                target="CLOUD",
                model_name="gemini-2.0-flash",
                priority_order=10,
                is_active=True,
            ),
            RoutingPolicy(
                name="batch_quantized",
                description="Route batch tasks to quantized model",
                conditions={"is_batch": True, "latency_sensitivity_lte": 0.4},
                target="QUANTIZED",
                model_name="phi3:mini",
                priority_order=20,
                is_active=True,
            ),
            RoutingPolicy(
                name="default_gpu",
                description="Default: use GPU if available",
                conditions={"gpu_available": True, "gpu_utilization_lte": 80.0},
                target="GPU",
                model_name="mistral:7b-instruct-q4_0",
                priority_order=100,
                is_active=True,
            ),
        ]

        for policy in policies:
            # Check if already exists
            from sqlalchemy import select
            result = await session.execute(
                select(RoutingPolicy).where(RoutingPolicy.name == policy.name)
            )
            if not result.scalar_one_or_none():
                session.add(policy)
                print(f"  Added policy: {policy.name}")
            else:
                print(f"  Policy already exists: {policy.name}")

        # Sample model performance baselines
        perfs = [
            ModelPerformance(model_name="mistral:7b-instruct-q4_0", target="GPU", avg_latency_ms=1500, avg_cost_usd=0.0, success_rate=0.98, sample_count=0),
            ModelPerformance(model_name="phi3:mini", target="CPU", avg_latency_ms=8000, avg_cost_usd=0.0, success_rate=0.95, sample_count=0),
            ModelPerformance(model_name="phi3:mini", target="QUANTIZED", avg_latency_ms=6000, avg_cost_usd=0.0, success_rate=0.94, sample_count=0),
            ModelPerformance(model_name="gemini-2.0-flash", target="CLOUD", avg_latency_ms=1800, avg_cost_usd=0.000125, success_rate=0.99, sample_count=0),
        ]

        for perf in perfs:
            result = await session.execute(
                select(ModelPerformance).where(
                    ModelPerformance.model_name == perf.model_name,
                    ModelPerformance.target == perf.target,
                )
            )
            if not result.scalar_one_or_none():
                session.add(perf)
                print(f"  Added performance baseline: {perf.target}:{perf.model_name}")

        await session.commit()

    print("\n✓ Database seeded successfully!")


if __name__ == "__main__":
    asyncio.run(main())
