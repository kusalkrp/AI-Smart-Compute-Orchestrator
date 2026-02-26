"""Locust load test: 100 concurrent task submissions."""
from __future__ import annotations

import random

from locust import HttpUser, between, task


TASK_TYPES = ["CHAT", "SUMMARIZATION", "CLASSIFICATION", "REASONING", "BATCH_SUMMARIZATION"]
PRIORITIES = ["LOW", "NORMAL", "HIGH", "URGENT"]

SAMPLE_PROMPTS = [
    "Summarize the key points from this quarterly report.",
    "Classify this customer review as positive, negative, or neutral.",
    "What are the main causes of climate change?",
    "Explain the difference between machine learning and deep learning.",
    "Generate a brief product description for a wireless keyboard.",
    "Analyze the sentiment of this text and provide reasoning.",
    "Translate the following concepts into simple terms.",
    "What is the best approach to handle this technical problem?",
]

API_KEY = "dev-key-change-in-production"


class OrchestratorUser(HttpUser):
    wait_time = between(0.5, 2.0)
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task(5)
    def submit_normal_task(self) -> None:
        payload = {
            "task_type": random.choice(TASK_TYPES[:4]),  # Exclude BATCH for normal tasks
            "input_text": random.choice(SAMPLE_PROMPTS) + " " + "x" * random.randint(50, 500),
            "priority": random.choice(["NORMAL", "HIGH"]),
        }
        with self.client.post(
            "/v1/tasks",
            json=payload,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 202:
                response.success()
                task_id = response.json().get("task_id")
                if task_id:
                    self._task_ids = getattr(self, "_task_ids", [])
                    self._task_ids.append(task_id)
                    if len(self._task_ids) > 20:
                        self._task_ids.pop(0)
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(2)
    def submit_urgent_task(self) -> None:
        with self.client.post(
            "/v1/tasks",
            json={
                "task_type": "CHAT",
                "input_text": "URGENT: " + random.choice(SAMPLE_PROMPTS),
                "priority": "URGENT",
            },
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 202:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(2)
    def submit_batch_task(self) -> None:
        with self.client.post(
            "/v1/tasks",
            json={
                "task_type": "BATCH_SUMMARIZATION",
                "input_text": "Batch process: " + "document content " * 50,
                "priority": "LOW",
                "is_batch": True,
            },
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 202:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(3)
    def poll_task_status(self) -> None:
        task_ids = getattr(self, "_task_ids", [])
        if not task_ids:
            return
        task_id = random.choice(task_ids)
        with self.client.get(
            f"/v1/tasks/{task_id}",
            headers=self.headers,
            catch_response=True,
            name="/v1/tasks/[task_id]",
        ) as response:
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)
    def get_routing_stats(self) -> None:
        with self.client.get(
            "/v1/routing/stats",
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)
    def health_check(self) -> None:
        with self.client.get("/v1/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
