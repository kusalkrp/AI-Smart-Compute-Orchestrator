from __future__ import annotations

import time
from typing import Any

import httpx

from config.settings import settings


class OllamaClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def generate(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any] | None = None,
        cpu_only: bool = False,
    ) -> dict[str, Any]:
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if cpu_only:
            payload.setdefault("options", {})["num_gpu"] = 0

        start = time.time()
        response = await client.post("/api/generate", json=payload)
        response.raise_for_status()
        elapsed_ms = int((time.time() - start) * 1000)

        data = response.json()
        data["_elapsed_ms"] = elapsed_ms
        return data

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        cpu_only: bool = False,
    ) -> dict[str, Any]:
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if cpu_only:
            payload.setdefault("options", {})["num_gpu"] = 0

        start = time.time()
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        elapsed_ms = int((time.time() - start) * 1000)

        data = response.json()
        data["_elapsed_ms"] = elapsed_ms
        return data

    async def list_models(self) -> list[str]:
        client = await self._get_client()
        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    async def is_available(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def pull_model(self, model: str) -> bool:
        client = await self._get_client()
        try:
            response = await client.post("/api/pull", json={"name": model, "stream": False})
            return response.status_code == 200
        except Exception:
            return False


# Singleton instance
ollama = OllamaClient()
