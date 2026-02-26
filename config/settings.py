from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ─── Server ───────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    # Comma-separated string — pydantic-settings v2 cannot JSON-parse plain strings
    # into list[str], so we keep it as str and expose a parsed property below.
    API_KEYS: str = "dev-key-change-in-production"
    DEBUG: bool = False
    VERSION: str = "0.1.0"

    # ─── Infrastructure ───────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    DATABASE_URL: str = "postgresql+asyncpg://orchestrator:orchestrator@localhost:5432/orchestrator"

    # ─── Ollama ───────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GPU_MODEL: str = "mistral:7b-instruct-q4_0"
    CPU_MODEL: str = "phi3:mini"
    QUANTIZED_MODEL_PATH: str = "./models/phi-3-mini-q4.gguf"

    # ─── Cloud LLM (Google Gemini) ────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    CLOUD_MODEL: str = "gemini-2.0-flash"

    # ─── Routing ──────────────────────────────────────────────────────────────
    ROUTING_STAGE: Literal["rule", "scored", "ml"] = "rule"
    ML_MODEL_PATH: str = "./models/routing_model.pkl"
    ML_CONFIDENCE_THRESHOLD: float = 0.6
    ROUTING_POLICY_PATH: str = "./config/routing_policy.yaml"
    COST_CONFIG_PATH: str = "./config/cost_config.yaml"

    # ─── Resource Thresholds ──────────────────────────────────────────────────
    GPU_OVERLOAD_PERCENT: float = 85.0
    CPU_OVERLOAD_PERCENT: float = 90.0
    GPU_VRAM_BUFFER_MB: float = 512.0

    # ─── Costs ────────────────────────────────────────────────────────────────
    # gemini-2.0-flash: ~$0.10/1M input + $0.40/1M output → avg $0.00025/1k tokens
    CLOUD_COST_PER_1K_TOKENS: float = 0.00025
    LOCAL_COMPUTE_COST_PER_HOUR: float = 0.0

    # ─── Celery ───────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ─── Monitoring ───────────────────────────────────────────────────────────
    RESOURCE_MONITOR_INTERVAL_SEC: int = 2
    RESOURCE_SNAPSHOT_DB_INTERVAL_SEC: int = 30

    # ─── Dashboard ────────────────────────────────────────────────────────────
    DASHBOARD_API_BASE_URL: str = "http://localhost:8000"

    # ─── Redis Stream Keys ────────────────────────────────────────────────────
    STREAM_TASKS_CPU: str = "stream:tasks:cpu"
    STREAM_TASKS_GPU: str = "stream:tasks:gpu"
    STREAM_TASKS_QUANTIZED: str = "stream:tasks:quantized"
    STREAM_TASKS_CLOUD: str = "stream:tasks:cloud"
    STREAM_RESULTS: str = "stream:results"
    STREAM_RESOURCE_EVENTS: str = "stream:resource:events"
    RESOURCE_CURRENT_KEY: str = "resource:current"
    MODEL_PERFORMANCE_KEY: str = "model:performance"

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.API_KEYS.split(",") if k.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
