# AI Smart Compute Orchestrator

An intelligent execution layer that dynamically routes AI inference workloads across local CPU, local GPU, quantized models, and cloud APIs — maximising cost efficiency and performance by analysing each task's complexity, urgency, and available resources before routing.

---

## What It Does

Instead of blindly sending every AI request to a single backend (e.g. always cloud, always local), the orchestrator evaluates each task and picks the optimal execution target:

- **LOW priority / batch jobs** → quantized model on CPU (near-zero cost)
- **URGENT tasks** → cloud API (lowest latency)
- **Complex reasoning** → GPU or cloud (highest capability)
- **GPU overloaded** → automatically falls back to CPU or cloud

The result is significant cost savings (avoid cloud for tasks that don't need it) with no manual routing logic in application code.

---

## Architecture

```
 Client Application
        |
        | POST /v1/tasks
        v
 +------------------------------------------------------+
 |                  FastAPI (port 8000)                  |
 |  Auth Middleware -> Rate Limiter -> Request Logger     |
 +------------------------------------------------------+
        |
        | Background task
        v
 +------------------------------------------------------+
 |               Workload Analyzer                       |
 |  TaskClassifier -> TokenEstimator -> ComplexityScorer |
 |  Output: ExecutionProfile (complexity, sensitivity)   |
 +------------------------------------------------------+
        |
        v
 +------------------------------------------------------+
 |              Resource Monitor                         |
 |  CPUMonitor (psutil) + GPUMonitor (pynvml)            |
 |  + QueueMonitor (Redis XLEN)                          |
 |  Output: ResourceSnapshot (cached in Redis, 10s TTL)  |
 +------------------------------------------------------+
        |
        v
 +------------------------------------------------------+
 |             Decision Engine (3 Stages)                |
 |                                                       |
 |  Stage 1: Rule Engine  (YAML rules, config-driven)   |
 |  Stage 2: Scoring      (weighted multi-factor)       |
 |  Stage 3: ML Model     (XGBoost classifier)          |
 |                                                       |
 |  Output: RoutingDecision (target, model, reasoning)  |
 +------------------------------------------------------+
        |
        | Redis XADD -> stream:tasks:{target}
        v
 +--------------+--------------+---------------+--------+
 | GPU Worker   | CPU Worker   | Quant Worker  | Cloud  |
 | Ollama/GPU   | Ollama/CPU   | llama.cpp Q4  | Gemini |
 | mistral:7b   | phi3:mini    | phi-3-mini    | 1.5-f  |
 +--------------+--------------+---------------+--------+
        |
        | Redis XADD -> stream:results
        v
 +------------------------------------------------------+
 |           Feedback Collector                          |
 |  Updates ModelPerformance rolling averages            |
 |  Feeds into Stage 2 scoring & Stage 3 retraining     |
 +------------------------------------------------------+
        |
        v
 PostgreSQL (tasks, routing_decisions, execution_logs)
 Streamlit Dashboard (port 8501)
```

---

## Features

- **Three-stage routing intelligence**: rule-based -> weighted scoring -> ML classifier (XGBoost)
- **Four execution targets**: GPU (Ollama), CPU (Ollama), Quantized (llama.cpp), Cloud (Gemini)
- **Real-time resource awareness**: CPU%, GPU%, VRAM, queue depths refreshed every 2 seconds
- **Automatic fallback chain**: GPU -> CPU -> Quantized -> Cloud
- **Cost tracking**: estimates and actuals per task, cumulative savings vs. all-cloud baseline
- **Live routing rules**: edit `config/routing_policy.yaml` without restart (volume-mounted)
- **Prometheus metrics** at `/v1/metrics`
- **Streamlit dashboard** with routing distribution, cost savings, resource heatmaps
- **API key auth** + sliding-window rate limiting
- **Structured JSON logging** (structlog)
- **Full async**: FastAPI + asyncpg + redis-py asyncio

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16-24 GB |
| GPU VRAM | optional | 6 GB+ (NVIDIA/CUDA) |
| Disk | 5 GB | 20 GB (model files) |

GPU inference requires an NVIDIA GPU with CUDA. The system works without a GPU — tasks fall back to CPU/cloud automatically.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose v2)
- [Ollama](https://ollama.com/) installed and running on the host
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier available)

Pull the inference models before starting:

```bash
ollama pull mistral:7b-instruct-q4_0   # GPU worker (~4 GB)
ollama pull phi3:mini                   # CPU worker (~2 GB)
```

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/your-username/ai-smart-compute-orchestrator.git
cd ai-smart-compute-orchestrator

cp .env.example .env
```

Edit `.env` and set at minimum:

```env
API_KEYS=your-secret-key-here
GEMINI_API_KEY=your-google-ai-studio-key
```

### 2. Start the stack

```bash
docker compose up -d
```

This starts:
- PostgreSQL (port 5432)
- Redis (port 6379)
- FastAPI API (port 8000) — auto-creates tables and seeds routing policies on first boot
- GPU worker, CPU worker, Cloud worker
- Streamlit dashboard (port 8501)

### 3. Verify

```bash
curl http://localhost:8000/v1/health
curl http://localhost:8000/v1/health/ready
```

### 4. Run the demo

```bash
python scripts/simulate_workload.py --scenario all
```

### 5. Open the dashboard

```
http://localhost:8501
```

---

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `API_KEYS` | `dev-key-change-in-production` | Comma-separated valid API keys |
| `GEMINI_API_KEY` | — | Google AI Studio API key (required for cloud routing) |
| `CLOUD_MODEL` | `gemini-1.5-flash` | Gemini model for cloud tasks |
| `GPU_MODEL` | `mistral:7b-instruct-q4_0` | Ollama model for GPU worker |
| `CPU_MODEL` | `phi3:mini` | Ollama model for CPU worker |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `ROUTING_STAGE` | `rule` | Active routing stage: `rule` / `scored` / `ml` |
| `GPU_OVERLOAD_PERCENT` | `85.0` | GPU utilization threshold before fallback |
| `CPU_OVERLOAD_PERCENT` | `90.0` | CPU utilization threshold before cloud escalation |
| `RESOURCE_MONITOR_INTERVAL_SEC` | `2` | How often to refresh resource snapshots (seconds) |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |

### Routing Rules (`config/routing_policy.yaml`)

Rules are evaluated in `priority_order` (lowest first). The first matching rule wins.

```yaml
rules:
  - name: urgent_complex_cloud
    description: "Urgent + complex tasks go to cloud for fastest response"
    priority_order: 10
    conditions:
      priority: URGENT
      complexity_score_gt: 0.6
    target: CLOUD
    model: gemini-1.5-flash

  - name: urgent_simple_gpu
    description: "Urgent + simple tasks use GPU for speed without cloud cost"
    priority_order: 15
    conditions:
      priority: URGENT
      complexity_score_lte: 0.6
      gpu_available: true
      gpu_utilization_lte: 85.0
    target: GPU
    model: mistral:7b-instruct-q4_0

  - name: default_cpu
    description: "Default: route to CPU for unknown/simple tasks"
    priority_order: 999
    conditions: {}
    target: CPU
    model: phi3:mini
```

**Available condition operators:**

| Suffix | Meaning | Example |
|---|---|---|
| (none) | Exact match | `priority: URGENT`, `gpu_available: true` |
| `_gt` | Greater than | `complexity_score_gt: 0.6` |
| `_gte` | Greater than or equal | `priority_gte: HIGH` |
| `_lte` | Less than or equal | `gpu_utilization_lte: 85.0` |
| `_lt` | Less than | `latency_sensitivity_lt: 0.3` |

**Available condition fields:**
`priority`, `complexity_score`, `latency_sensitivity`, `cost_sensitivity`, `estimated_tokens`, `is_batch`, `requires_reasoning`, `gpu_available`, `gpu_utilization`, `cpu_utilization`, `ram_percent`, `gpu_vram_used_mb`

> Routing policy changes take effect immediately — the file is volume-mounted and reloaded per request.

### Cost Config (`config/cost_config.yaml`)

Defines token costs and default latency estimates per target and model. Used by the cost calculator and scoring engine.

---

## API Reference

All endpoints except `/v1/health` require the `X-API-Key` header.

### Submit a task

```
POST /v1/tasks
Content-Type: application/json
X-API-Key: your-key
```

**Request body:**

```json
{
  "task_type": "REASONING",
  "input_text": "Explain quantum entanglement to a 10-year-old.",
  "priority": "HIGH",
  "is_batch": false,
  "max_cost_usd": 0.01,
  "max_latency_ms": 5000,
  "metadata": {},
  "callback_url": null
}
```

| Field | Type | Values |
|---|---|---|
| `task_type` | string | `CHAT`, `SUMMARIZATION`, `CLASSIFICATION`, `EMBEDDING`, `REASONING`, `BATCH_SUMMARIZATION` |
| `priority` | string | `LOW`, `NORMAL`, `HIGH`, `URGENT` |
| `is_batch` | bool | Marks as background batch job (cost-optimised routing) |
| `max_cost_usd` | float | Optional budget cap |
| `max_latency_ms` | int | Optional latency SLA |

**Response (202 Accepted):**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "created_at": "2025-01-15T10:23:45Z"
}
```

---

### Get task status + routing decision

```
GET /v1/tasks/{task_id}
X-API-Key: your-key
```

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "COMPLETED",
  "routing_decision": {
    "target": "CLOUD",
    "model_name": "gemini-1.5-flash",
    "estimated_cost_usd": 0.000029,
    "estimated_latency_ms": 2000,
    "confidence": 0.95,
    "reasoning": "Rule 'urgent_complex_cloud' matched: target=CLOUD, priority=HIGH, complexity=0.85",
    "fallback_target": "CPU",
    "decision_stage": "rule_based"
  },
  "result": "Imagine two coins that are magically linked...",
  "actual_cost_usd": 0.000025,
  "actual_latency_ms": 1843,
  "created_at": "2025-01-15T10:23:45Z",
  "completed_at": "2025-01-15T10:23:47Z"
}
```

**Task statuses:**

| Status | Meaning |
|---|---|
| `PENDING` | Accepted, not yet analysed |
| `ROUTING` | Being analysed and a target selected |
| `EXECUTING` | Running on the target worker |
| `COMPLETED` | Finished — result available |
| `FAILED` | Execution error (see worker logs) |
| `CANCELLED` | Cancelled by client |

---

### Cancel a task

```
POST /v1/tasks/{task_id}/cancel
```

---

### Routing statistics

```
GET /v1/routing/stats?hours=24
```

```json
{
  "total_tasks": 250,
  "by_target": { "GPU": 85, "CPU": 95, "QUANTIZED": 40, "CLOUD": 30 },
  "total_cost_usd": 0.00875,
  "estimated_cloud_cost_usd": 0.0625,
  "cost_saved_usd": 0.05375,
  "avg_latency_ms": 2145,
  "period_hours": 24
}
```

---

### Routing decision log

```
GET /v1/routing/decision-log?page=1&page_size=50
```

---

### Routing policies

```
GET  /v1/routing/policy          # List all active policies
POST /v1/routing/policy          # Create or update a policy
```

---

### Prometheus metrics

```
GET /v1/metrics
```

Returns Prometheus text format. Metrics include task counts by status and target, average latency by target, cost savings, and live CPU/GPU/RAM utilization.

---

### Health endpoints

```
GET /v1/health        # Liveness — always 200 if process is alive
GET /v1/health/ready  # Readiness — checks Redis + PostgreSQL
```

---

## How Routing Works

### Task Analysis Pipeline

Every submitted task goes through a three-step analysis before routing:

1. **TaskClassifier** — maps `task_type` + `priority` + `is_batch` to `latency_sensitivity` (0-1) and `cost_sensitivity` (0-1)
2. **TokenEstimator** — estimates token count from input length
3. **ComplexityScorer** — scores complexity (0-1) based on task type, token count, and reasoning flags

This produces an `ExecutionProfile` that feeds into the decision engine alongside a live `ResourceSnapshot`.

---

### Stage 1: Rule-Based Routing

YAML rules in `config/routing_policy.yaml` are evaluated in ascending `priority_order`. The first rule whose conditions all match determines the target. A `default_cpu` catch-all at order 999 always fires if nothing else matches.

This stage is **deterministic and auditable** — every routing decision includes the exact rule name and the values that triggered it in the `reasoning` field.

Activate with: `ROUTING_STAGE=rule`

---

### Stage 2: Weighted Scoring

All four targets are scored simultaneously:

```
score = 0.3 * cost_normalised
      + 0.3 * availability_penalty
      + 0.2 * (1 - success_rate)
      + 0.2 * latency_penalty
```

Lower score wins. Weights are updated weekly by the routing optimizer from execution history. This stage adapts to observed performance without requiring explicit rule authoring.

Activate with: `ROUTING_STAGE=scored`

---

### Stage 3: ML Classifier

An XGBoost multiclass classifier trained on 30 days of execution logs predicts the optimal target from 14 features: complexity, token count, latency/cost sensitivity, batch flag, CPU%, GPU%, VRAM%, GPU availability, and all four queue depths.

If prediction confidence is below the threshold (default 0.6), Stage 2 scoring is used as fallback.

Train the model manually:

```bash
make train-model
# or
python -m core.learning.model_trainer
```

Activate with: `ROUTING_STAGE=ml`

---

### Fallback Chain

If a worker fails or returns an error, the system automatically reroutes to the next target:

```
GPU -> CPU -> QUANTIZED -> CLOUD -> CPU
```

Fallback is transparent to the client — the task completes successfully on the fallback target.

---

## Demo Scenarios

```bash
# All scenarios end-to-end
python scripts/simulate_workload.py --scenario all

# Individual scenarios
python scripts/simulate_workload.py --scenario urgency    # Same task, different priority
python scripts/simulate_workload.py --scenario cost       # Batch vs interactive routing
python scripts/simulate_workload.py --scenario resource   # Burst load distribution
python scripts/simulate_workload.py --scenario reasoning  # Complexity-based routing
```

### Scenario 1: Urgency-Based Routing

The same summarisation task submitted twice with different priorities:

```
LOW priority   -> CPU  (phi3:mini,        ~$0.000001, 8000ms estimated)
URGENT priority -> CLOUD (gemini-1.5-flash, ~$0.000025, 2000ms estimated)
```

### Scenario 2: Cost vs Latency

- 5x batch classification tasks (`LOW` + `is_batch=true`) -> Quantized model (background, near-zero cost)
- 1x urgent classification (`HIGH` priority) -> GPU (fast, free)

### Scenario 3: Resource Pressure

10 concurrent CHAT tasks submitted with 200ms gaps. Routing distribution shows how the system spreads load as queue depths grow.

### Scenario 4: Reasoning Routing

- Complex multi-step reasoning (`HIGH`, complexity 0.85) -> Cloud (`reasoning_cloud` rule)
- Simple chat "What is 2+2?" (complexity 0.30) -> CPU (`default_cpu` rule)

---

## Dashboard

Open `http://localhost:8501` after starting the stack.

| Page | Content |
|---|---|
| **Overview** | Live task counts, cost saved, avg latency, resource heatmap |
| **Routing Intelligence** | Target distribution pie, decision log table, active policies |
| **Cost Analysis** | Actual cost vs all-cloud baseline, savings over time |
| **Resource Monitor** | CPU%, GPU%, RAM% gauges, queue depths bar chart |
| **Task Explorer** | Paginated searchable table of all tasks and routing decisions |

Enable **Auto-refresh (2s)** in the sidebar for live monitoring during demo runs.

---

## Development Setup

### Local (without Docker)

```bash
# Python 3.11+
pip install -e ".[dev]"

# Infrastructure only
docker compose up -d postgres redis

# Initialise database
python -c "import asyncio; from infrastructure.postgres_client import create_tables; asyncio.run(create_tables())"
python scripts/seed_db.py

# API (with hot reload)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Workers (separate terminals)
python -m workers.gpu_worker
python -m workers.cpu_worker
python -m workers.cloud_worker

# Dashboard
streamlit run dashboard/app.py --server.port 8501
```

### Makefile Reference

| Command | Description |
|---|---|
| `make install` | Install dev dependencies |
| `make dev` | Full stack via Docker Compose |
| `make test` | Run unit + integration tests |
| `make lint` | ruff + mypy checks |
| `make demo` | Run all simulate_workload.py scenarios |
| `make seed` | Re-seed routing policies into DB |
| `make migrate` | Run Alembic database migrations |
| `make train-model` | Train XGBoost routing model from execution history |
| `make logs` | Tail all container logs |
| `make down` | Stop and remove all containers |

---

## Testing

```bash
# All tests
pytest

# Unit tests (no external dependencies)
pytest tests/unit/ -v

# Integration tests (requires postgres + redis running)
pytest tests/integration/ -v

# Load test (requires API running)
locust -f tests/load/locustfile.py --host http://localhost:8000
```

| Suite | What It Tests |
|---|---|
| `test_workload_analyzer.py` | ExecutionProfile output for known inputs |
| `test_decision_engine.py` | Correct target for mocked resources and costs |
| `test_cost_calculator.py` | Token count x rate = expected cost |
| `test_rule_engine.py` | Each YAML rule fires for the correct conditions |
| `test_task_pipeline.py` | Full async: POST -> route -> GET result |
| `test_routing_fallback.py` | GPU unavailable -> fallback to CPU |
| `locustfile.py` | 100 concurrent users, P95 routing latency < 200ms |

---

## Project Structure

```
ai-smart-compute-orchestrator/
├── api/                         # FastAPI application
│   ├── main.py                  # App factory, lifespan, middleware
│   ├── dependencies.py          # DB session, API key injection
│   ├── middleware/
│   │   ├── auth.py              # X-API-Key validation
│   │   ├── rate_limiter.py      # Sliding window (100 req/min)
│   │   └── request_logger.py    # Structured request logging
│   └── routers/
│       ├── tasks.py             # POST/GET /v1/tasks
│       ├── results.py           # GET /v1/tasks/{id}/result
│       ├── routing.py           # Stats, decision log, policies
│       ├── metrics.py           # Prometheus metrics
│       └── health.py            # Liveness and readiness
│
├── core/
│   ├── analyzer/                # Request -> ExecutionProfile pipeline
│   ├── monitor/                 # CPU / GPU / queue resource monitoring
│   ├── intelligence/
│   │   ├── rule_engine.py       # Stage 1: YAML rule matching
│   │   ├── scoring_engine.py    # Stage 2: Weighted multi-factor scoring
│   │   ├── ml_engine.py         # Stage 3: XGBoost classifier
│   │   ├── decision_engine.py   # Orchestrates all three stages
│   │   └── cost_calculator.py   # Cost + latency estimation
│   ├── router/                  # Redis Stream dispatch + fallback
│   └── learning/                # Feedback collection + model retraining
│
├── workers/
│   ├── base_worker.py           # Abstract worker (streams, ack, callbacks)
│   ├── gpu_worker.py            # Ollama GPU inference
│   ├── cpu_worker.py            # Ollama CPU inference
│   ├── quantized_worker.py      # llama-cpp-python (GGUF Q4)
│   └── cloud_worker.py          # Google Gemini API
│
├── models/
│   ├── enums.py                 # TaskType, Priority, ExecutionTarget, etc.
│   ├── schemas.py               # Pydantic request/response schemas
│   └── database.py              # SQLAlchemy ORM models
│
├── infrastructure/
│   ├── redis_client.py          # Redis Streams helpers
│   ├── postgres_client.py       # Async SQLAlchemy session factory
│   ├── ollama_client.py         # Ollama HTTP client
│   └── celery_app.py            # Celery config (background jobs)
│
├── dashboard/
│   ├── app.py                   # Streamlit entrypoint
│   ├── data_fetcher.py          # API data fetching helpers
│   └── components/              # Chart components
│
├── config/
│   ├── settings.py              # Pydantic BaseSettings (env-driven)
│   ├── routing_policy.yaml      # Live routing rules (volume-mounted)
│   └── cost_config.yaml         # Cost and latency lookup tables
│
├── scripts/
│   ├── simulate_workload.py     # Demo scenarios
│   ├── seed_db.py               # Seed initial routing policies
│   └── benchmark_routing.py     # Compare routing stage performance
│
├── tests/
│   ├── unit/                    # Unit tests (no external deps)
│   ├── integration/             # End-to-end pipeline tests
│   └── load/                    # Locust load tests
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   ├── Dockerfile.dashboard
│   └── entrypoint.api.sh        # Auto-migrate + seed on first start
│
├── docker-compose.yml           # Full production stack
├── docker-compose.dev.yml       # Dev overrides (hot reload)
├── pyproject.toml               # Dependencies and tool config
├── Makefile                     # Developer commands
└── .env.example                 # Environment variable template
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Local inference | Ollama (GPU + CPU modes) |
| Quantized inference | llama-cpp-python (GGUF Q4) |
| Cloud LLM | Google Gemini API (google-genai SDK) |
| Task queue | Redis 7 Streams |
| Database | PostgreSQL 16 + asyncpg + SQLAlchemy 2 |
| ORM migrations | Alembic |
| ML routing model | XGBoost + scikit-learn |
| System monitoring | psutil (CPU/RAM) + pynvml (GPU) |
| Structured logging | structlog (JSON) |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| Testing | pytest + pytest-asyncio + Locust |
| Code quality | ruff + mypy |

---

## Observability

### Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `orchestrator_tasks_total` | counter | Total tasks by `status` label |
| `orchestrator_tasks_by_target` | counter | Tasks routed per `target` label |
| `orchestrator_avg_latency_ms` | gauge | Average execution latency by `target` |
| `orchestrator_cost_saved_usd_total` | counter | Cumulative savings vs all-cloud cost |
| `orchestrator_cpu_percent` | gauge | Current CPU utilisation |
| `orchestrator_gpu_percent` | gauge | Current GPU utilisation |
| `orchestrator_ram_percent` | gauge | Current RAM utilisation |

### Structured Logs

Every routing decision is logged in JSON with `task_id` on every line:

```json
{"event": "task.submitted",  "task_id": "550e...", "task_type": "REASONING", "priority": "HIGH"}
{"event": "task.routed",     "task_id": "550e...", "target": "CLOUD", "stage": "rule_based"}
{"event": "worker.completed","task_id": "550e...", "latency_ms": 1843, "tokens": 312}
```

---

## Security

- **Authentication**: `X-API-Key` header required on all non-health endpoints. Multiple keys supported as comma-separated list in `API_KEYS`.
- **Rate limiting**: 100 requests/minute per API key, enforced via Redis sliding window.
- **Input validation**: Pydantic v2 strict mode; `input_text` capped at 50,000 characters.
- **Secret safety**: Cloud API keys are never logged or returned in API responses.
- **SQL safety**: SQLAlchemy ORM only — no raw SQL, no string interpolation.
- **No shell injection**: GPU monitoring uses `pynvml` library, not subprocess calls.

---

## Routing Stage Upgrade Path

The system evolves through stages without any code changes — just a config flag:

| Stage | `ROUTING_STAGE` value | Characteristics |
|---|---|---|
| 1 — Rule-based | `rule` | Deterministic, zero ML dependency, fully auditable |
| 2 — Scored | `scored` | Adapts to load and historical performance automatically |
| 3 — ML Model | `ml` | Predicts from patterns in real execution history |
| 4 — RL (future) | — | Reinforcement learning from reward signal |

Change `ROUTING_STAGE` in `.env` and restart the API. No other changes required.

---

## Troubleshooting

**401 Unauthorized**

Check that `API_KEYS` in `.env` contains the key being sent. After editing `.env`, force-recreate the container:

```bash
docker compose up -d --force-recreate api
```

**Cloud worker gets 429 from Gemini**

The free tier quota is exhausted. Options:
- Switch `CLOUD_MODEL=gemini-1.5-flash` (separate quota pool)
- Wait ~24h for the daily quota to reset
- Enable billing on your Google AI project (pay-per-token, very cheap)

After changing `CLOUD_MODEL`, rebuild the worker:

```bash
docker compose stop worker-cloud && docker compose rm -f worker-cloud
docker compose build worker-cloud && docker compose up -d worker-cloud
```

**Tasks stuck in EXECUTING**

```bash
docker compose logs worker-gpu -f    # check for errors
ollama list                          # verify model is available on host
```

**GPU worker not using GPU**

Verify Ollama on the host is using the GPU:

```bash
ollama run mistral:7b-instruct-q4_0 "hello"
nvidia-smi   # GPU memory should increase while running
```

If no GPU is detected, the resource monitor sets `gpu_available=false` and all GPU-targeted tasks fall back to CPU.

**Dashboard shows "API: Offline"**

```bash
docker compose ps          # confirm api container is running
docker compose logs api    # check for startup errors
```

The dashboard connects to `http://api:8000` inside the Docker network by default.

---

## License

MIT
