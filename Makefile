.PHONY: dev test build demo clean install lint migrate seed

# ─── Development ──────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres redis
	@echo "Waiting for services to start..."
	@sleep 3
	alembic upgrade head
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
	python -m workers.gpu_worker &
	python -m workers.cpu_worker &
	python -m workers.quantized_worker &
	python -m workers.cloud_worker &
	streamlit run dashboard/app.py --server.port 8501 &
	@echo "✓ All services running"
	@echo "  API:       http://localhost:8000"
	@echo "  Docs:      http://localhost:8000/docs"
	@echo "  Dashboard: http://localhost:8501"

dev-infra:
	docker compose up -d postgres redis
	@sleep 2

dev-api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

dev-worker-gpu:
	python -m workers.gpu_worker

dev-worker-cpu:
	python -m workers.cpu_worker

dev-worker-cloud:
	python -m workers.cloud_worker

dev-dashboard:
	streamlit run dashboard/app.py --server.port 8501

# ─── Database ─────────────────────────────────────────────────────────────────

migrate:
	alembic upgrade head

migrate-create:
	alembic revision --autogenerate -m "$(name)"

migrate-down:
	alembic downgrade -1

seed:
	python scripts/seed_db.py

# ─── Testing ──────────────────────────────────────────────────────────────────

test:
	pytest tests/unit/ tests/integration/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-load:
	locust -f tests/load/locustfile.py --host http://localhost:8000

test-cov:
	pytest tests/unit/ tests/integration/ --cov=. --cov-report=html --cov-report=term

# ─── Demo ─────────────────────────────────────────────────────────────────────

demo:
	python scripts/simulate_workload.py --scenario all

demo-urgency:
	python scripts/simulate_workload.py --scenario urgency

demo-cost:
	python scripts/simulate_workload.py --scenario cost

demo-resource:
	python scripts/simulate_workload.py --scenario resource

demo-learning:
	python scripts/simulate_workload.py --scenario learning

benchmark:
	python scripts/benchmark_routing.py

# ─── Code Quality ─────────────────────────────────────────────────────────────

lint:
	ruff check . --fix
	mypy api/ core/ workers/ models/ infrastructure/

format:
	ruff format .

# ─── Build & Deploy ───────────────────────────────────────────────────────────

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage 2>/dev/null || true

# ─── ML Model ─────────────────────────────────────────────────────────────────

train-model:
	python -c "from core.learning.model_trainer import ModelTrainer; import asyncio; asyncio.run(ModelTrainer().train())"

# ─── Quick health check ───────────────────────────────────────────────────────

health:
	curl -s http://localhost:8000/v1/health | python -m json.tool
	curl -s http://localhost:8000/v1/health/ready | python -m json.tool

stats:
	curl -s http://localhost:8000/v1/routing/stats | python -m json.tool
