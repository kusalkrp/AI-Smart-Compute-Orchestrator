#!/bin/sh
set -e

echo ">>> Creating database tables..."
python -c "import asyncio; from infrastructure.postgres_client import create_tables; asyncio.run(create_tables())"

echo ">>> Seeding initial data..."
python scripts/seed_db.py

echo ">>> Starting API server..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
