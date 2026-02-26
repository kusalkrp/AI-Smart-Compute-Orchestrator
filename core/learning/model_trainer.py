"""ML Model Trainer — trains XGBoost routing classifier from execution logs."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import structlog
from celery import shared_task

from config.settings import settings

logger = structlog.get_logger(__name__)

_TARGET_LABELS = {"GPU": 0, "CPU": 1, "QUANTIZED": 2, "CLOUD": 3}
_LABEL_TARGETS = {v: k for k, v in _TARGET_LABELS.items()}


class ModelTrainer:
    def __init__(self) -> None:
        self._model_path = Path(settings.ML_MODEL_PATH)
        self._model_path.parent.mkdir(parents=True, exist_ok=True)

    async def fetch_training_data(self, days: int = 30) -> tuple:
        """Fetch execution logs + resource snapshots for training."""
        from infrastructure.postgres_client import AsyncSessionLocal
        from models.database import ExecutionLog, RoutingDecision, ResourceSnapshot
        from sqlalchemy import select, join

        since = datetime.utcnow() - timedelta(days=days)

        async with AsyncSessionLocal() as session:
            # Join execution_logs with routing_decisions for features
            result = await session.execute(
                select(
                    RoutingDecision.target,
                    RoutingDecision.confidence,
                    ExecutionLog.actual_latency_ms,
                    ExecutionLog.actual_cost_usd,
                    ExecutionLog.tokens_used,
                    ExecutionLog.success,
                    ExecutionLog.target_used,  # actual target (ground truth)
                )
                .join(ExecutionLog, ExecutionLog.task_id == RoutingDecision.task_id)
                .where(ExecutionLog.created_at >= since)
                .where(ExecutionLog.success.is_(True))
                .limit(10000)
            )
            rows = result.fetchall()

        return rows

    async def train(self) -> Optional[dict]:
        """Train XGBoost classifier and save to disk."""
        try:
            import numpy as np
            from xgboost import XGBClassifier
            import joblib
        except ImportError as e:
            logger.warning("model_trainer.missing_dependency", error=str(e))
            return None

        rows = await self.fetch_training_data()
        if len(rows) < 50:
            logger.info("model_trainer.insufficient_data", count=len(rows))
            return None

        # Build feature matrix
        X = []
        y = []
        for row in rows:
            predicted_target, confidence, lat_ms, cost_usd, tokens, success, actual_target = row
            if actual_target not in _TARGET_LABELS:
                continue

            features = [
                min(1.0, tokens / 8192.0),    # normalized token count
                cost_usd,
                lat_ms / 10000.0,              # normalized latency
                float(success),
                confidence,
                1.0 if actual_target == "GPU" else 0.0,
                1.0 if actual_target == "CPU" else 0.0,
                1.0 if actual_target == "QUANTIZED" else 0.0,
                1.0 if actual_target == "CLOUD" else 0.0,
            ]
            X.append(features)
            y.append(_TARGET_LABELS[actual_target])

        if len(X) < 50:
            logger.info("model_trainer.insufficient_valid_data", count=len(X))
            return None

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=int)

        # Train classifier
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save model
        joblib.dump(clf, self._model_path)

        metrics = {
            "accuracy": accuracy,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "trained_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            "model_trainer.training_complete",
            accuracy=accuracy,
            samples=len(X_arr),
            model_path=str(self._model_path),
        )

        return metrics


@shared_task(name="core.learning.model_trainer.train_routing_model")
def train_routing_model() -> None:
    """Celery beat task: retrain routing model nightly."""
    import asyncio
    trainer = ModelTrainer()
    asyncio.run(trainer.train())
