from celery import Celery
from celery.schedules import crontab

from config.settings import settings

celery_app = Celery(
    "orchestrator",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["core.learning.model_trainer", "core.learning.routing_optimizer"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_schedule={
        "retrain-ml-model-nightly": {
            "task": "core.learning.model_trainer.train_routing_model",
            "schedule": crontab(hour=2, minute=0),  # 2 AM UTC daily
        },
        "optimize-routing-weights-weekly": {
            "task": "core.learning.routing_optimizer.optimize_weights",
            "schedule": crontab(hour=3, minute=0, day_of_week=0),  # Sunday 3 AM UTC
        },
        "update-model-performance-cache": {
            "task": "core.learning.analytics.update_performance_cache",
            "schedule": 300.0,  # every 5 minutes
        },
    },
)
