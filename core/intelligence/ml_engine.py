from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import structlog

from config.settings import settings
from models.enums import ExecutionTarget
from models.schemas import ExecutionProfile, ResourceSnapshot

logger = structlog.get_logger(__name__)


class MLEngine:
    """
    Stage 3: XGBoost classifier for routing decisions.

    Input features:
        complexity_score, estimated_tokens, latency_sensitivity, cost_sensitivity,
        cpu_percent, gpu_percent, gpu_vram_used_mb, cpu_queue_depth,
        gpu_queue_depth, quantized_queue_depth, cloud_queue_depth

    Output: predicted ExecutionTarget (multiclass classification)
    """

    _TARGET_CLASSES = [
        ExecutionTarget.GPU,
        ExecutionTarget.CPU,
        ExecutionTarget.QUANTIZED,
        ExecutionTarget.CLOUD,
    ]

    _TARGET_MODELS = {
        ExecutionTarget.GPU: settings.GPU_MODEL,
        ExecutionTarget.CPU: settings.CPU_MODEL,
        ExecutionTarget.QUANTIZED: settings.CPU_MODEL,
        ExecutionTarget.CLOUD: settings.CLOUD_MODEL,
    }

    def __init__(self) -> None:
        self._model = None
        self._model_loaded = False
        self._try_load_model()

    def _try_load_model(self) -> None:
        model_path = Path(settings.ML_MODEL_PATH)
        if not model_path.exists():
            logger.info("ml_engine.no_model_file", path=str(model_path))
            return
        try:
            import joblib
            self._model = joblib.load(model_path)
            self._model_loaded = True
            logger.info("ml_engine.model_loaded", path=str(model_path))
        except Exception as e:
            logger.warning("ml_engine.load_failed", error=str(e))

    def is_available(self) -> bool:
        return self._model_loaded and self._model is not None

    def predict(
        self,
        profile: ExecutionProfile,
        resource: ResourceSnapshot,
    ) -> tuple[ExecutionTarget, str, float]:
        """
        Returns (target, model_name, confidence).
        Raises RuntimeError if model not available.
        """
        if not self.is_available():
            raise RuntimeError("ML model not available")

        import numpy as np
        features = self._extract_features(profile, resource)
        X = np.array([features])

        try:
            probabilities = self._model.predict_proba(X)[0]
            predicted_class_idx = int(probabilities.argmax())
            confidence = float(probabilities[predicted_class_idx])

            target = self._TARGET_CLASSES[predicted_class_idx % len(self._TARGET_CLASSES)]
            model_name = self._TARGET_MODELS.get(target, settings.CPU_MODEL)

            logger.debug(
                "ml_engine.prediction",
                target=target.value,
                confidence=confidence,
                probs=probabilities.tolist(),
            )

            return target, model_name, confidence
        except Exception as e:
            logger.warning("ml_engine.predict_failed", error=str(e))
            raise RuntimeError(f"ML prediction failed: {e}") from e

    def _extract_features(
        self,
        profile: ExecutionProfile,
        resource: ResourceSnapshot,
    ) -> list[float]:
        return [
            profile.complexity_score,
            min(1.0, profile.estimated_tokens / 8192.0),  # normalized
            profile.latency_sensitivity,
            profile.cost_sensitivity,
            float(profile.is_batch),
            float(profile.requires_reasoning),
            resource.cpu_percent / 100.0,
            resource.gpu_percent / 100.0,
            min(1.0, resource.gpu_vram_used_mb / 6144.0),  # normalize to 6GB
            float(resource.gpu_available),
            min(1.0, resource.cpu_queue_depth / 20.0),
            min(1.0, resource.gpu_queue_depth / 20.0),
            min(1.0, resource.quantized_queue_depth / 20.0),
            min(1.0, resource.cloud_queue_depth / 100.0),
        ]

    def reload(self) -> None:
        self._model = None
        self._model_loaded = False
        self._try_load_model()
