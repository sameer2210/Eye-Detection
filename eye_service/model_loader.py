# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Singleton model loader and manager for Ultralytics YOLO-World zero-shot eye detector."""

import logging
import threading
import numpy as np
from ultralytics import YOLOWorld
from eye_service.config import get_settings

logger = logging.getLogger("eye_service.model_loader")


class EyeModelManager:
    """Thread-safe manager for initializing, caching, and serving the YOLO-World model instance."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize empty manager state; call load_model() to load weights."""
        self.model: YOLOWorld | None = None
        self._is_loaded = False
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "EyeModelManager":
        """Return singleton instance of EyeModelManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self) -> YOLOWorld:
        """Load YOLO-World model, configure zero-shot offline prompt class, and perform warm-up.

        Returns:
            YOLOWorld: Loaded and warmed-up Ultralytics YOLOWorld instance.
        """
        if self._is_loaded and self.model is not None:
            return self.model

        with self._load_lock:
            if self._is_loaded and self.model is not None:
                return self.model

            settings = get_settings()
            logger.info("Loading YOLO-World model from %s on device=%s...", settings.model_path, settings.device)

            model = YOLOWorld(settings.model_path)
            
            # Configure zero-shot offline vocabulary text prompt ensemble
            if isinstance(settings.prompt, str):
                prompt_classes = [p.strip() for p in settings.prompt.split(",") if p.strip()]
            else:
                prompt_classes = list(settings.prompt)

            logger.info("Configuring zero-shot offline vocabulary prompt ensemble: %s", prompt_classes)
            model.set_classes(prompt_classes)

            # Perform model warmup pass
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            model.predict(source=dummy_img, verbose=False)

            self.model = model
            self._is_loaded = True
            logger.info("YOLO-World model initialized and warmed up successfully.")
            return self.model

    @property
    def is_loaded(self) -> bool:
        """Return True if model has been successfully initialized."""
        return self._is_loaded


def get_model() -> YOLOWorld:
    """Dependency helper to get loaded YOLOWorld model instance."""
    return EyeModelManager.get_instance().load_model()
