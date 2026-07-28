# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Singleton model loader and manager for custom-trained Ultralytics YOLOv8 eye detector."""

import os
import logging
import threading
import numpy as np
from ultralytics import YOLO
from eye_service.config import get_settings

logger = logging.getLogger("eye_service.model_loader")


class EyeModelManager:
    """Thread-safe manager for initializing, caching, and serving the YOLOv8 model instance."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize empty manager state; call load_model() to load weights."""
        self.model: YOLO | None = None
        self._is_loaded = False
        self._load_lock = threading.Lock()
        self.inference_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "EyeModelManager":
        """Return singleton instance of EyeModelManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self) -> YOLO:
        """Load custom YOLOv8 model, perform fail-fast validation, and run warm-up pass.

        Returns:
            YOLO: Loaded and warmed-up Ultralytics YOLO instance.
        """
        if self._is_loaded and self.model is not None:
            return self.model

        with self._load_lock:
            if self._is_loaded and self.model is not None:
                return self.model

            settings = get_settings()
            
            # Fail-fast validation of model weights path
            if not os.path.exists(settings.model_path):
                error_msg = f"Model file not found at path '{settings.model_path}'. Please verify custom model weights exist."
                logger.critical(error_msg)
                raise FileNotFoundError(error_msg)

            logger.info("Loading custom YOLOv8 model from %s on device=%s...", settings.model_path, settings.device)

            model = YOLO(settings.model_path)

            # Perform model warmup pass using configured resolution
            dummy_img = np.zeros((settings.image_size, settings.image_size, 3), dtype=np.uint8)
            model.predict(source=dummy_img, imgsz=settings.image_size, verbose=False)

            self.model = model
            self._is_loaded = True
            logger.info("Custom YOLOv8 model initialized and warmed up successfully.")
            return self.model

    @property
    def is_loaded(self) -> bool:
        """Return True if model has been successfully initialized."""
        return self._is_loaded


def get_model() -> YOLO:
    """Dependency helper to get loaded YOLO model instance."""
    return EyeModelManager.get_instance().load_model()

