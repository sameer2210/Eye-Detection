# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Inference predictor pipeline for eye detection using custom-trained Ultralytics YOLOv8."""

import time
import logging
import numpy as np

from eye_service.config import get_settings
from eye_service.model_loader import EyeModelManager
from eye_service.schemas import EyeDetectionResponse, BoundingBox
from eye_service.utils import load_and_preprocess_image, format_bounding_box

logger = logging.getLogger("eye_service.predictor")


class EyePredictor:
    """Predictor class orchestrating image loading, YOLOv8 inference, and output formatting."""

    def __init__(self):
        """Initialize EyePredictor with settings reference."""
        self.settings = get_settings()

    def predict(self, image_bytes: bytes) -> EyeDetectionResponse:
        """Execute complete eye detection pipeline on raw input image bytes.

        Args:
            image_bytes (bytes): Raw bytes of input image file.

        Returns:
            EyeDetectionResponse: Structured prediction response matching required specification.
        """
        start_time = time.perf_counter()

        # 1. Preprocess and validate image
        img_bgr, scale_x, scale_y = load_and_preprocess_image(image_bytes)

        # 2. Get loaded model manager & instance
        manager = EyeModelManager.get_instance()
        model = manager.load_model()

        # 3. Run thread-safe inference
        conf_threshold = self.settings.confidence_threshold
        with manager.inference_lock:
            results = model.predict(
                source=img_bgr,
                conf=conf_threshold,
                imgsz=self.settings.image_size,
                device=self.settings.device,
                verbose=False,
            )

        top_box: BoundingBox | None = None
        top_confidence = 0.0
        eye_detected = False

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                # Find detection with highest confidence score
                confidences = boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confidences))
                best_conf = float(confidences[best_idx])

                if best_conf >= conf_threshold:
                    xyxy = boxes.xyxy[best_idx].cpu().numpy()
                    top_box = format_bounding_box(xyxy, scale_x=scale_x, scale_y=scale_y)
                    top_confidence = round(best_conf, 4)
                    eye_detected = True

                    logger.info(
                        "Eye detected! Confidence=%.4f, BBox=[x=%d, y=%d, w=%d, h=%d], Total Detections=%d",
                        top_confidence,
                        top_box.x,
                        top_box.y,
                        top_box.width,
                        top_box.height,
                        len(boxes),
                    )

        if not eye_detected:
            logger.info("No eye detected above threshold (conf_threshold=%.2f).", conf_threshold)

        elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))

        return EyeDetectionResponse(
            success=True,
            eyeDetected=eye_detected,
            confidence=top_confidence,
            boundingBox=top_box,
            processingTime=elapsed_ms,
        )


_predictor_instance = None


def get_predictor() -> EyePredictor:
    """Dependency provider for EyePredictor singleton."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = EyePredictor()
    return _predictor_instance
