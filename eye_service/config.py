# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Configuration management for Eye Detection Service via environment variables."""

import os
from functools import lru_cache
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings with defaults overrideable by environment variables."""

    # Server Configuration
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Model Configuration
    model_path: str = Field(default_factory=lambda: os.getenv("MODEL_PATH", "yolov8s-world.pt"))
    confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.008"))
    )
    prompt: str = Field(default_factory=lambda: os.getenv("PROMPT", "eye, human eye, eyes, iris, pupil"))
    device: str = Field(default_factory=lambda: os.getenv("DEVICE", "cpu"))

    # Security & Image Validation Configuration
    max_image_size_mb: float = Field(
        default_factory=lambda: float(os.getenv("MAX_IMAGE_SIZE_MB", "10.0"))
    )
    min_image_dim: int = Field(default_factory=lambda: int(os.getenv("MIN_IMAGE_DIM", "8")))
    max_image_dim: int = Field(default_factory=lambda: int(os.getenv("MAX_IMAGE_DIM", "4096")))
    min_aspect_ratio: float = Field(
        default_factory=lambda: float(os.getenv("MIN_ASPECT_RATIO", "0.2"))
    )
    max_aspect_ratio: float = Field(
        default_factory=lambda: float(os.getenv("MAX_ASPECT_RATIO", "5.0"))
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of Settings."""
    return Settings()
