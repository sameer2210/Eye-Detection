# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pydantic schemas for request and response validation in Eye Detection Service."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected eye region in image pixel coordinates."""

    x: int = Field(..., description="Top-left x-coordinate of the bounding box")
    y: int = Field(..., description="Top-left y-coordinate of the bounding box")
    width: int = Field(..., description="Width of the bounding box in pixels")
    height: int = Field(..., description="Height of the bounding box in pixels")


class EyeDetectionResponse(BaseModel):
    """Standardized API response for eye detection request."""

    success: bool = Field(..., description="Indicates whether the request was processed successfully")
    eyeDetected: bool = Field(..., description="True if a human eye was detected above threshold")
    confidence: float = Field(..., description="Confidence score of the top eye detection (0.0 to 1.0)")
    boundingBox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box of top detected eye, or null if no eye detected"
    )
    processingTime: int = Field(..., description="End-to-end processing time in milliseconds")


class HealthCheckResponse(BaseModel):
    """API response for server health check endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Service status (e.g. 'ok')")
    model_loaded: bool = Field(..., description="Whether the YOLO model is initialized")
    model_path: str = Field(default="models/best.pt", description="Path to loaded model weights")
    framework: str = Field(default="Ultralytics YOLOv8", description="ML framework description")
    readiness: bool = Field(default=True, description="Service readiness status")
    device: str = Field(..., description="Inference execution device ('cpu' or 'cuda')")
    version: str = Field(..., description="Service version string")


class ErrorResponse(BaseModel):
    """Structured error payload for failed requests."""

    success: bool = Field(default=False, description="Always False for error responses")
    error: str = Field(..., description="Short error classification")
    message: str = Field(..., description="Detailed user-facing error message")
    request_id: Optional[str] = Field(default=None, description="Unique trace request ID")
