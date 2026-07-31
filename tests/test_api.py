"""Unit test suite for Eye Detection Service FastAPI endpoints, security headers, configuration, and image validation utilities."""

import io
from unittest.mock import MagicMock, PropertyMock, patch
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from eye_service.api import _get_dynamic_model_version
from eye_service.config import get_settings
from eye_service.main import app
from eye_service.model_loader import EyeModelManager
from eye_service.utils import (
    ImageValidationError,
    format_bounding_box,
    validate_image_bytes,
)


def create_synthetic_image(width: int = 200, height: int = 200, color: tuple = (128, 128, 128)) -> bytes:
    """Helper function to create synthetic JPEG image bytes for testing."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_check_healthy():
    """Test /health endpoint returns 200 OK and expected diagnostic metadata when model is loaded."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model_path"] == "models/best.pt"
        assert data["framework"] == "Ultralytics YOLOv8"
        assert data["readiness"] is True
        assert "device" in data
        assert "version" in data


def test_health_check_unhealthy():
    """Test /health endpoint returns 503 Service Unavailable when model is not loaded."""
    with TestClient(app) as client:
        with patch.object(EyeModelManager, "is_loaded", new_callable=PropertyMock, return_value=False):
            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False
            assert data["readiness"] is False


def test_security_and_tracing_headers():
    """Test response contains security and tracing middleware headers."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-MS" in response.headers
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("Cache-Control") == "no-store"
        assert response.headers.get("Pragma") == "no-cache"


def test_detect_eye_empty_file():
    """Test /detect-eye returns 400 Bad Request when 0-byte file is uploaded."""
    with TestClient(app) as client:
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        response = client.post("/detect-eye", files=files)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "IMAGE_VALIDATION_ERROR"
        assert "empty" in data["message"].lower()


def test_detect_eye_invalid_format():
    """Test /detect-eye returns 415 Unsupported Media Type for text/plain upload."""
    with TestClient(app) as client:
        files = {"file": ("test.txt", b"Hello world", "text/plain")}
        response = client.post("/detect-eye", files=files)
        assert response.status_code == 415
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "IMAGE_VALIDATION_ERROR"


def test_detect_eye_oversized_file():
    """Test /detect-eye returns 413 Payload Too Large when file size exceeds limit."""
    settings = get_settings()
    # Create dummy byte string exceeding max_image_size_mb with valid JPEG header
    oversized_bytes = b"\xff\xd8\xff" + b"\x00" * int((settings.max_image_size_mb + 1) * 1024 * 1024)
    with TestClient(app) as client:
        files = {"file": ("large.jpg", oversized_bytes, "image/jpeg")}
        response = client.post("/detect-eye", files=files)
        assert response.status_code == 413
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "IMAGE_VALIDATION_ERROR"
        assert "exceeds maximum allowed size" in data["message"]


def test_detect_eye_small_dimension_image():
    """Test /detect-eye returns 400 Bad Request for images with dimensions below min_image_dim."""
    img_bytes = create_synthetic_image(width=4, height=4)
    with TestClient(app) as client:
        files = {"file": ("small.jpg", img_bytes, "image/jpeg")}
        response = client.post("/detect-eye", files=files)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "IMAGE_VALIDATION_ERROR"
        assert "smaller than minimum allowed" in data["message"]


def test_detect_eye_valid_image():
    """Test /detect-eye returns 200 OK and valid EyeDetectionResponse structure for synthetic image."""
    with TestClient(app) as client:
        img_bytes = create_synthetic_image(300, 300)
        files = {"file": ("synthetic.jpg", img_bytes, "image/jpeg")}
        response = client.post("/detect-eye", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["eyeDetected"], bool)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["processingTime"], int)


def test_config_settings_defaults():
    """Test default configuration settings match expected production settings."""
    settings = get_settings()
    assert settings.port == 8001
    assert settings.host == "0.0.0.0"
    assert settings.confidence_threshold == 0.40
    assert settings.max_image_size_mb == 10.0


def test_dynamic_model_version():
    """Test dynamic model version detection logic under various model manager states."""
    mock_manager = MagicMock()
    mock_manager.is_loaded = True
    mock_manager.model = MagicMock()
    mock_manager.model.version = "1.2.3-custom"

    assert _get_dynamic_model_version(mock_manager) == "1.2.3-custom"

    # Test fallback to checkpoint dictionary version
    del mock_manager.model.version
    mock_manager.model.ckpt = {"version": "2.0.0-ckpt"}
    assert _get_dynamic_model_version(mock_manager) == "2.0.0-ckpt"

    # Test fallback when model is not loaded
    mock_manager.is_loaded = False
    mock_manager.model = None
    version_str = _get_dynamic_model_version(mock_manager)
    assert version_str != "unknown" and len(version_str) > 0


def test_format_bounding_box():
    """Test bounding box formatting and coordinate scaling."""
    xyxy = [10.0, 20.0, 110.0, 120.0]
    bbox = format_bounding_box(xyxy, scale_x=2.0, scale_y=2.0)
    assert bbox.x == 5
    assert bbox.y == 10
    assert bbox.width == 50
    assert bbox.height == 50
