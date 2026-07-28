"""Unit test suite for Eye Detection Service FastAPI endpoints."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from eye_service.main import app


def create_synthetic_image(width: int = 200, height: int = 200, color: tuple = (128, 128, 128)) -> bytes:
    """Helper function to create synthetic JPEG image bytes for testing."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_check():
    """Test /health endpoint returns 200 OK and expected diagnostic metadata."""
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
