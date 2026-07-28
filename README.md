# Eye Detection Service

Production-ready microservice built with **FastAPI** and custom-trained **Ultralytics YOLOv8** for human eye detection.

---

## Quick Start (Docker)

### Build Docker Image
```bash
docker build -t eye-detection-service .
```

### Run Container
```bash
docker run -d \
  --name eye-detection \
  -p 8000:8000 \
  --env CONFIDENCE_THRESHOLD=0.40 \
  eye-detection-service
```

---

## API Endpoints

### 1. Health & Readiness Probe
`GET /health`

**Response (`200 OK` when ready, `503 Service Unavailable` when unready):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/best.pt",
  "framework": "Ultralytics YOLOv8",
  "readiness": true,
  "device": "cpu",
  "version": "8.2.50"
}
```

### 2. Detect Eye in Image
`POST /detect-eye`

**Request:** `multipart/form-data` with `file` field containing an image (`JPEG`, `PNG`, `WEBP`, `BMP`).

**Response (`200 OK`):**
```json
{
  "success": true,
  "eyeDetected": true,
  "confidence": 0.8542,
  "boundingBox": {
    "x": 120,
    "y": 85,
    "width": 64,
    "height": 42
  },
  "processingTime": 45
}
```

---

## Environment Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | API bind address |
| `PORT` | `8000` | API bind port |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MODEL_PATH` | `models/best.pt` | Path to custom YOLOv8 weights |
| `CONFIDENCE_THRESHOLD` | `0.40` | Detection confidence threshold filter |
| `IMAGE_SIZE` | `640` | Model inference input resolution (pixels) |
| `DEVICE` | `cpu` | PyTorch execution device (`cpu` or `cuda`) |
| `MAX_IMAGE_SIZE_MB` | `10.0` | Maximum uploaded file size limit in MB |
| `MIN_IMAGE_DIM` | `8` | Minimum image dimension in pixels |
| `MAX_IMAGE_DIM` | `4096` | Maximum image dimension in pixels |
