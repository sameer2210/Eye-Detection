# Eye Detection Service

The **Eye Detection Service** is a high-performance, zero-shot microservice powered by Ultralytics YOLO-World and FastAPI. Engineered specifically for production runtime deployments, it processes uploaded images (JPEG, PNG, WEBP, BMP) in real time to identify human eye regions, determine presence confidence, and return original pixel coordinate bounding boxes. The service features pre-warmed model loading, bicubic resolution scaling for low-resolution eye crops, multi-layer security validation, and complete containerization via Docker.

---

## Features

- **FastAPI REST API**: High-throughput asynchronous web application with interactive OpenAPI documentation (`/docs`) and request correlation tracing.
- **Zero-Shot YOLO-World Inference**: Powered by `yolov8s-world.pt` with configurable prompt ensembles (`eye`, `human eye`, `eyes`, `iris`, `pupil`).
- **Comprehensive Image Security Validation**: Defense-in-depth image inspection enforcing file size limits, magic header binary verification (anti-spoofing), minimum/maximum dimension bounds, and aspect ratio checks.
- **Image Preprocessing & BBox Scaling**: Automatic EXIF orientation normalization and bicubic upscaling for small crop inputs (<128px) with exact coordinate re-mapping back to original image space.
- **Pre-Warmed Startup Lifespan**: Thread-safe model initialization and dummy array pre-warm pass on service boot to guarantee zero cold-start latency for first requests.
- **Docker Ready**: Production-grade `Dockerfile` and `.dockerignore` for containerized deployment across cloud and edge environments.
- **Configurable Runtime**: Full environment variable support via Pydantic Settings for device target (`cpu`, `cuda`), confidence thresholding, and prompt tuning.

---

## Repository Structure

```
Eye-Detection/
├── .dockerignore              # Excludes non-essential build contexts during Docker build
├── .gitignore                 # Excludes temporary artifacts from version control
├── pyproject.toml             # Minimal production dependency manifest
├── README.md                  # Complete service documentation
├── yolov8s-world.pt           # Production YOLO-World zero-shot model weights (27.1 MB)
└── eye_service/
    ├── Dockerfile             # Multi-stage optimized Docker build specification
    ├── __init__.py            # Package initialization marker
    ├── api.py                 # FastAPI router defining /health and /detect-eye endpoints
    ├── config.py              # Application configuration settings with environment overrides
    ├── main.py                # Application entrypoint, lifespan manager, and Uvicorn runner
    ├── model_loader.py        # Thread-safe singleton model loader and pre-warm manager
    ├── predictor.py           # Core inference pipeline orchestrating detection & formatting
    ├── schemas.py             # Pydantic request/response data contract models
    └── utils.py               # Security validation, PIL decoding, and coordinate scaling
```

---

## Requirements

- **Python**: Version 3.10, 3.11, or 3.12 (Python 3.11 recommended)
- **Supported Operating Systems**: Linux (Ubuntu 20.04+), macOS, Microsoft Windows 10/11
- **Hardware Recommendations**:
  - **CPU**: 2+ vCPUs (Intel Xeon / AMD EPYC or ARM64)
  - **RAM**: Minimum 2 GB (4 GB recommended)
  - **GPU (Optional)**: NVIDIA GPU with CUDA 11.8/12.1+ support for acceleration

---

## Installation

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/Eye-Detection.git
   cd Eye-Detection
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install production dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

4. **Run the API service**:
   ```bash
   python -m eye_service.main
   ```
   The service will start listening at `http://0.0.0.0:8000`.

---

## Docker

### Build Image
```bash
docker build -t eye-detection-service -f eye_service/Dockerfile .
```

### Run Container
```bash
docker run -d -p 8000:8000 --name eye-detection eye-detection-service
```

Access API docs at `http://localhost:8000/docs` and health check at `http://localhost:8000/health`.

---

## API Documentation

### 1. Health Check

- **Endpoint**: `GET /health`
- **Summary**: Health and Model Status Check

#### Response Example (200 OK)
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "version": "8.4.105"
}
```

---

### 2. Detect Eye

- **Endpoint**: `POST /detect-eye`
- **Summary**: Zero-Shot Eye Detection Endpoint
- **Content-Type**: `multipart/form-data`
- **Parameters**: `file` (UploadFile, required) — Supported formats: `JPEG`, `PNG`, `WEBP`, `BMP`.

#### Response Example — Eye Detected (200 OK)
```json
{
  "success": true,
  "eyeDetected": true,
  "confidence": 0.8745,
  "boundingBox": {
    "x": 142,
    "y": 88,
    "width": 110,
    "height": 95
  },
  "processingTime": 42
}
```

#### Response Example — No Eye Detected (200 OK)
```json
{
  "success": true,
  "eyeDetected": false,
  "confidence": 0.0,
  "boundingBox": null,
  "processingTime": 35
}
```

#### Error Responses
- `400 Bad Request`: Empty file, corrupted image data, or invalid aspect ratio.
- `413 Payload Too Large`: Image file size exceeds `MAX_IMAGE_SIZE_MB` (default: 10.0 MB).
- `415 Unsupported Media Type`: Image format not in supported list.
- `500 Internal Server Error`: Unexpected error during processing (logs traceback internally).

---

## Configuration

All configuration options are defined in `eye_service/config.py` and can be overridden via environment variables:

| Environment Variable | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `HOST` | `str` | `0.0.0.0` | Host IP address for Uvicorn binding. |
| `PORT` | `int` | `8000` | Port number for Uvicorn binding. |
| `LOG_LEVEL` | `str` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `MODEL_PATH` | `str` | `yolov8s-world.pt` | Path to YOLO-World production model weights. |
| `CONFIDENCE_THRESHOLD` | `float` | `0.008` | Minimum detection confidence threshold. |
| `PROMPT` | `str` | `eye, human eye, eyes, iris, pupil` | Comma-separated zero-shot vocabulary prompt ensemble. |
| `DEVICE` | `str` | `cpu` | Hardware execution target (`cpu` or `cuda`). |
| `MAX_IMAGE_SIZE_MB` | `float` | `10.0` | Maximum allowed upload image file size in Megabytes. |
| `MIN_IMAGE_DIM` | `int` | `8` | Minimum allowed width/height in pixels. |
| `MAX_IMAGE_DIM` | `int` | `4096` | Maximum allowed width/height in pixels. |
| `MIN_ASPECT_RATIO` | `float` | `0.2` | Minimum aspect ratio limit (`width / height`). |
| `MAX_ASPECT_RATIO` | `float` | `5.0` | Maximum aspect ratio limit (`width / height`). |

---

## Model Architecture & Zero-Shot Detection

The Eye Detection Service uses **YOLO-World** (`yolov8s-world.pt`), an open-vocabulary object detection model. 

- **Offline Vocabulary Prompts**: Rather than restricting predictions to pre-trained COCO classes, the service configures a prompt ensemble (`eye`, `human eye`, `eyes`, `iris`, `pupil`).
- **Confidence Scoring**: Predictions above `CONFIDENCE_THRESHOLD` are extracted, and the highest-confidence bounding box is formatted into pixel coordinates (`x`, `y`, `width`, `height`).

---

## Security Protections

1. **Magic Header Inspection**: Validates binary signature magic bytes (`FF D8 FF` for JPEG, `89 50 4E 47` for PNG, `RIFF...WEBP` for WEBP, `BM` for BMP) to prevent file extension spoofing and malicious payload upload.
2. **File Size Capping**: Rejects requests larger than `MAX_IMAGE_SIZE_MB` before reading excessive data into memory.
3. **PIL Integrity Verification**: Executes `Image.verify()` to catch corrupted images and malformed byte streams.
4. **Dimension & Aspect Ratio Boundaries**: Enforces strict bounds to protect against memory exhaustion attacks (Decompression Bomb / Zip Bomb).
5. **Sanitized Exception Handling**: Returns structured JSON error payloads (`ErrorResponse`) to API consumers without leaking internal Python stack trace details.

---

## Performance Optimizations

- **Pre-Warmed Startup**: Model weight loading and dummy image execution occur during application boot (`lifespan`), eliminating cold-start request latency.
- **Bicubic Crop Upscaling**: Low-resolution eye crops (<128px) undergo bicubic upscaling prior to feature extraction to maintain spatial resolution on small inputs.
- **Single-Pass Tensor Operations**: Utilizes vectorized NumPy operations for bounding box filtering and argmax confidence selection.

---

## Development & Architecture Notes

- **Separation of Concerns**: Modular structure isolating configuration (`config.py`), API routing (`api.py`), model state (`model_loader.py`), prediction execution (`predictor.py`), and validation (`utils.py`).
- **Singleton Model Manager**: `EyeModelManager` uses a double-checked locking thread-safe singleton pattern to manage the single model instance across concurrent API requests.
- **Asynchronous & Threading**: Non-blocking request correlation middleware attaches unique `X-Request-ID` tracing headers to all inbound HTTP requests.
