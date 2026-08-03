# Eye Detection Service — High-Performance Human Eye Detection API

> Production-ready, security-hardened microservice built with **FastAPI** and custom-trained **Ultralytics YOLOv8** for human eye detection.

![Status](https://img.shields.io/badge/status-Production%20Ready-success.svg)
![License](https://img.shields.io/badge/license-GNU%20AGPLv3-blue.svg)
![Framework](https://img.shields.io/badge/framework-FastAPI%200.111.0-009688.svg)
![ML Framework](https://img.shields.io/badge/model-Ultralytics%20YOLOv8.4.108-FF6F00.svg)
![Python](https://img.shields.io/badge/python-3.11-3776AB.svg)
![Docker](https://img.shields.io/badge/docker%20size-454%20MB-blue.svg)

--,-

## Project Overview

**Eye Detection Service** is a production-ready, Docker-optimized microservice designed for high-concurrency human eye localization within image streams. Built on Python 3.11 using **FastAPI** and **Ultralytics YOLOv8**, it executes CPU-only deep learning inference delivering structured bounding box coordinates, confidence scores, and processing timing metrics.

Fully optimized for **Google Cloud Run**, **Docker**, and **Kubernetes** deployments, the microservice features a thread-safe singleton model manager (`EyeModelManager`) with automatic startup pre-warming, multi-layer image validation (magic header inspection, payload size limits, resolution and aspect ratio constraints), EXIF orientation normalization, and dynamic bicubic upscaling for small crops (<128px).

The container footprint has been optimized from 9.22 GB down to **~454 MB** content size (~150 MB compressed) through multi-stage Docker builds, CPU-only PyTorch wheels, and headless OpenCV integration.

---

## Features

### Detection & Inference Engine
- **Custom YOLOv8 Model**: Deep learning object detection tuned specifically for human eye localization (`models/best.pt`).
- **CPU-Only PyTorch**: Optimized for CPU inference using lightweight `torch==2.4.1+cpu` wheels without heavy CUDA dependencies.
- **Thread-Safe Model Manager**: Singleton pattern (`EyeModelManager`) ensuring single-instance model caching and startup warm-up pass.
- **Configurable Confidence Filter**: Adjustable detection confidence threshold (`CONFIDENCE_THRESHOLD=0.40`).
- **Coordinate Mapping**: Automatically maps predicted bounding boxes back to original input image dimensions.
- **Bicubic Upscaling**: Dynamic resolution scaling for small image crops (<128px minimum dimension) to preserve detection accuracy.
- **Dynamic Model Versioning**: Resilient model version reporting on `/health` endpoint with non-blocking fallback (`"unknown"`).

### Image Validation & Security
- **Magic Header Verification**: Binary byte inspection confirming `JPEG`, `PNG`, `WEBP`, or `BMP` formats to prevent extension spoofing.
- **File Size Safeguards**: Enforces payload file size boundaries (`MAX_IMAGE_SIZE_MB=10.0`).
- **Dimension & Aspect Ratio Limits**: Enforces minimum (`8px`) and maximum (`4096px`) resolution bounds and aspect ratios (`0.2` to `5.0`).
- **Decompression Bomb Defense**: Configures Pillow `MAX_IMAGE_PIXELS` limit (64M pixels) to neutralize Denial-of-Service attacks.
- **EXIF Auto-Correction**: Automatic image orientation adjustment via `PIL.ImageOps.exif_transpose`.
- **Headless OpenCV**: Uses `opencv-python-headless` for zero GUI dependency overhead.

### API & Security Headers
- **Asynchronous FastAPI Engine**: High-concurrency REST endpoints served by Uvicorn ASGI on **Port 8001**.
- **Production Security Headers**: Middleware injecting `X-Content-Type-Options: nosniff`, `Cache-Control: no-store`, and `Pragma: no-cache`.
- **Health & Readiness Probe**: Endpoint (`GET /health`) returning runtime diagnostics, framework versions, and model readiness status.
- **Eye Detection Endpoint**: Endpoint (`POST /detect-eye`) processing `multipart/form-data` uploads.
- **Tracing & Telemetry**: Middleware auto-injecting `X-Request-ID` and execution timing (`X-Response-Time-MS`).
- **Self-Documenting API**: Interactive OpenAPI `/docs` (Swagger UI) and `/redoc` interfaces.

### Containerization & Production Readiness
- **Multi-Stage Docker Build**: Ultra-slim runtime image (~454 MB content size) built from `python:3.11-slim-bookworm`.
- **Non-Root User Execution**: Security-hardened execution under unprivileged system user (`appuser:appgroup`, UID 10001).
- **Read-Only Filesystem Compatibility**: Configured `YOLO_CONFIG_DIR=/tmp/Ultralytics` for GCP Cloud Run read-only environments.
- **Automated Docker Healthcheck**: Native HTTP healthcheck with dynamic `${PORT}` binding and network timeout (`timeout=3s`).
- **Clean Log Hygiene**: Pydantic v2 protected namespace warnings eliminated (`model_config = ConfigDict(protected_namespaces=())`).

---

## Tech Stack

### Microservice Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11 | Primary programming runtime environment |
| FastAPI | 0.111.0 | High-performance asynchronous web framework |
| Uvicorn | 0.30.1 | Production ASGI web server |
| Ultralytics | 8.4.108 | Computer vision object detection framework |
| PyTorch (CPU) | 2.4.1+cpu | CPU-optimized deep learning tensor runtime |
| Torchvision (CPU) | 0.19.1+cpu | Computer vision utilities for PyTorch |
| OpenCV Headless | 4.10.0.84 | Image colorspace conversion and bicubic resizing |
| Pillow | 10.4.0 | Image binary decoding, EXIF transpose, and header validation |
| Pydantic | 2.7.4 | Request/response data validation and settings schemas |
| Python-Multipart | 0.0.9 | Multipart form-data parsing for image file uploads |

### Storage & Model Artifacts
| Storage Layer | Purpose |
|---------------|---------|
| Local File System | Model weights file storage (`models/best.pt`, ~6.2 MB) |
| In-Memory Stream | Transient image byte processing (`io.BytesIO`) without temporary disk writes |
| `/tmp` Directory | Read-only container cache directory (`YOLO_CONFIG_DIR=/tmp/Ultralytics`) |

---

## Architecture & System Topography

### System Integration Architecture
```
Backend Service ──────────────► Port 8080
Eye Detection Service ────────► Port 8001
Cataract Detection Service ───► Port 8002
```

- **Target Deployment Platform**: Google Cloud Run, Docker Desktop/Engine, Kubernetes (GKE).
- **Layered Design**:
  - `eye_service/api.py`: FastAPI router defining HTTP endpoints (`/health`, `/detect-eye`).
  - `eye_service/predictor.py`: Orchestrates image decoding, pre-processing, model inference, and coordinate mapping.
  - `eye_service/model_loader.py`: Thread-safe `EyeModelManager` singleton providing pre-warming pass execution.
  - `eye_service/utils.py`: Security validation (magic header check, file size limits, pixel bomb caps) and upscaling.
  - `eye_service/config.py`: Environment configuration management using Pydantic Settings.
- **Client ↔ Server Communication**: RESTful JSON responses over HTTP (`multipart/form-data` uploads for `/detect-eye`).

---

## Project Structure

```
Eye-Detection/
├── .dockerignore          # Docker build exclusion rules
├── .env                   # Environment variable runtime overrides (PORT=8001)
├── .env.example           # Environment variable configuration template
├── .gitignore              # Git version control ignore rules
├── Dockerfile             # Multi-stage production Docker configuration (CMD, Port 8001)
├── LICENSE                # GNU Affero General Public License v3.0 text
├── README.md              # Project documentation
├── requirements.txt       # Python dependency specifications (Ultralytics 8.4.108)
├── eye_service/           # Core microservice package
│   ├── __init__.py        # Package initialization and version metadata (1.0.0)
│   ├── api.py             # FastAPI router endpoints (/health, /detect-eye)
│   ├── config.py          # Pydantic Settings management & environment parsing
│   ├── main.py            # FastAPI application factory, middleware, and security headers
│   ├── model_loader.py    # Thread-safe singleton model manager for YOLOv8
│   ├── predictor.py       # Eye detection prediction pipeline & coordinate mapping
│   ├── schemas.py         # Pydantic schemas (BoundingBox, EyeDetectionResponse, etc.)
│   └── utils.py           # Image magic header check, security validation & preprocessing
├── models/                # Machine learning model storage
│   └── best.pt            # Custom-trained YOLOv8 eye detection model weights (~6.2 MB)
└── tests/                 # Automated test suite
    └── test_api.py        # Pytest suite for FastAPI endpoints and validation
```

---

## API Reference

### Microservice Endpoints — Port `8001`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health & readiness probe returning diagnostic state |
| `GET` | `/docs` | Interactive OpenAPI Swagger UI documentation |
| `POST` | `/detect-eye` | Detect human eye location in uploaded image (`multipart/form-data`) |

---

### Endpoint Details

#### 1. Health & Readiness Probe
`GET /health`

**Response (`200 OK` when healthy):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/best.pt",
  "framework": "Ultralytics YOLOv8",
  "readiness": true,
  "device": "cpu",
  "version": "8.4.108"
}
```

**Security Headers Returned:**
```text
X-Content-Type-Options: nosniff
Cache-Control: no-store
Pragma: no-cache
X-Request-ID: 49ec984c-27f9-4345-ad1f-e18a0656d878
X-Response-Time-MS: 2.03
```

#### 2. Detect Eye in Image
`POST /detect-eye`

**Request Header:** `Content-Type: multipart/form-data`
**Body Parameter:** `file` (UploadFile containing `JPEG`, `PNG`, `WEBP`, or `BMP` image)

**Response (`200 OK` — Eye Detected):**
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
  "processingTime": 225
}
```

**Response (`200 OK` — No Eye Detected):**
```json
{
  "success": true,
  "eyeDetected": false,
  "confidence": 0.0,
  "boundingBox": null,
  "processingTime": 225
}
```

---

## Environment Variables

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `HOST` | `0.0.0.0` | API server network bind address |
| `PORT` | `8001` | API server network bind port |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MODEL_PATH` | `models/best.pt` | File path to custom YOLOv8 model weights |
| `CONFIDENCE_THRESHOLD` | `0.40` | Detection confidence threshold (0.0 to 1.0) |
| `IMAGE_SIZE` | `640` | Inference input square resolution in pixels |
| `DEVICE` | `cpu` | Execution hardware target (`cpu` or `cuda`) |
| `YOLO_CONFIG_DIR` | `/tmp/Ultralytics` | Config directory for read-only filesystem support |
| `MAX_IMAGE_SIZE_MB` | `10.0` | Maximum allowed upload file size in megabytes |
| `MIN_IMAGE_DIM` | `8` | Minimum allowed width/height in pixels |
| `MAX_IMAGE_DIM` | `4096` | Maximum allowed width/height in pixels |
| `MIN_ASPECT_RATIO` | `0.2` | Minimum allowed image aspect ratio |
| `MAX_ASPECT_RATIO` | `5.0` | Maximum allowed image aspect ratio |

---

## Installation & Local Execution

### Prerequisites
- **Python 3.11+**
- **Docker Engine / Desktop** (Recommended for containerized execution)

### 1. Clone Repository
```bash
git clone https://github.com/sameer2210/Eye-Detection.git
cd Eye-Detection
```

### 2. Local Python Environment (Without Docker)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CPU PyTorch and dependencies
pip install torch==2.4.1+cpu torchvision==0.19.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Start service on port 8001
uvicorn eye_service.main:app --host 0.0.0.0 --port 8001
```

### 3. Docker Container Execution (Recommended)
```bash
# Build multi-stage production image
docker build -t eye-detection-service:prod .

# Run production container on port 8001
docker run -d \
  --name eye-service-prod \
  -p 8001:8001 \
  eye-detection-service:prod
```

### 4. Verify Local Container Deployment
```bash
# Test Health Endpoint
curl -f http://localhost:8001/health

# Open Interactive Swagger Documentation
# Navigate browser to: http://localhost:8001/docs
```

---

## Docker & Containerization Details

The service uses a two-stage production `Dockerfile`:

1. **Builder Stage (`python:3.11-slim-bookworm`)**:
   - Installs PyTorch CPU wheels (`torch==2.4.1+cpu`) in a separate, cached layer.
   - Installs requirements and forces single `opencv-python-headless` installation.
   - Cleans Python bytecode (`.pyc`, `__pycache__`) and test files.
2. **Runner Stage (`python:3.11-slim-bookworm`)**:
   - Copies pre-built virtual environment from builder stage.
   - Installs minimal system libraries (`libgl1`, `libglib2.0-0`).
   - Runs under non-root system user `appuser:appgroup` (UID `10001`).
   - Configures `YOLO_CONFIG_DIR=/tmp/Ultralytics` for read-only root filesystems.
   - Uses `CMD ["uvicorn", "eye_service.main:app", "--host", "0.0.0.0", "--port", "8001"]`.
   - Automated `HEALTHCHECK` command with dynamic `${PORT}` evaluation and 3-second network timeout.

---

## Performance Benchmarks & Measurements

Empirical metrics collected from production container execution:

| Metric | Value / Measurement |
|:---|:---|
| **Docker Uncompressed Content Size** | **~454 MB** (Reduced from 3.15 GB) |
| **Docker Compressed Layer Size** | **~150 MB** (Reduced from 1.2 GB) |
| **Health Check Latency (`GET /health`)** | **~2.03 ms** |
| **CPU Warm Inference Latency (`POST /detect-eye`)** | **~225 ms** |
| **Model Load & Startup Warmup Time** | **~6.7 seconds** |
| **Docker Healthcheck Status** | **`healthy`** |

---

## Production Optimizations Summary

1. **Ultralytics Version Alignment**: Upgraded `ultralytics` to `8.4.108` to match model checkpoint version, natively resolving PyTorch 2.6+ deserialization without unsafe global monkey patching.
2. **CPU PyTorch & CUDA Removal**: Replaced GPU PyTorch dependencies with CPU-only wheels (`2.4.1+cpu`), eliminating 4.59 GB of NVIDIA CUDA 13 libraries and Triton compilers.
3. **Headless OpenCV Integration**: Standardized on `opencv-python-headless` to eliminate unneeded GUI X11 dependencies.
4. **Read-Only Filesystem Hardening**: Implemented `YOLO_CONFIG_DIR=/tmp/Ultralytics` to allow seamless deployment on GCP Cloud Run.
5. **Clean Log Hygiene**: Suppressed Pydantic v2 protected namespace warnings across config and response schemas.

---

## Automated GitHub & GCP Cloud Run Deployment

The Eye Detection Service is configured for zero-downtime, automated source-to-cloud deployment on **Google Cloud Run** using **GitHub Actions**.

### 1. One-Time Google Cloud Platform (GCP) Setup

Execute the following `gcloud` CLI commands to initialize Artifact Registry and IAM permissions:

```bash
# Set GCP Project ID
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
gcloud config set project $PROJECT_ID

# Enable required Google APIs
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  iamcredentials.googleapis.com

# Create Artifact Registry Repository for Docker images
gcloud artifacts repositories create eye-detection \
  --repository-format=docker \
  --location=$REGION \
  --description="Eye Detection Microservice Docker Repository"
```

### 2. Configure GitHub Repository Secrets

Add the following environment secrets in your GitHub repository (**Settings -> Secrets and variables -> Actions**):

| Secret Name | Description | Example / Required |
| :--- | :--- | :--- |
| `GCP_PROJECT_ID` | Your Google Cloud Project ID | `my-ml-project-12345` (Required) |
| `GCP_REGION` | Target GCP region for Artifact Registry & Cloud Run | `us-central1` (Required) |
| `ARTIFACT_REPOSITORY` | Artifact Registry Docker repository name | `eye-detection` (Required) |
| `CLOUD_RUN_SERVICE` | Google Cloud Run service name | `eye-detection-service` (Required) |
| `GCP_WIF_PROVIDER` | Workload Identity Provider resource ID (Preferred Auth) | `projects/123/locations/global/...` |
| `GCP_WIF_SERVICE_ACCOUNT` | Workload Identity Service Account Email (Preferred Auth) | `github-runner@project.iam...` |
| `GCP_SA_KEY` | *(Fallback)* Service Account JSON key string | `{"type": "service_account"...}` |

### 3. Automated CI/CD Workflow Pipeline

On every `git push` to `main`, `.github/workflows/deploy.yml` automatically:
1. Executes `pytest` unit test suite.
2. Authenticates to Google Cloud using WIF / SA Credentials.
3. Builds the multi-stage Docker container (`python:3.11-slim-bookworm`).
4. Pushes tagged image (`:latest` and `:${{ github.sha }}`) to Artifact Registry.
5. Deploys container to Google Cloud Run with dynamic `${PORT}` binding and non-root execution (`appuser:appgroup`).


---

## Author

**SAM**

[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/sameer2210)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/sameer2210/)
[![Portfolio](https://img.shields.io/badge/Portfolio-orange?logo=vercel)](https://sameerkhan-io.vercel.app/)

---

<p align="center">Built with ❤️ by <strong>SAM</strong></p>
