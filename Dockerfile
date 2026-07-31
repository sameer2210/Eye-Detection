# Multi-stage enterprise production Dockerfile for Eye Detection Service
# Stage 1: Build virtual environment and install dependencies
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Step 1: Install CPU-only PyTorch (Cached Layer)
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu \
    torchvision==0.19.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install application requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 3: Ensure opencv-python-headless is the active cv2 implementation
RUN pip install --no-cache-dir --no-deps --force-reinstall opencv-python-headless==4.10.0.84

# Step 4: Purge Python bytecode, tests, and unneeded caches from virtual environment
RUN find /opt/venv -type f -name '*.pyc' -delete && \
    find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true


# Stage 2: Minimal immutable runtime image
FROM python:3.11-slim-bookworm AS runner

WORKDIR /app

# Production environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8001 \
    MODEL_PATH=models/best.pt \
    IMAGE_SIZE=640 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# Install minimal runtime system libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security (UID 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser && \
    chown -R appuser:appgroup /app

# Copy model weights and application source code
COPY --chown=appuser:appgroup models /app/models
COPY --chown=appuser:appgroup eye_service /app/eye_service

USER appuser:appgroup

EXPOSE 8001

# Dynamic Health check using ${PORT} and 3s network timeout
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request, os; port = os.getenv('PORT', '8001'); urllib.request.urlopen(f'http://localhost:{port}/health', timeout=3)" || exit 1

# Production CMD instruction allowing flexible override
CMD ["uvicorn", "eye_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
