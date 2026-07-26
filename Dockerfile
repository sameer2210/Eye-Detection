FROM python:3.11-slim

LABEL org.opencontainers.image.title="eye-service" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.description="Eye Detection Service Production Runtime"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HOST=0.0.0.0 \
    MODEL_PATH=models/yolov8s-world.pt \
    YOLO_CONFIG_DIR=/tmp \
    TORCH_HOME=/tmp/torch \
    MPLCONFIGDIR=/tmp/matplotlib

# Install runtime system libraries (libgl1, libglib2.0-0, libgomp1 for OpenCV; curl for healthcheck; git for YOLOWorld CLIP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

WORKDIR /app

# Single unified dependency installation pass using PyTorch CPU extra-index-url
COPY pyproject.toml /app/
COPY eye_service /app/eye_service
RUN pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

# Copy model weights and set file permissions
COPY models /app/models
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "eye_service.main"]
