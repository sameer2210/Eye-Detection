FROM python:3.11-slim

LABEL org.opencontainers.image.title="eye-service" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.description="Eye Detection Service Production Runtime (CPU Optimized)"

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

# Layer 1: Install runtime C-libraries (libgl1, libglib2.0-0, libgomp1 for OpenCV; curl for healthcheck; git for CLIP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Create unprivileged application user
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

WORKDIR /app

# Layer 3: Pre-install CPU-only PyTorch and TorchVision (heaviest layer, cached independently)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Layer 4: Install Ultralytics CLIP using CPU PyTorch extra-index URL (cached independently)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu git+https://github.com/ultralytics/CLIP.git

# Layer 5: Copy project packaging configuration AND source directory for valid setuptools build
COPY pyproject.toml /app/
COPY eye_service /app/eye_service

# Layer 6: Install project package and third-party dependencies into Python site-packages
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

# Layer 7: Copy model weights (cached independently from source code)
COPY models /app/models

# Layer 8: Set unprivileged file ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "eye_service.main"]
