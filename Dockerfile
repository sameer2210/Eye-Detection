FROM python:3.11-slim

# Prevent Python from writing bytecode and set unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HOST=0.0.0.0 \
    MODEL_PATH=models/yolov8s-world.pt

# Install system dependencies for OpenCV, GL, and Healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy production package metadata, models, and application code
COPY pyproject.toml /app/
COPY models /app/models
COPY eye_service /app/eye_service

# Install production dependencies
RUN pip install --no-cache-dir .

# Pre-warm model weights during build to eliminate cold-start latency
RUN python -c "from ultralytics import YOLOWorld; m = YOLOWorld('models/yolov8s-world.pt'); m.set_classes(['human eye'])"

EXPOSE 8000

# Container Healthcheck directive for Docker, Kubernetes, and Cloud orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "eye_service.main"]
