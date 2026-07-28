# Multi-stage production Dockerfile for Eye Detection Service
# Stage 1: Build virtual environment and install dependencies
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for building C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Minimal runtime image
FROM python:3.11-slim-bookworm AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH=models/best.pt \
    IMAGE_SIZE=640

# Install runtime system libraries required by OpenCV (headless) and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser && \
    chown -R appuser:appgroup /app

# Copy application source code and models
COPY --chown=appuser:appgroup eye_service /app/eye_service
COPY --chown=appuser:appgroup models /app/models

USER appuser:appgroup

EXPOSE 8000

# Health check using FastAPI readiness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn", "eye_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
