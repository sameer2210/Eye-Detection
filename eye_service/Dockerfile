FROM python:3.11-slim

# Prevent Python from writing bytecode and set unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV and GL libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy production application files and model weights
COPY pyproject.toml /app/
COPY eye_service /app/eye_service
COPY yolov8s-world.pt /app/

# Install production dependencies
RUN pip install --no-cache-dir .

# Pre-warm model weights during build to prevent cold-start latency
RUN python -c "from ultralytics import YOLOWorld; m = YOLOWorld('yolov8s-world.pt'); m.set_classes(['human eye'])"

EXPOSE 8000

CMD ["python", "-m", "eye_service.main"]
