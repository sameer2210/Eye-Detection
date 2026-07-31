# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Main entrypoint for Eye Detection Service running FastAPI application with Uvicorn."""

import os
import sys
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from eye_service.config import get_settings
from eye_service.model_loader import EyeModelManager
from eye_service.api import router

settings = get_settings()

# Configure logging format and level
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("eye_service.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle context manager performing model pre-loading and cleanup."""
    logger.info("Initializing Eye Detection Service...")
    logger.info(
        "Configured Settings: model_path=%s, threshold=%.2f, image_size=%d, device=%s, port=%d",
        settings.model_path,
        settings.confidence_threshold,
        settings.image_size,
        settings.device,
        settings.port,
    )

    # Initialize and pre-warm model on startup
    try:
        manager = EyeModelManager.get_instance()
        manager.load_model()
        logger.info("Custom YOLOv8 Eye Detector ready to serve requests.")
    except Exception as exc:
        logger.critical("Failed to load model during startup: %s", exc, exc_info=True)
        raise exc

    yield

    logger.info("Shutting down Eye Detection Service.")


app = FastAPI(
    title="Eye Detection API",
    description="Production-ready Eye Detection Service powered by custom-trained Ultralytics YOLOv8 model.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configurable CORS middleware adhering to W3C specification
cors_origins_str = os.getenv("CORS_ORIGINS", "*")
cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]

# W3C CORS Compliance: Access-Control-Allow-Origin: * cannot be used with Access-Control-Allow-Credentials: true
allow_credentials = True
if "*" in cors_origins:
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers_and_timing(request: Request, call_next):
    """Middleware attaching security headers, unique request IDs, and tracking request duration."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Tracing and Performance Headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-MS"] = f"{elapsed_ms:.2f}"

    # Production Security Headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    logger.info(
        "[%s] %s %s -> status=%d (%.2f ms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )

    return response


# Include API router
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("eye_service.main:app", host=settings.host, port=settings.port, reload=False)
