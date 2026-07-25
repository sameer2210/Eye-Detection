# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FastAPI router definition with production endpoints for Eye Detection Service."""

import logging
import uuid
from fastapi import APIRouter, File, UploadFile, Request, status
from fastapi.responses import JSONResponse

import ultralytics
from eye_service.config import get_settings
from eye_service.model_loader import EyeModelManager
from eye_service.predictor import get_predictor
from eye_service.schemas import EyeDetectionResponse, HealthCheckResponse, ErrorResponse
from eye_service.utils import ImageValidationError

logger = logging.getLogger("eye_service.api")

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check whether the Eye Detection service and YOLO-World model are healthy and ready for inference.",
)
async def health_check():
    """Health check endpoint returning service status, model load status, and device info."""
    settings = get_settings()
    manager = EyeModelManager.get_instance()

    return HealthCheckResponse(
        status="ok",
        model_loaded=manager.is_loaded,
        device=settings.device,
        version=ultralytics.__version__,
    )


@router.post(
    "/detect-eye",
    response_model=EyeDetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or validation error"},
        413: {"model": ErrorResponse, "description": "Uploaded image exceeds size limit"},
        415: {"model": ErrorResponse, "description": "Unsupported image format"},
        500: {"model": ErrorResponse, "description": "Internal server error during detection"},
    },
    summary="Detect eye in image",
    description="Zero-shot prediction endpoint determining whether an uploaded image contains a human eye.",
)
def detect_eye(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, BMP)"),
):
    """Synchronous endpoint executing zero-shot eye detection off the asyncio event loop.

    Args:
        request (Request): FastAPI request object for extracting trace IDs.
        file (UploadFile): Uploaded image multipart file.

    Returns:
        EyeDetectionResponse: Bounding box, confidence score, eyeDetected flag, and processing time.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    try:
        if not file or not file.file:
            raise ImageValidationError("No file uploaded. Please upload a valid image file.", status_code=400)

        # Read file binary contents
        contents = file.file.read()

        # Run prediction pipeline
        predictor = get_predictor()
        response = predictor.predict(contents)

        return response

    except ImageValidationError as ve:
        logger.warning("[%s] Image validation failed: %s", request_id, ve.message)
        return JSONResponse(
            status_code=ve.status_code,
            content=ErrorResponse(
                success=False,
                error="IMAGE_VALIDATION_ERROR",
                message=ve.message,
                request_id=request_id,
            ).model_dump(),
        )

    except Exception as exc:
        logger.exception("[%s] Unexpected error during eye detection: %s", request_id, str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                success=False,
                error="INTERNAL_SERVER_ERROR",
                message=f"An unexpected error occurred: {str(exc)}",
                request_id=request_id,
            ).model_dump(),
        )
