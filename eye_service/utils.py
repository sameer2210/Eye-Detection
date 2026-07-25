# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Image validation, security inspection, and preprocessing utilities for Eye Detection Service."""

import io
import cv2
import numpy as np
from PIL import Image, ImageOps

from eye_service.config import get_settings
from eye_service.schemas import BoundingBox


class ImageValidationError(ValueError):
    """Custom exception raised when uploaded image fails security or format validation."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def validate_image_bytes(image_bytes: bytes) -> str:
    """Perform security and format validation on raw image binary stream.

    Args:
        image_bytes (bytes): Raw bytes of uploaded file.

    Returns:
        str: Detected image format ('jpeg', 'png', 'webp', 'bmp').

    Raises:
        ImageValidationError: If file is empty, oversized, unsupported format, or corrupt.
    """
    settings = get_settings()

    if not image_bytes:
        raise ImageValidationError("Uploaded image file is empty.", status_code=400)

    # Check file size limit
    max_bytes = int(settings.max_image_size_mb * 1024 * 1024)
    if len(image_bytes) > max_bytes:
        raise ImageValidationError(
            f"Image file size ({len(image_bytes) / (1024*1024):.2f}MB) exceeds maximum allowed size ({settings.max_image_size_mb:.1f}MB).",
            status_code=413,
        )

    # Magic header verification (anti-spoofing / malicious payload check)
    detected_format = None
    if image_bytes.startswith(b"\xff\xd8\xff"):
        detected_format = "jpeg"
    elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        detected_format = "png"
    elif image_bytes.startswith(b"RIFF") and len(image_bytes) >= 12 and image_bytes[8:12] == b"WEBP":
        detected_format = "webp"
    elif image_bytes.startswith(b"BM"):
        detected_format = "bmp"

    if not detected_format:
        raise ImageValidationError(
            "Unsupported or invalid image file format. Supported formats: JPEG, PNG, WEBP, BMP.",
            status_code=415,
        )

    return detected_format


def load_and_preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, float, float]:
    """Validate image dimensions/integrity and load as BGR NumPy array for Ultralytics inference engine.

    Performs bicubic upscaling for small eye crops (<128px) while preserving aspect ratio and
    returning scale factors to map predicted bounding box coordinates back to original image space.

    Args:
        image_bytes (bytes): Raw bytes of uploaded file.

    Returns:
        tuple[np.ndarray, float, float]: (Preprocessed BGR NumPy array, scale_x, scale_y).

    Raises:
        ImageValidationError: If image is corrupted, or fails dimension/aspect ratio rules.
    """
    validate_image_bytes(image_bytes)
    settings = get_settings()

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()  # Verify integrity
    except Exception as e:
        raise ImageValidationError(f"Corrupted or malformed image data: {str(e)}", status_code=400)

    # Re-open after verify() (PIL requires re-opening after verify)
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception as e:
        raise ImageValidationError(f"Failed to decode image: {str(e)}", status_code=400)

    # Convert to RGB (handles RGBA, Palette, Grayscale, etc.)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    width, height = pil_img.size

    # Check minimum dimensions (minimum 8px to reject corrupted/empty 1x1 images)
    if width < settings.min_image_dim or height < settings.min_image_dim:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) are smaller than minimum allowed ({settings.min_image_dim}x{settings.min_image_dim}).",
            status_code=400,
        )

    # Check maximum dimensions
    if width > settings.max_image_dim or height > settings.max_image_dim:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed ({settings.max_image_dim}x{settings.max_image_dim}).",
            status_code=400,
        )

    # Check aspect ratio
    aspect_ratio = width / float(height)
    if aspect_ratio < settings.min_aspect_ratio or aspect_ratio > settings.max_aspect_ratio:
        raise ImageValidationError(
            f"Invalid image aspect ratio ({aspect_ratio:.2f}). Must be between {settings.min_aspect_ratio} and {settings.max_aspect_ratio}.",
            status_code=400,
        )

    # Convert PIL RGB Image to BGR NumPy array
    rgb_array = np.array(pil_img)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # Bicubic upscaling for small crop images (<128px) to enhance feature map resolution
    target_min_dim = 128
    min_dim = min(width, height)
    scale_x, scale_y = 1.0, 1.0

    if min_dim < target_min_dim:
        scale = target_min_dim / float(min_dim)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        bgr_array = cv2.resize(bgr_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        scale_x = new_w / float(width)
        scale_y = new_h / float(height)

    return bgr_array, scale_x, scale_y


def format_bounding_box(
    xyxy: list[float] | np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> BoundingBox:
    """Convert xyxy bounding box array into BoundingBox object mapped back to original image space.

    Args:
        xyxy (list[float] | np.ndarray): [xmin, ymin, xmax, ymax] coordinates in preprocessed image pixels.
        scale_x (float): Horizontal scale factor applied during preprocessing.
        scale_y (float): Vertical scale factor applied during preprocessing.

    Returns:
        BoundingBox: Formatted bounding box schema containing x, y, width, height in original image coordinates.
    """
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    orig_x1 = max(0, int(round(x1 / scale_x)))
    orig_y1 = max(0, int(round(y1 / scale_y)))
    orig_x2 = int(round(x2 / scale_x))
    orig_y2 = int(round(y2 / scale_y))

    width = max(1, orig_x2 - orig_x1)
    height = max(1, orig_y2 - orig_y1)
    return BoundingBox(x=orig_x1, y=orig_y1, width=width, height=height)
