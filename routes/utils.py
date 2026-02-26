"""
routes/utils.py — Shared helpers for API route handlers.
"""

import base64

import cv2
import numpy as np

import config


def decode_image(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded JPEG string into a BGR numpy array.

    Handles optional ``data:image/...;base64,`` prefix automatically.

    Args:
        b64_string: Base64 JPEG data (may include data-URI prefix).

    Returns:
        OpenCV BGR image.

    Raises:
        ValueError: If decoding fails or payload exceeds size limit.
    """
    if ',' in b64_string:
        b64_string = b64_string.split(',', 1)[1]

    raw: bytes = base64.b64decode(b64_string)

    max_bytes: int = config.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise ValueError(
            f"Image exceeds {config.MAX_IMAGE_SIZE_MB} MB limit."
        )

    arr: np.ndarray = np.frombuffer(raw, dtype=np.uint8)
    img: np.ndarray = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image data.")
    return img
