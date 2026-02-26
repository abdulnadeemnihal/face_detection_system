"""
routes/detection.py — Detection API endpoint.

POST /detect_faces — accept a base64 JPEG, return annotated detections.
"""

import base64
import logging

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from routes.utils import decode_image

logger = logging.getLogger(__name__)

detection_bp = Blueprint('detection', __name__)


def _serialisable_results(results: list[dict]) -> list[dict]:
    """Strip non-JSON-serialisable keys (numpy arrays, etc.) from results."""
    clean: list[dict] = []
    for r in results:
        clean.append({
            'bbox': r['bbox'],
            'name': r['name'],
            'confidence': r['confidence'],
            'similarity': r['similarity'],
        })
    return clean


@detection_bp.route('/detect_faces', methods=['POST'])
def detect_faces():
    """Detect and recognise faces in a base64 image.

    Request JSON::

        {"image": "<base64 JPEG>"}

    Response 200::

        {
            "faces": [
                {"bbox": [...], "name": str, "confidence": float,
                 "similarity": float}
            ],
            "count": int,
            "frame_b64": "<annotated base64 JPEG>"
        }
    """
    data: dict = request.get_json(silent=True) or {}
    image_b64: str = data.get('image', '')

    if not image_b64:
        return jsonify({'error': 'Missing "image" field.'}), 400

    try:
        frame: np.ndarray = decode_image(image_b64)
    except (ValueError, Exception) as exc:
        logger.warning("Image decode error: %s", exc)
        return jsonify({'error': str(exc)}), 400

    try:
        pipeline = current_app.extensions['face_pipeline']
        results: list[dict] = pipeline.process_frame(frame)
        annotated: np.ndarray = pipeline.annotate_frame(frame, results)

        success: bool
        buf: np.ndarray
        success, buf = cv2.imencode('.jpg', annotated)
        frame_b64: str = ''
        if success:
            frame_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

        return jsonify({
            'faces': _serialisable_results(results),
            'count': len(results),
            'frame_b64': frame_b64,
        }), 200

    except Exception as exc:
        logger.exception("Detection pipeline error: %s", exc)
        return jsonify({'error': 'Internal detection error.'}), 500
