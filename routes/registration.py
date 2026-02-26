"""
routes/registration.py — Face registration, listing, and deletion endpoints.

POST   /register_face  — register a new face
GET    /faces           — list all registered faces
DELETE /faces/<name>    — remove a registered face
"""

import logging
import re

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from routes.utils import decode_image

logger = logging.getLogger(__name__)

registration_bp = Blueprint('registration', __name__)

_NAME_PATTERN: re.Pattern = re.compile(r'^[A-Za-z0-9 ]+$')


def _validate_name(raw: str) -> str:
    """Validate and sanitise a person name.

    Rules: 2–50 characters, alphanumeric + spaces, trimmed.

    Args:
        raw: Raw name string from the request.

    Returns:
        Cleaned name.

    Raises:
        ValueError: If the name is invalid.
    """
    name: str = raw.strip()
    if len(name) < 2 or len(name) > 50:
        raise ValueError("Name must be between 2 and 50 characters.")
    if not _NAME_PATTERN.match(name):
        raise ValueError(
            "Name may only contain letters, digits, and spaces."
        )
    return name


# -----------------------------------------------------------------------
# POST /register_face
# -----------------------------------------------------------------------

@registration_bp.route('/register_face', methods=['POST'])
def register_face():
    """Register a face from a base64 image.

    Request JSON::

        {"image": "<base64 JPEG>", "name": "<string>"}

    Response 201::

        {"success": true, "name": str, "message": "Face registered successfully"}
    """
    data: dict = request.get_json(silent=True) or {}
    raw_name: str = data.get('name', '')
    image_b64: str = data.get('image', '')

    # --- validate name ---------------------------------------------------
    try:
        name: str = _validate_name(raw_name)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    # --- validate image --------------------------------------------------
    if not image_b64:
        return jsonify({'error': 'Missing "image" field.'}), 400

    try:
        frame: np.ndarray = decode_image(image_b64)
    except (ValueError, Exception) as exc:
        logger.warning("Image decode error during registration: %s", exc)
        return jsonify({'error': str(exc)}), 400

    # --- register --------------------------------------------------------
    try:
        pipeline = current_app.extensions['face_pipeline']
        result: dict = pipeline.register_face(name, frame)

        if not result['success']:
            return jsonify({'error': result['message']}), 400

        return jsonify({
            'success': True,
            'name': result['name'],
            'message': result['message'],
        }), 201

    except Exception as exc:
        logger.exception("Registration error: %s", exc)
        return jsonify({'error': 'Internal registration error.'}), 500


# -----------------------------------------------------------------------
# GET /faces
# -----------------------------------------------------------------------

@registration_bp.route('/faces', methods=['GET'])
def list_faces():
    """Return all registered faces (metadata only).

    Response 200::

        {
            "faces": [
                {"name": str, "registered_at": str,
                 "recognition_count": int, "last_seen_at": str|null}
            ],
            "total": int
        }
    """
    try:
        db = current_app.extensions['face_db']
        faces: list[dict] = db.get_face_list()
        return jsonify({'faces': faces, 'total': len(faces)}), 200
    except Exception as exc:
        logger.exception("Error listing faces: %s", exc)
        return jsonify({'error': 'Internal server error.'}), 500


# -----------------------------------------------------------------------
# DELETE /faces/<name>
# -----------------------------------------------------------------------

@registration_bp.route('/faces/<name>', methods=['DELETE'])
def delete_face(name: str):
    """Delete a registered face by name.

    Response 200::

        {"success": true, "name": str}

    Error 404::

        {"error": "Face not found"}
    """
    try:
        db = current_app.extensions['face_db']
        pipeline = current_app.extensions['face_pipeline']

        deleted: bool = db.delete_face(name)
        if not deleted:
            return jsonify({'error': 'Face not found'}), 404

        pipeline.refresh_cache()
        return jsonify({'success': True, 'name': name}), 200

    except Exception as exc:
        logger.exception("Error deleting face '%s': %s", name, exc)
        return jsonify({'error': 'Internal server error.'}), 500
