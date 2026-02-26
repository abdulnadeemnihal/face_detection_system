"""
routes — Flask blueprint package for API endpoints.
"""

from routes.detection import detection_bp
from routes.registration import registration_bp
from routes.utils import decode_image

__all__: list[str] = ['detection_bp', 'registration_bp', 'decode_image']
