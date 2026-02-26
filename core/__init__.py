"""
core — Core modules for face detection, recognition, database, and pipeline.
"""

from core.database import FaceDatabase
from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.pipeline import FacePipeline

__all__: list[str] = [
    'FaceDatabase',
    'FaceDetector',
    'FaceRecognizer',
    'FacePipeline',
]
