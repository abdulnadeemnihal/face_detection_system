"""
config.py — Central configuration for Face Detection & Recognition System.

All constants are defined here. No other module should define configuration values.
Loads ``.env`` automatically so every module gets the right values.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (if present) BEFORE any os.getenv calls.
_ENV_PATH: Path = Path(__file__).resolve().parent / '.env'
load_dotenv(_ENV_PATH)

# ---------------------------------------------------------------------------
# YOLOv8 Detection Settings
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH: str = os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')
YOLO_CONFIDENCE: float = float(os.getenv('YOLO_CONFIDENCE', '0.35'))
YOLO_IOU_THRESHOLD: float = float(os.getenv('YOLO_IOU_THRESHOLD', '0.45'))
MIN_FACE_SIZE: int = int(os.getenv('MIN_FACE_SIZE', '48'))

# ---------------------------------------------------------------------------
# Face Region Extraction (from person bounding box)
# ---------------------------------------------------------------------------
# Fraction of the person bbox height to use as the face region (top portion)
FACE_REGION_TOP_RATIO: float = float(
    os.getenv('FACE_REGION_TOP_RATIO', '0.35')
)
# Horizontal shrink: crop the inner N% width to exclude shoulders
FACE_REGION_WIDTH_RATIO: float = float(
    os.getenv('FACE_REGION_WIDTH_RATIO', '0.60')
)
# Proportional padding as fraction of bbox dimension (replaces fixed 10px)
FACE_PAD_RATIO: float = float(os.getenv('FACE_PAD_RATIO', '0.15'))

# ---------------------------------------------------------------------------
# Quality Validation
# ---------------------------------------------------------------------------
# Laplacian variance threshold for blur rejection (adaptive base value)
LAPLACIAN_THRESHOLD: float = float(
    os.getenv('LAPLACIAN_THRESHOLD', '50.0')
)
# Reference area (px²) for adaptive blur scaling
LAPLACIAN_REF_AREA: int = int(os.getenv('LAPLACIAN_REF_AREA', '10000'))

# ---------------------------------------------------------------------------
# Recognition Settings — LBP + HOG structural features
# ---------------------------------------------------------------------------
RECOGNITION_THRESHOLD: float = float(
    os.getenv('RECOGNITION_THRESHOLD', '0.45')
)
MAX_FACES_PER_FRAME: int = int(os.getenv('MAX_FACES_PER_FRAME', '20'))

# Face crop resize before feature extraction
FACE_RESIZE_DIM: int = int(os.getenv('FACE_RESIZE_DIM', '160'))

# CLAHE illumination normalisation
CLAHE_CLIP_LIMIT: float = float(os.getenv('CLAHE_CLIP_LIMIT', '2.0'))
CLAHE_TILE_SIZE: int = int(os.getenv('CLAHE_TILE_SIZE', '8'))

# LBP grid: divide face into NxN cells and compute per-cell histograms
LBP_GRID_SIZE: int = int(os.getenv('LBP_GRID_SIZE', '8'))
# Number of LBP neighbours / radius
LBP_RADIUS: int = int(os.getenv('LBP_RADIUS', '1'))

# Multi-sample gallery: max samples stored per person
MAX_SAMPLES_PER_PERSON: int = int(
    os.getenv('MAX_SAMPLES_PER_PERSON', '5')
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: str = os.getenv('DB_PATH', 'data/faces.db')

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH: str = os.getenv('LOG_PATH', 'logs/app.log')
LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT: int = 3

# ---------------------------------------------------------------------------
# Flask / Server
# ---------------------------------------------------------------------------
FLASK_HOST: str = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT: int = int(os.getenv('FLASK_PORT', '5000'))
MAX_IMAGE_SIZE_MB: int = int(os.getenv('MAX_IMAGE_SIZE_MB', '10'))
