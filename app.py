"""
app.py — Flask application factory.

Creates and configures the Flask app, instantiates all core components,
and registers API blueprints.  No global variables — everything is stored
in ``app.extensions``.
"""

import logging
import os
import time
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

import config
from core.database import FaceDatabase
from core.detector import FaceDetector
from core.pipeline import FacePipeline
from core.recognizer import FaceRecognizer
from routes.detection import detection_bp
from routes.registration import registration_bp

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory — build and return a fully configured Flask app.

    Steps:
        1. Set up rotating file logger.
        2. Instantiate core components (DB, detector, recognizer, pipeline).
        3. Register API blueprints.
        4. Define root, health-check, and error-handler routes.

    Returns:
        Configured Flask application instance.
    """
    # .env is already loaded by config.py at import time.

    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='templates',
    )
    CORS(app)

    # 1. Logging -----------------------------------------------------------
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
    file_handler = RotatingFileHandler(
        config.LOG_PATH,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s  %(levelname)-8s  %(name)s  %(message)s'
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Also log to console during development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logger.info("Starting Face Detection & Recognition System …")

    # 2. Core components ---------------------------------------------------
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    os.makedirs('models', exist_ok=True)

    db = FaceDatabase(config.DB_PATH)
    detector = FaceDetector(
        model_path=config.YOLO_MODEL_PATH,
        conf=config.YOLO_CONFIDENCE,
        iou=config.YOLO_IOU_THRESHOLD,
    )
    recognizer = FaceRecognizer(threshold=config.RECOGNITION_THRESHOLD)
    pipeline = FacePipeline(detector, recognizer, db)

    app.extensions['face_db'] = db
    app.extensions['face_detector'] = detector
    app.extensions['face_recognizer'] = recognizer
    app.extensions['face_pipeline'] = pipeline
    app.extensions['start_time'] = time.time()

    # 3. Blueprints --------------------------------------------------------
    app.register_blueprint(detection_bp)
    app.register_blueprint(registration_bp)

    # 4. Root & health routes ----------------------------------------------

    @app.route('/')
    def index():
        """Serve the single-page frontend."""
        return send_from_directory('templates', 'index.html')

    @app.route('/health')
    def health():
        """Health-check endpoint with uptime and face count."""
        uptime: float = time.time() - app.extensions['start_time']
        stats: dict = db.get_stats()
        return jsonify({
            'status': 'healthy',
            'face_count': stats['total_faces'],
            'uptime_seconds': round(uptime, 2),
        })

    # 5. Error handlers ----------------------------------------------------

    @app.errorhandler(400)
    def bad_request(exc):
        """Handle 400 Bad Request."""
        return jsonify({'error': 'Bad request.'}), 400

    @app.errorhandler(404)
    def not_found(exc):
        """Handle 404 Not Found."""
        return jsonify({'error': 'Resource not found.'}), 404

    @app.errorhandler(500)
    def internal_error(exc):
        """Handle 500 Internal Server Error."""
        logger.exception("Unhandled 500 error: %s", exc)
        return jsonify({'error': 'Internal server error.'}), 500

    logger.info("Flask app created successfully.")
    return app
