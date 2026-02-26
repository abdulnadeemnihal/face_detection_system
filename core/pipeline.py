"""
core/pipeline.py — High-level orchestration of detection, recognition, and DB.

Provides a thread-safe face processing pipeline that combines the detector,
recognizer, and database into a single coherent API.  The recognizer now uses
LBP + HOG features with cosine similarity and a multi-sample gallery.
"""

import logging
import threading

import cv2
import numpy as np

from core.database import FaceDatabase
from core.detector import FaceDetector
from core.recognizer import FaceRecognizer

logger = logging.getLogger(__name__)


class FacePipeline:
    """Orchestrate face detection, recognition, and storage.

    Thread-safe feature cache is refreshed from the database whenever
    a new face is registered or on demand.  The cache now stores a
    multi-sample gallery per person.

    Attributes:
        detector: YOLOv8 person detector instance.
        recognizer: LBP+HOG recognizer instance.
        db: SQLite face database instance.
    """

    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        db: FaceDatabase,
    ) -> None:
        self.detector: FaceDetector = detector
        self.recognizer: FaceRecognizer = recognizer
        self.db: FaceDatabase = db
        self._cache_lock: threading.Lock = threading.Lock()
        self._feature_cache: list[dict] = []
        self.refresh_cache()
        logger.info(
            "FacePipeline initialised with %d cached faces.",
            len(self._feature_cache),
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def refresh_cache(self) -> None:
        """Reload all feature vectors from the database into memory.

        Thread-safe — acquires ``_cache_lock`` before writing.
        Each entry contains ``name``, ``features``, and ``samples``.
        """
        features: list[dict] = self.db.get_all_features()
        with self._cache_lock:
            self._feature_cache = features
        logger.debug("Feature cache refreshed (%d entries).", len(features))

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """Detect and recognise faces in a single frame.

        Args:
            frame: BGR image (H×W×3 uint8).

        Returns:
            List of result dicts::

                {
                    'bbox': [x1, y1, x2, y2],        # face region
                    'person_bbox': [x1, y1, x2, y2], # full person box
                    'name': str,
                    'confidence': float,
                    'similarity': float,
                }
        """
        detections: list[dict] = self.detector.detect(frame)
        results: list[dict] = []

        with self._cache_lock:
            cached: list[dict] = list(self._feature_cache)

        for det in detections:
            feat: np.ndarray = self.recognizer.get_features(
                det['face_crop']
            )
            name: str
            score: float
            name, score = self.recognizer.find_match(feat, cached)

            results.append({
                'bbox': det['bbox'],
                'person_bbox': det.get('person_bbox', det['bbox']),
                'name': name,
                'confidence': det['confidence'],
                'similarity': score,
            })

        return results

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_face(self, name: str, frame: np.ndarray) -> dict:
        """Detect the largest face in *frame* and register it under *name*.

        Each call adds a new sample to the gallery for the person.
        Up to ``MAX_SAMPLES_PER_PERSON`` samples are kept (newest wins).

        Args:
            name: Display name (already validated by the caller).
            frame: BGR image with at least one face.

        Returns:
            Dict with ``success`` (bool), ``name`` (str), ``message`` (str).
        """
        detections: list[dict] = self.detector.detect(frame)
        if not detections:
            logger.warning("No face found during registration for '%s'.", name)
            return {
                'success': False,
                'name': name,
                'message': 'No face detected in the image.',
            }

        # Pick the crop with the largest face bbox area
        best: dict = max(
            detections,
            key=lambda d: (
                (d['bbox'][2] - d['bbox'][0])
                * (d['bbox'][3] - d['bbox'][1])
            ),
        )

        crop: np.ndarray = best['face_crop']
        features: np.ndarray = self.recognizer.get_features(crop)

        # Encode crop to JPEG bytes for storage
        success_encode: bool
        buf: np.ndarray
        success_encode, buf = cv2.imencode('.jpg', crop)
        if not success_encode:
            logger.error("Failed to encode face crop for '%s'.", name)
            return {
                'success': False,
                'name': name,
                'message': 'Image encoding failed.',
            }

        img_bytes: bytes = buf.tobytes()
        self.db.store_face(name, img_bytes, features)
        self.refresh_cache()

        logger.info("Face registered for '%s'.", name)
        return {
            'success': True,
            'name': name,
            'message': 'Face registered successfully',
        }

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def annotate_frame(
        self,
        frame: np.ndarray,
        results: list[dict],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame.

        Two boxes are drawn per detection:
        - **Thin** person-level box (person_bbox).
        - **Thick** face-region box (bbox).

        Green = known, Red = unknown.

        Args:
            frame: Original BGR frame.
            results: Output of ``process_frame``.

        Returns:
            Annotated BGR frame (copy).
        """
        annotated: np.ndarray = frame.copy()
        known_count: int = 0
        unknown_count: int = 0

        for res in results:
            fx1, fy1, fx2, fy2 = res['bbox']
            px1, py1, px2, py2 = res.get('person_bbox', res['bbox'])
            name: str = res['name']
            conf: float = res['confidence']
            sim: float = res['similarity']

            if name != 'Unknown':
                colour: tuple[int, int, int] = (0, 255, 0)  # green
                known_count += 1
            else:
                colour = (0, 0, 255)  # red
                unknown_count += 1

            # Person bounding box — thin dashed-style
            cv2.rectangle(annotated, (px1, py1), (px2, py2), colour, 1)
            # Face region box — thick
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), colour, 2)

            label: str = f"{name} {conf:.0%}"
            if sim > 0:
                label += f" sim:{sim:.0%}"

            label_y: int = max(fy1 - 10, 20)
            cv2.putText(
                annotated,
                label,
                (fx1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
            )

        total: int = known_count + unknown_count
        status: str = (
            f"Persons: {total} (known: {known_count}, "
            f"unknown: {unknown_count})"
        )
        cv2.putText(
            annotated,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return annotated
