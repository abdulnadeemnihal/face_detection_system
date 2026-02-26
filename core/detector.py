"""
core/detector.py — YOLOv8-based face detector.

Uses ultralytics YOLO for person detection, then extracts the face region
from the upper portion of the person bounding box.  No PyTorch imports,
no Caffe, no Haar.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detect faces by locating persons with YOLOv8, then cropping the
    face region from the upper portion of each person bounding box.

    Attributes:
        model: Loaded YOLO model instance.
        conf: Minimum confidence threshold.
        iou: IoU threshold for NMS.
    """

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_PATH,
        conf: float = config.YOLO_CONFIDENCE,
        iou: float = config.YOLO_IOU_THRESHOLD,
    ) -> None:
        """Load the YOLOv8 model and run a warm-up inference.

        Args:
            model_path: Path to the ``.pt`` weights file.
            conf: Confidence threshold (0-1).
            iou: IoU threshold for non-max suppression.
        """
        self.conf: float = conf
        self.iou: float = iou
        logger.info("Loading YOLOv8 model from %s ...", model_path)
        self.model: YOLO = YOLO(model_path)
        self.warmup()
        logger.info("FaceDetector ready (conf=%.2f, iou=%.2f).", conf, iou)

    def warmup(self) -> None:
        """Run one dummy inference on a blank frame to initialise the model."""
        blank: np.ndarray = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(
            blank,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        logger.debug("Model warm-up complete.")

    # ------------------------------------------------------------------
    # Face region extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_face_region(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> tuple[np.ndarray, list[int]]:
        """Extract the face region from the upper portion of a person bbox.

        The face is assumed to occupy the top ``FACE_REGION_TOP_RATIO`` of
        the person height, centred within ``FACE_REGION_WIDTH_RATIO`` of
        the person width.  Proportional padding is applied.

        Args:
            frame: Full BGR frame.
            x1, y1, x2, y2: Person bounding box coordinates.

        Returns:
            Tuple of (face_crop, [fx1, fy1, fx2, fy2]) where the second
            element is the face bbox in frame coordinates.
        """
        h_frame, w_frame = frame.shape[:2]
        bw: int = x2 - x1
        bh: int = y2 - y1

        # Vertical: top portion
        face_h: int = max(1, int(bh * config.FACE_REGION_TOP_RATIO))
        fy1: int = y1
        fy2: int = y1 + face_h

        # Horizontal: centre portion (exclude shoulders)
        inset: int = int(bw * (1.0 - config.FACE_REGION_WIDTH_RATIO) / 2.0)
        fx1: int = x1 + inset
        fx2: int = x2 - inset

        # Proportional padding
        pad_x: int = int(bw * config.FACE_PAD_RATIO)
        pad_y: int = int(face_h * config.FACE_PAD_RATIO)
        fx1 = max(0, fx1 - pad_x)
        fy1 = max(0, fy1 - pad_y)
        fx2 = min(w_frame, fx2 + pad_x)
        fy2 = min(h_frame, fy2 + pad_y)

        face_crop: np.ndarray = frame[fy1:fy2, fx1:fx2].copy()
        return face_crop, [fx1, fy1, fx2, fy2]

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run person detection, then extract face regions.

        Args:
            frame: OpenCV BGR image (H x W x 3 uint8).

        Returns:
            List of detection dicts, each containing:
            - ``bbox``: [x1, y1, x2, y2] (int) — face region coords
            - ``person_bbox``: [x1, y1, x2, y2] — full person coords
            - ``confidence``: float
            - ``face_crop``: np.ndarray (BGR face crop)
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to detect().")
            return []

        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=[0],  # class 0 = person (standard COCO model)
            verbose=False,
        )

        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
                confidence: float = float(box.conf[0])

                face_crop, face_bbox = self._extract_face_region(
                    frame, px1, py1, px2, py2
                )

                if not self._validate_face(face_crop):
                    continue

                detections.append({
                    'bbox': face_bbox,
                    'person_bbox': [px1, py1, px2, py2],
                    'confidence': round(confidence, 4),
                    'face_crop': face_crop,
                })

                if len(detections) >= config.MAX_FACES_PER_FRAME:
                    break

        logger.debug("Detected %d valid face(s).", len(detections))
        return detections

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_face(crop: Optional[np.ndarray]) -> bool:
        """Check that a face crop meets quality thresholds.

        Criteria:
        - Minimum size ``MIN_FACE_SIZE`` x ``MIN_FACE_SIZE`` pixels.
        - Aspect ratio between 0.3 and 3.0 (relaxed for faces).
        - Adaptive Laplacian blur rejection: threshold scales with the
          crop area relative to ``LAPLACIAN_REF_AREA``.

        Args:
            crop: BGR face crop image.

        Returns:
            True if the crop passes all checks.
        """
        if crop is None or crop.size == 0:
            return False

        ch, cw = crop.shape[:2]
        if ch < config.MIN_FACE_SIZE or cw < config.MIN_FACE_SIZE:
            return False

        aspect: float = cw / max(ch, 1)
        if aspect < 0.3 or aspect > 3.0:
            return False

        # Adaptive Laplacian threshold
        area: int = ch * cw
        scale: float = area / max(config.LAPLACIAN_REF_AREA, 1)
        adaptive_thresh: float = config.LAPLACIAN_THRESHOLD * max(
            0.3, min(scale, 3.0)
        )

        grey: np.ndarray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var: float = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        if lap_var < adaptive_thresh:
            return False

        return True
