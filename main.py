"""
main.py — Desktop OpenCV mode (standalone, no Flask).

Opens the webcam and runs real-time face detection + recognition via the
pipeline.  Hotkeys:
    a — register a face (prompts for name in the console)
    d — toggle debug overlay (confidence / similarity values)
    q — quit
"""

import logging
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

import config
from core.database import FaceDatabase
from core.detector import FaceDetector
from core.pipeline import FacePipeline
from core.recognizer import FaceRecognizer

# ---------------------------------------------------------------------------
# Logging (file + console)
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Camera index — override via CAMERA_INDEX env var (default 0)
CAMERA_INDEX: int = int(os.getenv('CAMERA_INDEX', '0'))
# Max consecutive frame-read failures before giving up
MAX_FRAME_FAILURES: int = 30


def build_pipeline() -> FacePipeline:
    """Construct the full detection -> recognition -> DB pipeline."""
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    os.makedirs('models', exist_ok=True)

    db = FaceDatabase(config.DB_PATH)
    detector = FaceDetector(
        model_path=config.YOLO_MODEL_PATH,
        conf=config.YOLO_CONFIDENCE,
        iou=config.YOLO_IOU_THRESHOLD,
    )
    recognizer = FaceRecognizer(threshold=config.RECOGNITION_THRESHOLD)
    return FacePipeline(detector, recognizer, db)


def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """Open the webcam with a retry.  Exits on failure."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Cannot open webcam (device %d).", index)
        print(f"ERROR: Cannot open webcam (device {index}).")
        sys.exit(1)
    # Read one test frame to confirm the camera actually works
    ok, _ = cap.read()
    if not ok:
        logger.warning("Camera opened but first read failed — retrying …")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            logger.error("Retry failed — cannot open webcam.")
            print("ERROR: Webcam retry failed.")
            sys.exit(1)
    return cap


def draw_debug_overlay(
    frame: np.ndarray,
    results: list[dict],
) -> np.ndarray:
    """Draw extended debug info on the frame (confidence + similarity).

    Args:
        frame: Annotated frame (already has bboxes from pipeline).
        results: Detection result dicts.

    Returns:
        Frame with additional debug text.
    """
    for idx, res in enumerate(results):
        x1, y1, x2, y2 = res['bbox']
        debug_text: str = (
            f"conf={res['confidence']:.2f}  "
            f"sim={res['similarity']:.2f}"
        )
        y_pos: int = y2 + 20 + idx * 0
        cv2.putText(
            frame,
            debug_text,
            (x1, min(y_pos, frame.shape[0] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
        )
    return frame


def main() -> None:
    """Entry point — webcam loop with detection, recognition, and hotkeys."""
    print("=== Face Detection & Recognition System ===")
    print("Hotkeys:  [a] register face  |  [d] toggle debug  |  [q] quit")
    print()

    pipeline: FacePipeline = build_pipeline()
    cap: cv2.VideoCapture = open_camera()

    debug_mode: bool = False
    fps_deque: deque = deque(maxlen=30)
    prev_time: float = time.time()
    fail_count: int = 0

    try:
        while True:
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count >= MAX_FRAME_FAILURES:
                    logger.error(
                        "Lost webcam after %d consecutive read failures.",
                        fail_count,
                    )
                    print("ERROR: Webcam connection lost.")
                    break
                time.sleep(0.05)
                continue
            fail_count = 0

            # --- FPS calculation ------------------------------------------
            now: float = time.time()
            dt: float = now - prev_time
            prev_time = now
            if dt > 0:
                fps_deque.append(1.0 / dt)
            avg_fps: float = (
                sum(fps_deque) / len(fps_deque) if fps_deque else 0.0
            )

            # --- Detection + recognition ---------------------------------
            results: list[dict] = pipeline.process_frame(frame)
            display: np.ndarray = pipeline.annotate_frame(frame, results)

            if debug_mode:
                display = draw_debug_overlay(display, results)

            # --- FPS counter (top-left, below person count) ---------------
            cv2.putText(
                display,
                f"FPS: {avg_fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow('Face Detection & Recognition', display)

            # --- Hotkeys --------------------------------------------------
            key: int = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting …")
                break

            elif key == ord('a'):
                name: str = input("Enter name to register: ").strip()
                if name:
                    result: dict = pipeline.register_face(name, frame)
                    if result['success']:
                        print(f"✓ Registered '{result['name']}'.")
                    else:
                        print(f"✗ Registration failed: {result['message']}")
                else:
                    print("Registration cancelled (empty name).")

            elif key == ord('d'):
                debug_mode = not debug_mode
                state: str = "ON" if debug_mode else "OFF"
                print(f"Debug overlay: {state}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam released. Exiting.")
        print("Camera released. Goodbye.")


if __name__ == '__main__':
    main()
