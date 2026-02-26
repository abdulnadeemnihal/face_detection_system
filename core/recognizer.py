"""
core/recognizer.py — Structural face recognizer using OpenCV only.

Extracts CLAHE-normalised LBP (Local Binary Pattern) and HOG (Histogram of
Oriented Gradients) features, then matches via cosine similarity.
Zero torch / neural-network dependencies.
"""

import logging

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Recognise faces using structural features (LBP + HOG).

    Pipeline per crop:
        1. Resize to ``FACE_RESIZE_DIM x FACE_RESIZE_DIM``.
        2. CLAHE illumination normalisation on the L channel (LAB).
        3. Compute spatial LBP histogram (grayscale texture).
        4. Compute HOG descriptor (edge orientations).
        5. Concatenate, L2-normalise → single feature vector.

    Matching uses cosine similarity.

    Attributes:
        threshold: Minimum cosine similarity to consider a match.
        resize_dim: Target (width, height) for face crops.
    """

    def __init__(
        self,
        threshold: float = config.RECOGNITION_THRESHOLD,
    ) -> None:
        """Initialise the recognizer — create CLAHE and HOG instances.

        Args:
            threshold: Cosine similarity in [0, 1] above which two
                       feature vectors are considered the same person.
        """
        self.threshold: float = threshold
        self.resize_dim: tuple[int, int] = (
            config.FACE_RESIZE_DIM,
            config.FACE_RESIZE_DIM,
        )

        # CLAHE for illumination normalisation
        self._clahe: cv2.CLAHE = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=(config.CLAHE_TILE_SIZE, config.CLAHE_TILE_SIZE),
        )

        # HOG descriptor tuned for face-sized images
        win_size: tuple[int, int] = self.resize_dim
        block_size: tuple[int, int] = (16, 16)
        block_stride: tuple[int, int] = (8, 8)
        cell_size: tuple[int, int] = (8, 8)
        n_bins: int = 9
        self._hog: cv2.HOGDescriptor = cv2.HOGDescriptor(
            win_size, block_size, block_stride, cell_size, n_bins
        )

        self._grid: int = config.LBP_GRID_SIZE
        self._radius: int = config.LBP_RADIUS

        logger.info(
            "FaceRecognizer initialised (threshold=%.2f, "
            "resize=%dx%d, LBP grid=%d, CLAHE clip=%.1f).",
            threshold,
            self.resize_dim[0],
            self.resize_dim[1],
            self._grid,
            config.CLAHE_CLIP_LIMIT,
        )

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Resize and apply CLAHE illumination normalisation.

        Args:
            face_crop: BGR face crop (any size).

        Returns:
            CLAHE-normalised grayscale image at ``resize_dim``.
        """
        resized: np.ndarray = cv2.resize(face_crop, self.resize_dim)
        lab: np.ndarray = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        normalised: np.ndarray = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        grey: np.ndarray = cv2.cvtColor(normalised, cv2.COLOR_BGR2GRAY)
        return grey

    # ------------------------------------------------------------------
    # LBP (pure NumPy, no scikit-image)
    # ------------------------------------------------------------------

    def _compute_lbp_image(self, grey: np.ndarray) -> np.ndarray:
        """Compute a basic 8-neighbour LBP image.

        For each pixel, compare it with its 8 immediate neighbours
        (radius=1).  Each comparison yields one bit → 8-bit code (0-255).

        Args:
            grey: Grayscale uint8 image.

        Returns:
            LBP image of same shape minus the border (uint8).
        """
        r: int = self._radius
        h, w = grey.shape
        # Pad to avoid boundary issues
        padded: np.ndarray = cv2.copyMakeBorder(
            grey, r, r, r, r, cv2.BORDER_REFLECT
        )
        centre: np.ndarray = padded[r:r + h, r:r + w].astype(np.int16)

        # 8 neighbours (clockwise from top-left)
        offsets: list[tuple[int, int]] = [
            (-r, -r), (-r, 0), (-r, r),
            (0, r), (r, r), (r, 0),
            (r, -r), (0, -r),
        ]

        lbp: np.ndarray = np.zeros((h, w), dtype=np.uint8)
        for bit, (dy, dx) in enumerate(offsets):
            neighbour: np.ndarray = padded[
                r + dy: r + dy + h,
                r + dx: r + dx + w,
            ].astype(np.int16)
            lbp |= ((neighbour >= centre).astype(np.uint8) << bit)

        return lbp

    def _compute_lbp_features(self, grey: np.ndarray) -> np.ndarray:
        """Compute spatial LBP histogram features.

        The image is divided into a ``grid x grid`` of non-overlapping
        cells.  A 256-bin histogram is computed per cell and all are
        concatenated (total length = grid^2 * 256).

        Args:
            grey: Preprocessed grayscale face image.

        Returns:
            L2-normalised float32 feature vector.
        """
        lbp_img: np.ndarray = self._compute_lbp_image(grey)
        h, w = lbp_img.shape
        cell_h: int = h // self._grid
        cell_w: int = w // self._grid

        histograms: list[np.ndarray] = []
        for row in range(self._grid):
            for col in range(self._grid):
                cell: np.ndarray = lbp_img[
                    row * cell_h: (row + 1) * cell_h,
                    col * cell_w: (col + 1) * cell_w,
                ]
                hist: np.ndarray = cv2.calcHist(
                    [cell], [0], None, [256], [0, 256]
                ).flatten()
                histograms.append(hist)

        lbp_feat: np.ndarray = np.concatenate(histograms).astype(np.float32)
        norm: float = float(np.linalg.norm(lbp_feat))
        if norm > 0:
            lbp_feat /= norm
        return lbp_feat

    # ------------------------------------------------------------------
    # HOG
    # ------------------------------------------------------------------

    def _compute_hog_features(self, grey: np.ndarray) -> np.ndarray:
        """Compute HOG descriptor for the face image.

        Args:
            grey: Preprocessed grayscale face image at ``resize_dim``.

        Returns:
            L2-normalised float32 HOG feature vector.
        """
        hog_feat: np.ndarray = self._hog.compute(grey).flatten().astype(
            np.float32
        )
        norm: float = float(np.linalg.norm(hog_feat))
        if norm > 0:
            hog_feat /= norm
        return hog_feat

    # ------------------------------------------------------------------
    # Public feature extraction
    # ------------------------------------------------------------------

    def get_features(self, face_crop: np.ndarray) -> np.ndarray:
        """Extract a combined LBP + HOG feature vector.

        Args:
            face_crop: BGR face crop (any size).

        Returns:
            L2-normalised float32 feature vector.
        """
        grey: np.ndarray = self._preprocess(face_crop)

        lbp_feat: np.ndarray = self._compute_lbp_features(grey)
        hog_feat: np.ndarray = self._compute_hog_features(grey)

        # Concatenate with equal weighting (both already L2-normalised)
        combined: np.ndarray = np.concatenate([lbp_feat, hog_feat])
        norm: float = float(np.linalg.norm(combined))
        if norm > 0:
            combined /= norm
        return combined

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def compare(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors.

        Args:
            feat1: First feature vector (float32).
            feat2: Second feature vector (float32).

        Returns:
            Similarity score in [0, 1].
        """
        n1: float = float(np.linalg.norm(feat1))
        n2: float = float(np.linalg.norm(feat2))
        if n1 == 0 or n2 == 0:
            return 0.0
        score: float = float(np.dot(feat1, feat2) / (n1 * n2))
        return max(0.0, min(1.0, score))

    def find_match(
        self,
        query_feat: np.ndarray,
        stored: list[dict],
    ) -> tuple[str, float]:
        """Find the best matching stored face.

        Supports multi-sample galleries: each entry in *stored* may
        contain either a single ``features`` vector or a list of
        ``samples`` vectors.  The best (max) score across all samples
        is used.

        Args:
            query_feat: Feature vector of the query face.
            stored: List of dicts. Each dict has ``name`` and either
                    ``features`` (single vector) or ``samples`` (list
                    of vectors).

        Returns:
            Tuple of (best_name, best_score).  Returns
            ``('Unknown', 0.0)`` if no match exceeds the threshold.
        """
        best_name: str = 'Unknown'
        best_score: float = 0.0

        for entry in stored:
            samples: list[np.ndarray] = entry.get('samples', [])
            if not samples and 'features' in entry:
                samples = [entry['features']]

            for sample_feat in samples:
                score: float = self.compare(query_feat, sample_feat)
                if score > best_score:
                    best_score = score
                    best_name = entry['name']

        if best_score < self.threshold:
            return ('Unknown', 0.0)

        return (best_name, round(best_score, 4))

    def is_same_person(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
    ) -> bool:
        """Check whether two feature vectors belong to the same person.

        Args:
            feat1: First feature vector.
            feat2: Second feature vector.

        Returns:
            True if cosine similarity >= threshold.
        """
        return self.compare(feat1, feat2) >= self.threshold
