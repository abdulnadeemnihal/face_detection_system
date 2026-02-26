"""
core/database.py — SQLite3 face database for storing registered face data.

Stores face images, LBP+HOG feature vectors, and metadata.
Supports a multi-sample gallery: up to N feature vectors per person.
Uses context managers for every connection — never leaves connections open.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


class FaceDatabase:
    """SQLite3-backed storage for registered face features and metadata.

    Two tables:
    - ``faces`` — one row per person (name, thumbnail, timestamps).
    - ``face_samples`` — multiple feature vectors per person (gallery).

    Attributes:
        db_path: Filesystem path to the SQLite database file.
    """

    _SCHEMA: str = """
        CREATE TABLE IF NOT EXISTS faces (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            name              TEXT NOT NULL UNIQUE,
            face_image        BLOB NOT NULL,
            features          BLOB NOT NULL,
            registered_at     TEXT NOT NULL,
            last_seen_at      TEXT,
            recognition_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS face_samples (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            features   BLOB NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (name) REFERENCES faces(name) ON DELETE CASCADE
        );
    """

    def __init__(self, db_path: str) -> None:
        """Initialise the database, creating files and tables if needed.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path: str = db_path
        self._create_tables()
        logger.info("FaceDatabase initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a new connection with row-factory and FK support."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _create_tables(self) -> None:
        """Create the faces and face_samples tables if they don't exist."""
        conn = self._connect()
        try:
            conn.executescript(self._SCHEMA)
            conn.commit()
        finally:
            conn.close()
        logger.debug("Database tables verified.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_face(
        self,
        name: str,
        img_bytes: bytes,
        features: np.ndarray,
    ) -> None:
        """Insert or update a face record and add a gallery sample.

        On conflict the image and primary features are updated,
        last_seen_at is refreshed, and recognition_count increments.
        A new sample is also added to the gallery (oldest trimmed if
        exceeding ``MAX_SAMPLES_PER_PERSON``).

        Args:
            name: Unique display name for the person.
            img_bytes: JPEG-encoded face crop as raw bytes.
            features: 1-D float32 feature vector from FaceRecognizer.
        """
        now: str = datetime.now(timezone.utc).isoformat()
        feat_blob: bytes = features.astype(np.float32).tobytes()

        sql_upsert = """
            INSERT INTO faces (name, face_image, features, registered_at,
                               last_seen_at, recognition_count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(name) DO UPDATE SET
                face_image        = excluded.face_image,
                features          = excluded.features,
                last_seen_at      = excluded.last_seen_at,
                recognition_count = recognition_count + 1
        """
        sql_sample = """
            INSERT INTO face_samples (name, features, created_at)
            VALUES (?, ?, ?)
        """
        conn = self._connect()
        try:
            conn.execute(sql_upsert, (name, img_bytes, feat_blob, now, now))
            conn.execute(sql_sample, (name, feat_blob, now))
            self._trim_samples(conn, name)
            conn.commit()
        finally:
            conn.close()
        logger.info("Stored face + sample for '%s'.", name)

    @staticmethod
    def _trim_samples(conn: sqlite3.Connection, name: str) -> None:
        """Keep only the newest MAX_SAMPLES_PER_PERSON gallery entries."""
        limit: int = config.MAX_SAMPLES_PER_PERSON
        conn.execute(
            """
            DELETE FROM face_samples
            WHERE name = ? AND id NOT IN (
                SELECT id FROM face_samples
                WHERE name = ? ORDER BY id DESC LIMIT ?
            )
            """,
            (name, name, limit),
        )

    def get_all_features(self) -> list[dict]:
        """Return all stored feature vectors grouped by name.

        Returns:
            List of dicts with keys:
            - ``name`` (str)
            - ``features`` (np.ndarray, float32) — primary vector
            - ``samples`` (list[np.ndarray]) — gallery vectors
        """
        conn = self._connect()
        try:
            face_rows = conn.execute(
                "SELECT name, features FROM faces"
            ).fetchall()
            sample_rows = conn.execute(
                "SELECT name, features FROM face_samples ORDER BY id DESC"
            ).fetchall()
        finally:
            conn.close()

        # Group samples by name
        samples_map: dict[str, list[np.ndarray]] = {}
        for row in sample_rows:
            feat = np.frombuffer(row['features'], dtype=np.float32).copy()
            samples_map.setdefault(row['name'], []).append(feat)

        results: list[dict] = []
        for row in face_rows:
            primary = np.frombuffer(
                row['features'], dtype=np.float32
            ).copy()
            name: str = row['name']
            gallery: list[np.ndarray] = samples_map.get(name, [primary])
            results.append({
                'name': name,
                'features': primary,
                'samples': gallery,
            })
        return results

    def get_face(self, name: str) -> Optional[dict]:
        """Retrieve a single face record by name.

        Args:
            name: The registered person name.

        Returns:
            Dict with face metadata or ``None`` if not found.
        """
        sql = """
            SELECT id, name, face_image, features,
                   registered_at, last_seen_at, recognition_count
            FROM faces WHERE name = ?
        """
        conn = self._connect()
        try:
            row = conn.execute(sql, (name,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            'id': row['id'],
            'name': row['name'],
            'face_image': row['face_image'],
            'features': np.frombuffer(
                row['features'], dtype=np.float32
            ).copy(),
            'registered_at': row['registered_at'],
            'last_seen_at': row['last_seen_at'],
            'recognition_count': row['recognition_count'],
        }

    def delete_face(self, name: str) -> bool:
        """Delete a face record and all its gallery samples by name.

        Args:
            name: The registered person name.

        Returns:
            True if a row was deleted, False otherwise.
        """
        conn = self._connect()
        try:
            # Samples deleted via ON DELETE CASCADE
            cursor = conn.execute(
                "DELETE FROM faces WHERE name = ?", (name,)
            )
            conn.commit()
            deleted: bool = cursor.rowcount > 0
        finally:
            conn.close()
        if deleted:
            logger.info("Deleted face '%s'.", name)
        else:
            logger.warning(
                "Attempted to delete non-existent face '%s'.", name
            )
        return deleted

    def clear_all(self) -> int:
        """Delete every face record and all gallery samples.

        Returns:
            Number of face rows deleted.
        """
        conn = self._connect()
        try:
            conn.execute("DELETE FROM face_samples")
            cursor = conn.execute("DELETE FROM faces")
            conn.commit()
            count: int = cursor.rowcount
        finally:
            conn.close()
        logger.info("Cleared all faces (%d removed).", count)
        return count

    def get_stats(self) -> dict:
        """Return summary statistics about stored faces.

        Returns:
            Dict with ``total_faces``, ``last_registered``, ``last_seen``.
        """
        conn = self._connect()
        try:
            total: int = conn.execute(
                "SELECT COUNT(*) FROM faces"
            ).fetchone()[0]
            last_reg_row = conn.execute(
                "SELECT registered_at FROM faces "
                "ORDER BY registered_at DESC LIMIT 1"
            ).fetchone()
            last_seen_row = conn.execute(
                "SELECT last_seen_at FROM faces "
                "WHERE last_seen_at IS NOT NULL "
                "ORDER BY last_seen_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        return {
            'total_faces': total,
            'last_registered': (
                last_reg_row['registered_at'] if last_reg_row else None
            ),
            'last_seen': (
                last_seen_row['last_seen_at'] if last_seen_row else None
            ),
        }

    def get_face_list(self) -> list[dict]:
        """Return a lightweight list of all registered faces (no blobs).

        Returns:
            List of dicts with name, registered_at, recognition_count,
            last_seen_at.
        """
        sql = """
            SELECT name, registered_at, recognition_count, last_seen_at
            FROM faces ORDER BY registered_at DESC
        """
        conn = self._connect()
        try:
            rows = conn.execute(sql).fetchall()
        finally:
            conn.close()
        return [
            {
                'name': r['name'],
                'registered_at': r['registered_at'],
                'recognition_count': r['recognition_count'],
                'last_seen_at': r['last_seen_at'],
            }
            for r in rows
        ]
