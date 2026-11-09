"""Persistent SQLite store for revoked JWT identifiers (JTI) across restarts and worker processes."""
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from app.config import settings


class RevocationStore:
    """Thread-safe and process-shared SQLite revocation registry."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path or settings.SQLITE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jwt_revocations (
                    issuer TEXT NOT NULL,
                    jti TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    revoked_at INTEGER NOT NULL,
                    PRIMARY KEY (issuer, jti)
                );
                CREATE INDEX IF NOT EXISTS idx_jwt_revocations_expires_at
                ON jwt_revocations(expires_at);
                """
            )

    def revoke(
        self,
        issuer: str,
        jti: str,
        expires_at: int,
        revoked_at: Optional[int] = None,
    ) -> bool:
        if not issuer or not jti or len(jti) > 256:
            return False
        now = int(revoked_at if revoked_at is not None else time.time())
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO jwt_revocations (issuer, jti, expires_at, revoked_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(issuer, jti) DO UPDATE SET
                    expires_at = MAX(jwt_revocations.expires_at, excluded.expires_at)
                """,
                (issuer, jti, int(expires_at), now),
            )
            conn.commit()
        return True

    def is_revoked(
        self,
        issuer: str,
        jti: str,
        now: Optional[int] = None,
    ) -> bool:
        if not issuer or not jti:
            return False
        current = int(now if now is not None else time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM jwt_revocations
                WHERE issuer = ? AND jti = ? AND expires_at > ?
                LIMIT 1
                """,
                (issuer, jti, current - 60),
            ).fetchone()
            return row is not None

    def purge_expired(
        self,
        now: Optional[int] = None,
        clock_skew_seconds: int = 60,
        limit: int = 500,
    ) -> int:
        current = int(now if now is not None else time.time())
        cutoff = current - max(0, int(clock_skew_seconds))
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                DELETE FROM jwt_revocations
                WHERE rowid IN (
                    SELECT rowid FROM jwt_revocations
                    WHERE expires_at <= ?
                    LIMIT ?
                )
                """,
                (cutoff, int(limit)),
            )
            deleted = cursor.rowcount if cursor.rowcount is not None else 0
            conn.commit()
            return deleted
