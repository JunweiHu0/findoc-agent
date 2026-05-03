"""SQLite cache for VLM page reads.

Key = sha256(image_path + instruction). Same page + same instruction
produces deterministic output from VLM (temperature=0), so caching is safe.

Cache file: data/vlm_cache.db (gitignored).
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from loguru import logger

from agent.config import ROOT

CACHE_PATH = ROOT / "data" / "vlm_cache.db"


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vlm_cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()


def _get_conn() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_PATH))
    _ensure_table(conn)
    return conn


def cache_key(image_path: str, instruction: str) -> str:
    raw = f"{image_path}|{instruction}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(image_path: str, instruction: str) -> str | None:
    """Return cached VLM output, or None if not cached."""
    key = cache_key(image_path, instruction)
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT value FROM vlm_cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            logger.debug(f"VLM cache hit for {Path(image_path).name}")
            return row[0]
    except Exception as e:
        logger.warning(f"VLM cache read failed: {e}")
    return None


def put(image_path: str, instruction: str, value: str) -> None:
    """Store VLM output in cache."""
    key = cache_key(image_path, instruction)
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO vlm_cache (key, value, created_at) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"VLM cache write failed: {e}")


def clear() -> int:
    """Delete all cache entries. Returns count of deleted rows."""
    try:
        conn = _get_conn()
        count = conn.execute("DELETE FROM vlm_cache").rowcount
        conn.commit()
        logger.info(f"Cleared {count} VLM cache entries")
        return count
    except Exception as e:
        logger.warning(f"VLM cache clear failed: {e}")
        return 0


def stats() -> dict:
    """Return cache statistics."""
    try:
        conn = _get_conn()
        count = conn.execute("SELECT COUNT(*) FROM vlm_cache").fetchone()[0]
        size = CACHE_PATH.stat().st_size if CACHE_PATH.exists() else 0
        return {"entries": count, "size_bytes": size, "path": str(CACHE_PATH)}
    except Exception:
        return {"entries": 0, "size_bytes": 0, "path": str(CACHE_PATH)}


if __name__ == "__main__":
    # Smoke test
    print(stats())
    print(get("test.png", "what is on this page?"))
    put("test.png", "what is on this page?", "nothing")
    print(get("test.png", "what is on this page?"))
    clear()
    print(stats())
