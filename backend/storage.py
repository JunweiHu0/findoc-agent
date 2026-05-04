"""SQLite-backed persistence for conversations, messages, and documents.

Single database file: data/findoc.db (WAL mode, thread-safe).
Used by the backend server for conversation history (P13) and
document registry (P14).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger

from agent.config import ROOT

DB_PATH = ROOT / "data" / "findoc.db"

_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conv_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            citations TEXT DEFAULT '[]',
            pages TEXT DEFAULT '[]',
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conv
            ON messages(conv_id, created_at);

        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            source_filename TEXT NOT NULL,
            page_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'queued'
                CHECK(status IN ('queued','encoding','ready','failed')),
            created_at REAL NOT NULL
        );

        -- P25: cross-turn fact memory
        CREATE TABLE IF NOT EXISTS conv_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            entity TEXT DEFAULT '',
            period TEXT DEFAULT '',
            metric TEXT DEFAULT '',
            value REAL,
            unit TEXT DEFAULT '',
            source_doc TEXT NOT NULL DEFAULT '',
            source_page INTEGER NOT NULL DEFAULT 0,
            text TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conv_facts_lookup
            ON conv_facts(conv_id, entity, period, metric);
        """
    )
    conn.commit()


def init_db() -> None:
    with _lock:
        conn = _get_conn()
        _ensure_tables(conn)
        conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(title: str = "") -> dict:
    conv_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _lock:
        conn = _get_conn()
        _ensure_tables(conn)
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now),
        )
        conn.commit()
        conn.close()
    logger.info(f"Created conversation {conv_id}: {title}")
    return {"id": conv_id, "title": title, "created_at": now, "updated_at": now}


def create_conversation_with_id(conv_id: str, title: str = "") -> dict:
    """Create a conversation with a specific ID (used by DataLayer to sync Chainlit thread IDs)."""
    now = time.time()
    with _lock:
        conn = _get_conn()
        _ensure_tables(conn)
        try:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title, now, now),
            )
            conn.commit()
        except Exception:
            pass  # already exists
        conn.close()
    return {"id": conv_id, "title": title, "created_at": now, "updated_at": now}


def update_conversation_title(conv_id: str, title: str) -> bool:
    now = time.time()
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conv_id),
        )
        conn.commit()
        affected = conn.total_changes
        conn.close()
    return affected > 0


def list_conversations() -> list[dict]:
    with _lock:
        conn = _get_conn()
        _ensure_tables(conn)
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
    return [
        {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
        for r in rows
    ]


def get_conversation(conv_id: str) -> dict | None:
    with _lock:
        conn = _get_conn()
        _ensure_tables(conn)
        conv = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        if not conv:
            conn.close()
            return None
        msgs = conn.execute(
            "SELECT id, role, content, citations, pages, created_at FROM messages WHERE conv_id = ? ORDER BY created_at",
            (conv_id,),
        ).fetchall()
        conn.close()
    return {
        "id": conv[0],
        "title": conv[1],
        "created_at": conv[2],
        "updated_at": conv[3],
        "messages": [
            {
                "id": m[0],
                "role": m[1],
                "content": m[2],
                "citations": json.loads(m[3]),
                "pages": json.loads(m[4]),
                "created_at": m[5],
            }
            for m in msgs
        ],
    }


def delete_conversation(conv_id: str) -> bool:
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        affected = conn.total_changes
        conn.close()
    return affected > 0


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def add_message(conv_id: str, role: str, content: str,
                citations: list | None = None, pages: list | None = None) -> str:
    msg_id = uuid.uuid4().hex[:12]
    now = time.time()
    citations_json = json.dumps(citations or [], ensure_ascii=False)
    pages_json = json.dumps(pages or [], ensure_ascii=False)
    with _lock:
        conn = _get_conn()
        _ensure_tables(conn)
        conn.execute(
            "INSERT INTO messages (id, conv_id, role, content, citations, pages, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg_id, conv_id, role, content, citations_json, pages_json, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conv_id),
        )
        conn.commit()
        conn.close()
    return msg_id


def get_messages(conv_id: str) -> list[dict]:
    with _lock:
        conn = _get_conn()
        _ensure_tables(conn)
        rows = conn.execute(
            "SELECT id, role, content, citations, pages, created_at "
            "FROM messages WHERE conv_id = ? ORDER BY created_at",
            (conv_id,),
        ).fetchall()
        conn.close()
    return [
        {
            "id": r[0], "role": r[1], "content": r[2],
            "citations": json.loads(r[3]), "pages": json.loads(r[4]),
            "created_at": r[5],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Documents (P14)
# ---------------------------------------------------------------------------

def add_document(doc_id: str, source_filename: str, page_count: int = 0,
                 status: str = "queued") -> None:
    now = time.time()
    with _lock:
        conn = _get_conn()
        _ensure_tables(conn)
        conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, source_filename, page_count, status, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_id, source_filename, page_count, status, now),
        )
        conn.commit()
        conn.close()


def update_document_status(doc_id: str, status: str, page_count: int | None = None) -> None:
    with _lock:
        conn = _get_conn()
        if page_count is not None:
            conn.execute(
                "UPDATE documents SET status = ?, page_count = ? WHERE doc_id = ?",
                (status, page_count, doc_id),
            )
        else:
            conn.execute(
                "UPDATE documents SET status = ? WHERE doc_id = ?",
                (status, doc_id),
            )
        conn.commit()
        conn.close()


def list_documents() -> list[dict]:
    with _lock:
        conn = _get_conn()
        _ensure_tables(conn)
        rows = conn.execute(
            "SELECT doc_id, source_filename, page_count, status, created_at "
            "FROM documents ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
    return [
        {"doc_id": r[0], "source_filename": r[1], "page_count": r[2],
         "status": r[3], "created_at": r[4]}
        for r in rows
    ]


def delete_document(doc_id: str) -> bool:
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()
        affected = conn.total_changes
        conn.close()
    return affected > 0


# ---------------------------------------------------------------------------
# P25: Cross-turn fact memory
# ---------------------------------------------------------------------------

def save_conv_facts(conv_id: str, facts: list[dict]) -> None:
    """Persist structured facts from the current turn for cross-turn reuse.

    Each fact dict is a pydantic Fact.model_dump() or equivalent dict with
    entity/period/metric/value/unit/source_doc/source_page/text fields.
    Only facts with at least one structured field are persisted.
    """
    if not conv_id or not facts:
        return
    now = time.time()
    with _lock:
        conn = _get_conn()
        _ensure_tables(conn)
        for f in facts:
            entity = (f.get("entity") or "").strip()
            period = (f.get("period") or "").strip()
            metric = (f.get("metric") or "").strip()
            # Skip fully unstructured facts — no lookup key to reuse
            if not entity and not period and not metric:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO conv_facts "
                "(conv_id, entity, period, metric, value, unit, source_doc, source_page, text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    conv_id, entity, period, metric,
                    f.get("value"), (f.get("unit") or ""),
                    f.get("source_doc", ""), f.get("source_page", 0),
                    f.get("text", ""), now,
                ),
            )
        conn.commit()
        conn.close()
    logger.debug(f"Saved {len(facts)} facts for conv {conv_id}")


def load_conv_facts(conv_id: str) -> list[dict]:
    """Load all structured facts from prior turns of this conversation."""
    if not conv_id:
        return []
    with _lock:
        conn = _get_conn()
        _ensure_tables(conn)
        rows = conn.execute(
            "SELECT entity, period, metric, value, unit, source_doc, source_page, text "
            "FROM conv_facts WHERE conv_id = ? ORDER BY created_at",
            (conv_id,),
        ).fetchall()
        conn.close()
    return [
        {
            "entity": r[0], "period": r[1], "metric": r[2],
            "value": r[3], "unit": r[4],
            "source_doc": r[5], "source_page": r[6], "text": r[7],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Init on import
# ---------------------------------------------------------------------------
init_db()
