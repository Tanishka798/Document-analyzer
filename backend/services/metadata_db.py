"""
metadata_db.py
--------------
Why this module exists:
    ChromaDB stores chunk text + embeddings, but has no good concept of
    "a document" as a whole (upload time, original filename, file size,
    ingestion status/errors). SQLite holds exactly that: one row per
    uploaded document, used to populate the document list UI and to know
    what to clean up when a document is deleted.

Why plain sqlite3, not an ORM:
    One table, five columns, a handful of queries. An ORM would add a
    dependency and an abstraction layer for no real benefit at this
    scale. Every function opens and closes its own short-lived
    connection — simplest safe way to use sqlite3 from a multi-request
    ASGI app without worrying about sharing connections across threads.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from backend.config import settings


@contextmanager
def _connect():
    conn = sqlite3.connect(str(settings.sqlite_db_abs_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the documents table if it doesn't exist yet. Safe to call
    on every startup."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL,
                num_chunks INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                uploaded_at TEXT NOT NULL
            )
            """
        )


def insert_document(document_id: str, filename: str, file_size_bytes: int) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, file_size_bytes, status, num_chunks, error, uploaded_at)
            VALUES (?, ?, ?, 'processing', 0, NULL, ?)
            """,
            (document_id, filename, file_size_bytes, datetime.now(timezone.utc).isoformat()),
        )


def mark_ready(document_id: str, num_chunks: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET status = 'ready', num_chunks = ?, error = NULL WHERE id = ?",
            (num_chunks, document_id),
        )


def mark_failed(document_id: str, error: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET status = 'failed', error = ? WHERE id = ?",
            (error, document_id),
        )


def list_documents() -> list[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()


def get_document(document_id: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()


def delete_document(document_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
