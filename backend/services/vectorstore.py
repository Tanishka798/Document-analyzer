"""
vectorstore.py
--------------
Why this module exists:
    All ChromaDB-specific code lives here, behind a small, plain-Python
    interface (add_chunks / query / delete_document). Routers and other
    services never import chromadb directly — if we ever swapped vector
    stores, this is the only file that would change.

Why embeddings are computed by us, not by Chroma:
    We pass already-computed vectors in (via embeddings.py) instead of
    letting Chroma manage its own embedding function, so there's exactly
    one place (embeddings.py) that knows which model is in use, and
    query-time and ingestion-time embeddings are guaranteed to come from
    the same code path.
"""

from __future__ import annotations

import threading
from typing import Any

from backend.config import settings

_lock = threading.Lock()
_client = None
_collection = None

COLLECTION_NAME = "documents"


def _get_collection():
    global _client, _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                import chromadb

                _client = chromadb.PersistentClient(path=str(settings.chroma_db_abs_path))
                _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def add_chunks(
    document_id: str,
    filename: str,
    chunk_texts: list[str],
    chunk_embeddings: list[list[float]],
) -> None:
    """Store one document's chunks. Chunk ids are namespaced by document
    id (f"{document_id}::{i}") so deleting a document is a single
    metadata-filtered delete, and re-ingesting the same document id
    cleanly overwrites its previous chunks."""
    if not chunk_texts:
        return
    collection = _get_collection()
    ids = [f"{document_id}::{i}" for i in range(len(chunk_texts))]
    metadatas = [
        {"document_id": document_id, "filename": filename, "chunk_index": i}
        for i in range(len(chunk_texts))
    ]
    collection.add(
        ids=ids,
        embeddings=chunk_embeddings,
        documents=chunk_texts,
        metadatas=metadatas,
    )


def query(
    query_embedding: list[float],
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Return up to top_k chunks most similar to query_embedding, optionally
    restricted to a set of document ids. Each result dict has: text,
    document_id, filename, chunk_index, distance (lower = more similar).
    """
    collection = _get_collection()
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    matches: list[dict[str, Any]] = []
    if not result.get("ids") or not result["ids"][0]:
        return matches

    for i in range(len(result["ids"][0])):
        metadata = result["metadatas"][0][i]
        matches.append(
            {
                "text": result["documents"][0][i],
                "document_id": metadata["document_id"],
                "filename": metadata["filename"],
                "chunk_index": metadata["chunk_index"],
                "distance": result["distances"][0][i],
            }
        )
    return matches


def get_all_chunks_for_document(document_id: str) -> list[str]:
    """Fetch every chunk's text for one document, ordered by chunk_index.
    Used by summarize/compare, which need the whole document rather than
    a similarity-ranked subset."""
    collection = _get_collection()
    result = collection.get(where={"document_id": document_id})
    pairs = sorted(
        zip(result["metadatas"], result["documents"]),
        key=lambda pair: pair[0]["chunk_index"],
    )
    return [text for _, text in pairs]


def delete_document(document_id: str) -> None:
    collection = _get_collection()
    collection.delete(where={"document_id": document_id})


def is_reachable() -> bool:
    """Cheap check used by /health: can we open/create the collection at all."""
    try:
        _get_collection()
        return True
    except Exception:  # noqa: BLE001
        return False
