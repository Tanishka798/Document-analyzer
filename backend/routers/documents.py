"""
routers/documents.py
---------------------
Why this module exists:
    HTTP surface for document ingestion (Phase 2/3): upload a file, list
    what's been uploaded, fetch one document's status, delete a document.
    All the real work (extraction, chunking, embedding, storage) is
    delegated to backend/services/* — this file only handles HTTP
    concerns: validating the upload, mapping errors to status codes, and
    shaping responses with the schemas in schemas.py.

Ingestion flow for POST /documents/upload:
    1. Validate size (MAX_FILE_SIZE_MB) and extension before doing any
       real work — fail fast and cheaply on bad input.
    2. Save the raw file to data/documents/{id}{ext} under a generated
       id (never the user-supplied filename) so path traversal / weird
       filenames can't affect the filesystem (Section 14, security).
    3. Record a 'processing' row in SQLite immediately, so the document
       shows up in GET /documents even while extraction/embedding is
       still running.
    4. Extract -> chunk -> embed -> store in Chroma. On any failure,
       mark the SQLite row 'failed' with the real error message rather
       than deleting it or hiding the failure.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from backend.config import settings
from backend.schemas import DocumentInfo, DocumentListResponse
from backend.services import chunking, embeddings, extraction, metadata_db, vectorstore
from backend.services.extraction import ExtractionError

router = APIRouter(prefix="/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx"}


def _row_to_info(row) -> DocumentInfo:
    return DocumentInfo(
        id=row["id"],
        filename=row["filename"],
        file_size_bytes=row["file_size_bytes"],
        status=row["status"],
        num_chunks=row["num_chunks"],
        error=row["error"],
        uploaded_at=row["uploaded_at"],
    )


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile) -> DocumentInfo:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f}MB, exceeds MAX_FILE_SIZE_MB={settings.max_file_size_mb}.",
        )
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    document_id = uuid.uuid4().hex
    saved_path = settings.chroma_db_abs_path.parent / "documents" / f"{document_id}{suffix}"
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_bytes(raw)

    metadata_db.insert_document(document_id, file.filename or document_id, len(raw))

    try:
        text = extraction.extract_text(saved_path, file.filename or document_id)
        chunks = chunking.chunk_text(
            text,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        if not chunks:
            raise ExtractionError("Document contained no text after chunking.")

        chunk_texts = [t for t in (embeddings.sanitize_text(c.text) for c in chunks) if t]
        if not chunk_texts:
            raise ExtractionError("Document contained no usable text after cleaning.")
        vectors = embeddings.embed_texts(chunk_texts)
        vectorstore.add_chunks(document_id, file.filename or document_id, chunk_texts, vectors)
        metadata_db.mark_ready(document_id, num_chunks=len(chunks))
    except ExtractionError as exc:
        metadata_db.mark_failed(document_id, str(exc))
    except Exception as exc:  # noqa: BLE001 - record, don't crash the request
        metadata_db.mark_failed(document_id, f"Unexpected error during ingestion: {exc}")

    row = metadata_db.get_document(document_id)
    return _row_to_info(row)


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    rows = metadata_db.list_documents()
    return DocumentListResponse(documents=[_row_to_info(r) for r in rows])


@router.get("/{document_id}", response_model=DocumentInfo)
def get_document(document_id: str) -> DocumentInfo:
    row = metadata_db.get_document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return _row_to_info(row)


@router.delete("/{document_id}")
def delete_document(document_id: str) -> dict:
    row = metadata_db.get_document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    vectorstore.delete_document(document_id)
    metadata_db.delete_document(document_id)

    # Best-effort: also remove the stored raw file. Its extension isn't in
    # the metadata row, so glob for the document id under data/documents.
    documents_dir = settings.chroma_db_abs_path.parent / "documents"
    for match in documents_dir.glob(f"{document_id}.*"):
        match.unlink(missing_ok=True)

    return {"deleted": True, "id": document_id}
