"""
routers/summarize.py
---------------------
Why this module exists:
    Phase 6, first half: summarize one already-ingested document in full,
    as opposed to /chat's top-k similarity retrieval. A summary needs the
    *whole* document (in chunk order), not just the chunks that happen to
    be most similar to some query, so this reads directly from Chroma via
    get_all_chunks_for_document rather than going through vectorstore.query.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.schemas import SummarizeResponse
from backend.services import metadata_db, ollama_client, vectorstore

router = APIRouter(tags=["summarize"])

_SYSTEM_PROMPT = (
    "You are a document assistant. Write a clear, well-organized summary "
    "of the provided document text. Use only what's in the text."
)


def summarize_document_text(full_text: str) -> str:
    """Shared by this router and compare.py: trim to budget and ask Ollama
    for a summary. Kept as a plain function (not inlined in the route) so
    compare.py can reuse it without an HTTP round-trip."""
    trimmed = ollama_client.trim_to_budget(full_text, settings.ollama_context_budget_tokens)
    prompt = f"Document text:\n{trimmed}\n\nWrite a concise summary (a few paragraphs):"
    return ollama_client.generate(prompt, system=_SYSTEM_PROMPT)


@router.post("/documents/{document_id}/summarize", response_model=SummarizeResponse)
def summarize(document_id: str) -> SummarizeResponse:
    row = metadata_db.get_document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if row["status"] != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Document status is '{row['status']}', not 'ready' yet.",
        )

    chunks = vectorstore.get_all_chunks_for_document(document_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="No indexed content found for this document.")

    full_text = "\n\n".join(chunks)
    try:
        summary = summarize_document_text(full_text)
    except ollama_client.OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SummarizeResponse(document_id=document_id, filename=row["filename"], summary=summary)
