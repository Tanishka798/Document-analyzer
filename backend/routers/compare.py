"""
routers/compare.py
-------------------
Why this module exists:
    Phase 6, second half: compare 2-5 documents. Rather than dumping every
    full document into one giant prompt (which would blow the context
    budget fast with more than two documents), we first summarize each
    document individually (reusing summarize.py's helper), then ask
    Ollama to compare the summaries. This scales to more documents and
    keeps each document's contribution to the final prompt bounded.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas import CompareRequest, CompareResponse
from backend.services import metadata_db, ollama_client, vectorstore
from backend.routers.summarize import summarize_document_text

router = APIRouter(tags=["compare"])

_SYSTEM_PROMPT = (
    "You are a document assistant. Compare the following document "
    "summaries. Point out concrete similarities and differences. Refer "
    "to each document by its filename."
)


@router.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    summaries: list[str] = []
    for document_id in request.document_ids:
        row = metadata_db.get_document(document_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        if row["status"] != "ready":
            raise HTTPException(
                status_code=409,
                detail=f"Document '{row['filename']}' status is '{row['status']}', not 'ready' yet.",
            )

        chunks = vectorstore.get_all_chunks_for_document(document_id)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No indexed content found for document '{row['filename']}'.",
            )

        try:
            summary = summarize_document_text("\n\n".join(chunks))
        except ollama_client.OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        summaries.append(f"Document '{row['filename']}':\n{summary}")

    prompt = "\n\n".join(summaries) + "\n\nCompare these documents:"
    try:
        comparison = ollama_client.generate(prompt, system=_SYSTEM_PROMPT)
    except ollama_client.OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return CompareResponse(document_ids=request.document_ids, comparison=comparison)
