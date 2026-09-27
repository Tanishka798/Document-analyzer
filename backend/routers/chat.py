"""
routers/chat.py
----------------
Why this module exists:
    Implements the core RAG loop (Phase 5): embed the question, retrieve
    the most relevant chunks from ChromaDB, build a grounded prompt, and
    ask Ollama to answer using only that retrieved context. Returns the
    answer together with the exact source chunks used, so the caller can
    show citations rather than trusting the model unverified.

Why we refuse to answer with zero retrieved context:
    If nothing relevant is in the index (e.g. no documents uploaded yet,
    or the corpus really doesn't cover the question), the honest answer
    is "I don't have information about that in the uploaded documents" —
    not letting the base model guess from its own training data, which
    would defeat the point of a grounded, cited local assistant.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.schemas import ChatRequest, ChatResponse, SourceChunk
from backend.services import embeddings, ollama_client, vectorstore

router = APIRouter(tags=["chat"])

_SYSTEM_PROMPT = (
    "You are a document assistant. Answer ONLY using the provided context "
    "chunks. If the context does not contain the answer, say clearly that "
    "the uploaded documents don't cover it. Do not use outside knowledge. "
    "Cite which source number(s) you used, like [1], [2]."
)


def _build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(f"[{i}] (from {chunk['filename']})\n{chunk['text']}")
    context = "\n\n".join(context_blocks)
    context = ollama_client.trim_to_budget(context, settings.ollama_context_budget_tokens)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    query_vector = embeddings.embed_query(request.query)
    matches = vectorstore.query(
        query_vector,
        top_k=settings.retrieval_top_k,
        document_ids=request.document_ids,
    )

    if not matches:
        return ChatResponse(
            answer=(
                "I don't have any relevant information in the uploaded "
                "documents to answer that. Try uploading a document first, "
                "or rephrase your question."
            ),
            sources=[],
        )

    prompt = _build_prompt(request.query, matches)

    try:
        answer = ollama_client.generate(prompt, system=_SYSTEM_PROMPT)
    except ollama_client.OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    sources = [
        SourceChunk(
            document_id=m["document_id"],
            filename=m["filename"],
            chunk_index=m["chunk_index"],
            text=m["text"],
            distance=m["distance"],
        )
        for m in matches
    ]
    return ChatResponse(answer=answer, sources=sources)
