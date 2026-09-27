"""
main.py
-------
Why this module exists:
    This is the single FastAPI application object. It wires together
    startup/shutdown behavior and the routers defined in backend/routers/.
    Nothing business-logic-related lives directly in this file — its only
    job is composition, so it stays small as the app grows across phases.

Data flow (Phase 1):
    Streamlit (later phase) or curl/browser
        -> HTTP GET /health
        -> this module's health_check()
        -> checks whether Ollama's HTTP API is reachable
        -> returns a HealthResponse (see schemas.py)

Why we check Ollama here already, even before Phase 5 (LLM integration):
    Section 20 (acceptance criteria) and Section 8 both say the app must
    never silently pretend a capability works. A visible, honest health
    check from day one sets that pattern early, and gives you (the user)
    something real to point a browser at right away to confirm the backend
    runs at all.
"""

import importlib.util
import logging
import os

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.routers import chat, compare, documents, summarize, voice
from backend.schemas import DependencyStatus, HealthResponse
from backend.services import metadata_db, vectorstore

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Assistant Backend",
    description="Local-first RAG backend: ingestion, retrieval, chat, "
    "summaries, comparisons, and voice endpoints.",
    version="0.1.0",
)

_cors_origins = settings.cors_origin_list()
if _cors_origins:
    # Streamlit server-side HTTP does not use browser CORS. This is only
    # needed if a browser (or Streamlit custom component) calls the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(summarize.router)
app.include_router(compare.router)
app.include_router(voice.router)


@app.on_event("startup")
def on_startup() -> None:
    """Runs once when uvicorn starts the app.

    Makes sure the local data directories exist (ChromaDB folder, SQLite
    parent folder, cache, audio, documents) and that the SQLite documents
    table exists, so every router can assume both are already there
    instead of re-checking on every request.
    """
    settings.ensure_data_dirs_exist()
    (settings.chroma_db_abs_path.parent / "documents").mkdir(parents=True, exist_ok=True)
    metadata_db.init_db()
    logger.info("Backend startup complete. Data dir: %s", settings.chroma_db_abs_path.parent)


def _check_ollama() -> DependencyStatus:
    """
    Check whether Ollama's local HTTP API is reachable and whether the
    configured chat model has been pulled.

    Why a short timeout: this runs on every /health call, which the
    Streamlit sidebar may poll periodically. We'd rather show "unavailable"
    quickly than hang the health check waiting on a dead service.

    Returns a DependencyStatus rather than raising, because an unavailable
    Ollama is an expected, normal state (e.g. before the user has started
    it) — not a backend bug.
    """
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", [])]
        if settings.ollama_chat_model in models:
            return DependencyStatus(
                name="ollama",
                available=True,
                detail=f"Reachable; model '{settings.ollama_chat_model}' is pulled.",
            )
        return DependencyStatus(
            name="ollama",
            available=False,
            detail=(
                f"Ollama is running, but model '{settings.ollama_chat_model}' "
                f"is not pulled yet. Available models: {models or 'none'}."
            ),
        )
    except httpx.ConnectError:
        return DependencyStatus(
            name="ollama",
            available=False,
            detail=(
                f"Could not connect to Ollama at {settings.ollama_base_url}. "
                "Is it installed and running?"
            ),
        )
    except httpx.HTTPError as exc:
        return DependencyStatus(name="ollama", available=False, detail=f"Ollama error: {exc}")


def _check_openai() -> DependencyStatus:
    if not settings.openai_api_key:
        return DependencyStatus(
            name="openai",
            available=False,
            detail="OPENAI_API_KEY is not set.",
        )
    return DependencyStatus(
        name="openai",
        available=True,
        detail=(
            f"API key configured; chat model '{settings.openai_chat_model}', "
            f"embeddings '{settings.openai_embedding_model}'."
        ),
    )


def _check_huggingface_inference() -> DependencyStatus:
    if not settings.hf_token:
        return DependencyStatus(
            name="huggingface_inference",
            available=False,
            detail="HF_TOKEN is not configured as a Space secret.",
        )
    return DependencyStatus(
        name="huggingface_inference",
        available=True,
        detail=f"Token configured; model '{settings.hf_model_id}' will use Inference Providers.",
    )


def _check_chroma() -> DependencyStatus:
    """ChromaDB is a local embedded/persistent store, not a network
    service — 'available' here means the on-disk collection can be
    opened/created, i.e. no permissions or corruption problem."""
    if vectorstore.is_reachable():
        return DependencyStatus(
            name="chromadb", available=True, detail=str(settings.chroma_db_abs_path)
        )
    return DependencyStatus(
        name="chromadb",
        available=False,
        detail=f"Could not open ChromaDB at {settings.chroma_db_abs_path}.",
    )


def _check_package(package_name: str, dep_name: str, extra_detail: str = "") -> DependencyStatus:
    """Cheap presence check (does the package import) for the heavier
    optional dependencies (embeddings, whisper, tts). This deliberately
    does NOT load the actual model — that can mean a large download and
    real load time, which /health must never trigger on every poll."""
    available = importlib.util.find_spec(package_name) is not None
    detail = extra_detail or ("installed" if available else f"'{package_name}' not installed")
    return DependencyStatus(name=dep_name, available=available, detail=detail)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Report backend process status and real dependency availability.

    The configured LLM provider and ChromaDB are checked. sentence-
    transformers, faster-whisper, and piper-tts are only checked for
    "is the package importable" — actually loading any of those models
    here would make every health poll slow (and, for TTS/whisper,
    pointless when the user has them disabled in .env). Each check is
    independent so one missing optional dependency (e.g. no TTS) never
    hides the status of everything else.
    """
    provider = settings.llm_provider.lower()
    if provider == "openai":
        llm_status = _check_openai()
    elif provider == "huggingface":
        llm_status = _check_huggingface_inference()
    elif provider == "ollama":
        llm_status = _check_ollama()
    else:
        llm_status = DependencyStatus(
            name="llm_provider",
            available=False,
            detail=f"Unsupported LLM_PROVIDER '{settings.llm_provider}'.",
        )
    embedding_status = (
        DependencyStatus(
            name="embeddings",
            available=bool(settings.openai_api_key),
            detail=(
                f"OpenAI embeddings ({settings.openai_embedding_model})"
                if settings.openai_api_key
                else "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
            ),
        )
        if settings.embedding_provider.lower() == "openai"
        else _check_package(
            "sentence_transformers",
            "embeddings",
            extra_detail="sentence-transformers (local). Use EMBEDDING_PROVIDER=openai if you want OpenAI embeddings.",
        )
    )
    dependencies = [
        llm_status,
        _check_chroma(),
        embedding_status,
    ]
    if settings.tts_enabled:
        dependencies.append(_check_package("piper", "tts"))
    if settings.ocr_enabled:
        dependencies.append(_check_package("pytesseract", "ocr"))
    return HealthResponse(status="ok", dependencies=dependencies)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", settings.backend_port)),
    )
