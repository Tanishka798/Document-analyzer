# Document Assistant (Streamlit + FastAPI RAG)

A Streamlit UI with a FastAPI RAG backend for local document work: upload
PDF/DOCX/PPTX/XLSX/TXT/MD files, ask questions with cited sources,
summarize and compare documents, and use voice input.

This project is designed for local development only.

## Architecture

| Piece | Location | Notes |
|---|---|---|
| Streamlit UI | `frontend/app.py` | Upload, document list, Chat / Summarize / Compare, voice |
| HTTP client | `frontend/api_client.py` | Uses `BACKEND_URL` when set; otherwise in-process TestClient |
| FastAPI app | `backend/main.py` | `app` — start with Uvicorn |
| Upload / list / delete | `backend/routers/documents.py` | Size/type checks, extract → chunk → embed → Chroma |
| RAG chat | `backend/routers/chat.py` | Vector retrieval + LLM |
| Summarize / compare | `backend/routers/summarize.py`, `compare.py` | Full-document text from Chroma |
| Voice | `backend/routers/voice.py` | Local faster-whisper or OpenAI Whisper |
| Extraction / OCR | `backend/services/extraction.py` | OCR only if `OCR_ENABLED=true` |
| Embeddings | `backend/services/embeddings.py` | `local` (sentence-transformers) or `openai` |
| Vectors / metadata | ChromaDB + SQLite under `data/` | Local-only persistent storage |

## Models and credentials

| Capability | Local option |
|---|---|
| Chat / summarize / compare | Ollama (`qwen2.5:3b`) or OpenAI |
| Embeddings | sentence-transformers or OpenAI embeddings |
| Speech-to-text | faster-whisper or OpenAI Whisper |
| Spoken replies in the UI | Browser speech synthesis |

## Running locally (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
Copy-Item .env.example .env
```

For local Ollama, in `.env` set `LLM_PROVIDER=ollama` and
`EMBEDDING_PROVIDER=local`, then:

```powershell
ollama pull qwen2.5:3b
ollama serve
```

Optional standalone API:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Streamlit without `BACKEND_URL` still uses in-process FastAPI:

```powershell
streamlit run frontend/app.py
```

To run the backend separately, start Uvicorn as above and set
`BACKEND_URL=http://127.0.0.1:8000` in `.env` or `.streamlit/secrets.toml`.

## What's *not* implemented yet

- Hybrid BM25 + vector retrieval (pure vector search).
- Automated tests (`tests/` is currently empty).

## What's implemented

- **`/health`** — configured LLM, ChromaDB, embeddings.
- **`/documents/upload`**, list, get, delete — extract, chunk, embed, store.
- **`POST /chat`** — RAG with source chunks.
- **`POST /documents/{id}/summarize`**, **`POST /compare`**.
- **`POST /voice/transcribe`** — local Whisper or OpenAI Whisper.
- **`frontend/app.py`** — Streamlit UI.

## Roadmap

1. ✅ Project skeleton, config, health endpoint
2. ✅ Document extraction & chunking
3. ✅ Embeddings + persistent ChromaDB + SQLite metadata
4. ⬜ BM25 + hybrid retrieval (currently pure vector search)
5. ✅ LLM Q&A with citations (Ollama / Hugging Face / OpenAI)
6. ✅ Summaries & comparisons
7. ✅ Streamlit frontend
8. ✅ Voice input (browser mic + transcribe)
9. ⬜ Automated tests
10. ⬜ Additional quality improvements

