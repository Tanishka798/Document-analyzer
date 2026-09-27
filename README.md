# Document Assistant (Streamlit + FastAPI RAG)

A Streamlit UI with a FastAPI RAG backend: upload documents
(PDF/DOCX/PPTX/XLSX/TXT/MD), ask questions with cited sources,
summarize and compare documents, and use voice input.

**Hosting target:** Streamlit Community Cloud (unchanged Streamlit UI) +
Render (FastAPI). Chat/embeddings on those hosts use the OpenAI API.
Local development can still use Ollama and sentence-transformers.

The Streamlit widgets and layout in `frontend/app.py` are unchanged.

## Architecture (what already existed)

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
| Vectors / metadata | ChromaDB + SQLite under `data/` | Ephemeral on free Render |

There was **no OpenAI integration** in the original repo (Ollama locally,
optional Hugging Face). OpenAI chat, embeddings, and Whisper are added
for Render. Do not put `OPENAI_API_KEY` in the Streamlit app or frontend.

## Deployment blockers (free / starter plans)

These are limitations of the hosts, not missing UI features:

1. **Ollama is not available on Render or Streamlit Cloud.** Set
   `LLM_PROVIDER=openai` and `OPENAI_API_KEY` on Render.
2. **Local embedding models (torch / sentence-transformers) will typically
   OOM on free Render (~512MB).** Use `EMBEDDING_PROVIDER=openai`.
3. **faster-whisper is too large for free Render.** Deployed transcription
   uses OpenAI Whisper when `LLM_PROVIDER=openai`.
4. **Render and Streamlit disks are ephemeral.** Chroma + SQLite live on
   the Render instance and are wiped when the service sleeps, restarts, or
   is redeployed. Re-upload documents after a cold start.
5. **Free Render sleeps after idle.** The first Streamlit request after
   sleep can take 30–60+ seconds and may time out; retry.
6. **OCR (Tesseract + Poppler) is not installed on Render’s Python
   runtime.** Keep `OCR_ENABLED=false` unless you switch the backend to a
   Docker image that installs `tesseract-ocr` and `poppler-utils`.
7. **CORS is not required** for Streamlit’s server-side `httpx` calls.
   Set `CORS_ORIGINS` only if a browser will call the API directly.
8. **Streamlit Community Cloud has no private document isolation.** Anyone
   with the app URL can use the shared backend store.

## Models and credentials

| Capability | Local | Render + Streamlit Cloud |
|---|---|---|
| Chat / summarize / compare | Ollama (`qwen2.5:3b`) | OpenAI (`OPENAI_CHAT_MODEL`, default `gpt-4o-mini`) |
| Embeddings | sentence-transformers | OpenAI embeddings API |
| Speech-to-text | faster-whisper | OpenAI Whisper API |
| Spoken replies in the UI | Browser speech synthesis | Same (no backend TTS required) |

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

To mimic production (two processes), start Uvicorn as above and set
`BACKEND_URL=http://127.0.0.1:8000` in `.env` or `.streamlit/secrets.toml`.

## Deploy A — FastAPI on Render

1. Push this repo to GitHub (do not commit `.env` or `.streamlit/secrets.toml`).
2. In Render: **New → Web Service** → this repo.
3. Settings:
   - **Runtime:** Python 3.11
   - **Build command:** `pip install -r backend/requirements.txt`
   - **Start command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Root directory: repository root (so `backend.main:app` imports).
4. Environment variables (Dashboard → Environment):

   | Key | Value |
   |---|---|
   | `LLM_PROVIDER` | `openai` |
   | `EMBEDDING_PROVIDER` | `openai` |
   | `OPENAI_API_KEY` | your key (Render secret) |
   | `OPENAI_CHAT_MODEL` | `gpt-4o-mini` (or another chat model you have access to) |
   | `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |
   | `OCR_ENABLED` | `false` |
   | `TTS_ENABLED` | `false` |
   | `CORS_ORIGINS` | leave empty |

   Render injects `PORT`; do not set it yourself.
5. Deploy. Open `https://<your-service>.onrender.com/health`. You should
   see `"status": "ok"` and OpenAI + Chroma marked available. Open
   `/docs` to try upload/chat. **This repo has not been deployed for you;
   confirm those URLs after you click Deploy.**

Optional: `render.yaml` in the repo root matches these commands.

### OCR on Render (optional, not on the free Python runtime)

Scanned PDFs need `pytesseract`, `pdf2image`, OS packages `tesseract-ocr`
and `poppler-utils`, plus `OCR_ENABLED=true`. Use a Docker-based Render
service if you need that; native Python builds cannot `apt-get` those
binaries.

## Deploy B — Streamlit Community Cloud (same UI)

1. [share.streamlit.io](https://share.streamlit.io) → **New app** → this repo.
2. **Main file path:** `frontend/app.py`
3. Community Cloud installs **root** `requirements.txt` only (Streamlit +
   httpx). It must **not** install torch/Chroma.
4. **Settings → Secrets:**

   ```toml
   BACKEND_URL = "https://<your-service>.onrender.com"
   ```

   Do **not** put `OPENAI_API_KEY` here unless you are running the backend
   in-process (not the intended cloud setup).

## Deploy C — secrets map

| Secret | Where |
|---|---|
| `OPENAI_API_KEY` | Render only |
| `BACKEND_URL` | Streamlit secrets (and optional local `.env`) |
| `HF_TOKEN` | Only if you keep `LLM_PROVIDER=huggingface` |

## Deploy D — test after both services are live

1. Streamlit sidebar health should not show a backend error.
2. Upload a small `.txt` or text-layer PDF → status ready, chunk count > 0.
3. Chat tab: ask a question that is answered only in that file; confirm
   the answer and source expander.
4. Summarize tab: summarize the same file.
5. Compare tab: upload a second file and compare.
6. Optional: record audio in Chat and use **Ask with recording**.

If upload or chat fails with a timeout, the Render instance may be
sleeping or still embedding; wait and retry. If health shows OpenAI
unavailable, the key is missing on Render.

## What's *not* implemented yet

- Hybrid BM25 + vector retrieval (pure vector search).
- Durable document storage across Render restarts (would need a disk or
  hosted vector DB).
- Automated tests (`tests/` is currently empty).


## What's implemented

- **`/health`** — configured LLM, ChromaDB, embeddings (never fakes "ok").
- **`/documents/upload`**, list, get, delete — extract, chunk, embed, store.
- **`POST /chat`** — RAG with source chunks.
- **`POST /documents/{id}/summarize`**, **`POST /compare`**.
- **`POST /voice/transcribe`** — local Whisper or OpenAI Whisper.
- **`frontend/app.py`** — original Streamlit UI (not rewritten).

## Roadmap

1. ✅ Project skeleton, config, health endpoint
2. ✅ Document extraction & chunking
3. ✅ Embeddings + persistent ChromaDB + SQLite metadata
4. ⬜ BM25 + hybrid retrieval (currently pure vector search)
5. ✅ LLM Q&A with citations (Ollama / Hugging Face / OpenAI)
6. ✅ Summaries & comparisons
7. ✅ Streamlit frontend
8. ✅ Voice input (browser mic + transcribe)
9. ⬜ Durable hosted storage, automated tests
10. ⬜ End-to-end verification on your Render + Streamlit Cloud accounts

