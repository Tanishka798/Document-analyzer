# Document Assistant (Local-First, Voice-Enabled RAG)

> Status: **Full stack complete** (Phases 1–3, 5–8 of the original
> 10-phase roadmap). Document ingestion, RAG chat, summarize, compare,
> voice, and a Streamlit frontend are all implemented. Not yet built:
> BM25/hybrid retrieval (pure vector search only, for now) and an
> automated test suite — see "Roadmap" below.

## What this is

A local FastAPI backend that lets you upload documents
(PDF/DOCX/PPTX/XLSX/TXT/MD), ask questions across them with cited sources,
summarize and compare documents, and interact by voice. Local development
uses [Ollama](https://ollama.com) by default; Streamlit Community Cloud can
use hosted Hugging Face Inference Providers for the LLM.

## Models and credentials

Local Ollama mode needs no API key. Streamlit Community Cloud can use a
Hugging Face token, stored in Streamlit Secrets, for hosted text generation.
Embeddings and speech recognition run in the Streamlit app.

| Capability | Library | Where the model comes from | Account/API key? |
|---|---|---|---|
| Chat/LLM answers (local) | Ollama (`qwen2.5:3b`) | `ollama pull qwen2.5:3b`, run once | **No** |
| Chat/LLM answers (cloud) | Hugging Face Inference Providers | Hosted inference | `HF_TOKEN` Streamlit secret |
| Embeddings | sentence-transformers | auto-downloaded from huggingface.co on first use | **No** (public model) |
| Speech-to-text | faster-whisper | auto-downloaded from huggingface.co on first use | **No** (public model) |
| Spoken AI replies | Browser speech synthesis | Runs in the visitor's browser | **No** |
| Optional `/voice/speak` API | Piper | Local installation | **No** |

Local models require an internet connection on first use to download public
model files. Hugging Face Inference Provider calls use your configured
account and are subject to provider availability, quota, and billing.

## What's implemented

- **`/health`** — status of the configured LLM provider, ChromaDB, and embeddings
  package (never fakes "ok").
- **`/documents/upload`, `/documents`, `/documents/{id}`, `DELETE /documents/{id}`**
  — upload a file, extract its text (PDF/DOCX/PPTX/XLSX/TXT/MD), chunk it,
  embed it, and store it in ChromaDB + a SQLite metadata row. Failures are
  recorded on the document (`status: "failed"`, `error: "..."`), not hidden.
- **`POST /chat`** — embeds your question, retrieves the most relevant
  chunks (optionally scoped to specific `document_ids`), and asks the configured LLM
  to answer using *only* that retrieved context, returning the answer plus
  the exact source chunks used.
- **`POST /documents/{id}/summarize`** — summarizes one document in full.
- **`POST /compare`** — compares 2–5 documents by summarizing each and
  asking the configured LLM to contrast them.
- **`POST /voice/transcribe`** — speech-to-text via faster-whisper.
- **`POST /voice/speak`** — text-to-speech via Piper (only if
  `TTS_ENABLED=true` in `.env`).
- **`frontend/app.py`** — a Streamlit UI on top of all of the above:
  upload/list/delete documents in the sidebar, a chat tab with citations
  and a "🔊 Read aloud" button, a summarize tab, and a compare tab. Voice
  input is a built-in microphone recorder in the chat tab; recordings can
  be submitted directly as questions. Browser speech synthesis reads replies
  aloud with pause/resume and stop controls.

## What's *not* implemented yet

- Hybrid BM25 + vector retrieval (pure vector search only right now).
- OCR is wired up in `extraction.py` but its packages
  (`pytesseract`/`pdf2image`) are commented out in `requirements.txt` —
  uncomment them and set `OCR_ENABLED=true` if you need scanned-PDF support.
- Automated tests (`tests/` is currently empty).

## Running this on Windows

Open PowerShell in the project folder:

```powershell
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate it
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# Optional OCR and Piper API packages are commented out in requirements.txt.
# For OCR, install pytesseract/pdf2image and the Tesseract + poppler apps.
# For the optional /voice/speak API, install piper-tts.

# 4. Create your local .env file from the template
Copy-Item .env.example .env

# 5. Install and start Ollama separately (one-time), in its own terminal:
#    Download from https://ollama.com, then:
ollama pull qwen2.5:3b
ollama serve

# 6. Optional: run the standalone backend for REST API usage (/docs, curl)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open **http://127.0.0.1:8000/docs** in a browser for interactive API
docs — you can upload a document and try `/chat`, `/summarize`, and
`/compare` directly from there. **http://127.0.0.1:8000/health** shows status
for the configured LLM provider, ChromaDB, and the embeddings package.

## 7. Run the Streamlit frontend

The Streamlit UI calls the FastAPI app in-process, so a separate Uvicorn
terminal is not required. From the project root:

```powershell
.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

This opens **http://localhost:8501** in your browser automatically. From
there: upload documents in the sidebar, select which ones to scope a
question to (or leave none selected to search everything), and use the
Chat / Summarize / Compare tabs. To ask by voice, click the microphone
widget in the Chat tab, record, then choose "Ask with recording" to
transcribe the question and submit it directly.

AI replies can be read aloud using browser speech synthesis; no Piper setup
is needed for the Streamlit UI.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Do not commit `.env`, `.venv`,
  or `data/`.
2. In Streamlit Community Cloud, create an app from that repository and set
  the main file path to `frontend/app.py`.
3. In the app's **Settings → Secrets**, add:

  ```toml
  LLM_PROVIDER = "huggingface"
  HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
  HF_TOKEN = "your Hugging Face token"
  WHISPER_DEVICE = "cpu"
  ```

  Keep the token in Streamlit Secrets only. The app calls FastAPI in-process,
  so you do not need to deploy a separate backend or set a backend URL.
4. The app has one shared document store and no user authentication. Anyone
  who can access the public app can use that store; do not upload private
  documents. Cloud disk is temporary, so uploaded documents can disappear
  when the app restarts or sleeps.
5. The first use downloads embedding and Whisper models. Hugging Face
  Inference Provider calls may have quota or billing limits.

## Roadmap

1. ✅ Project skeleton, config, health endpoint
2. ✅ Document extraction & chunking
3. ✅ Embeddings + persistent ChromaDB + SQLite metadata
4. ⬜ BM25 + hybrid retrieval (currently pure vector search)
5. ✅ Ollama integration + grounded Q&A with citations
6. ✅ Summaries & comparisons
7. ✅ Streamlit frontend
8. ✅ Voice input/output (built and smoke-tested via Streamlit's AppTest;
   not yet hardware-tested with a real mic on your machine — that's the
   first thing to try)
9. ⬜ Performance, security hardening, automated tests
10. ⬜ Final README polish, end-to-end verification on your machine
