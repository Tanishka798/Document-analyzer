"""
embeddings.py
-------------
Why this module exists:
    Wraps sentence-transformers behind a lazy singleton so the (fairly
    slow, ~100-400MB) model is loaded from disk exactly once per backend
    process, on first actual use — not at import time, so `/health` and
    other unrelated endpoints stay fast even before any embedding has
    happened.

On API keys / network:
    sentence-transformers/all-MiniLM-L6-v2 is a public Hugging Face model.
    The *first* time this runs on your machine, it downloads ~90MB of
    model files from huggingface.co automatically — no account, login, or
    API key required. After that first download it's cached locally
    (in your Hugging Face cache dir) and works fully offline. This
    sandbox has no access to huggingface.co, so this module is untested
    end-to-end here; it will download and run normally on your Windows
    machine, which does have internet access.
"""

from __future__ import annotations

import threading

from backend.config import settings

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:  # re-check inside the lock
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(settings.embedding_model)
    return _model


def sanitize_text(text: str) -> str:
    """Make extracted text safe for the Hugging Face tokenizer.

    PDF extractors (especially textbooks with figures) often emit NUL
    bytes, C0 control chars, and unpaired surrogates. The fast tokenizer
    then raises TypeError: TextEncodeInput must be Union[...] because it
    sees an empty/invalid C string rather than Python text.
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", " ")
    text = "".join(ch if (ch >= " " or ch in "\n\t\r") else " " for ch in text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return " ".join(text.split())


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunk texts (used at ingestion time)."""
    if not texts:
        return []
    # Never pass empty strings: sentence-transformers strips each item and
    # the fast tokenizer rejects "" with TextEncodeInput TypeError.
    cleaned = [sanitize_text(t) or " " for t in texts]
    model = _get_model()
    vectors = model.encode(
        cleaned,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=32,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single query string (used at retrieval time)."""
    return embed_texts([text])[0]
