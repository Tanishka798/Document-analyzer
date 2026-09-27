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
    cleaned = [sanitize_text(t) or " " for t in texts]
    if settings.embedding_provider.lower() == "openai":
        return _embed_openai(cleaned)
    model = _get_model()
    vectors = model.encode(
        cleaned,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=32,
    )
    return [v.tolist() for v in vectors]


def _embed_openai(texts: list[str]) -> list[list[float]]:
    if not settings.openai_api_key:
        raise RuntimeError(
            "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set. "
            "Add the key as a Render environment variable."
        )
    import httpx

    vectors: list[list[float]] = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": settings.openai_embedding_model, "input": batch},
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"OpenAI embeddings failed (status {exc.response.status_code}): "
                f"{exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenAI embeddings request failed: {exc}") from exc
        data = response.json().get("data") or []
        data.sort(key=lambda item: item.get("index", 0))
        vectors.extend(item["embedding"] for item in data)
    if len(vectors) != len(texts):
        raise RuntimeError("OpenAI embeddings returned a different number of vectors than inputs.")
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string (used at retrieval time)."""
    return embed_texts([text])[0]
