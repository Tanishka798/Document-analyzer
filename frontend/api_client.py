"""
api_client.py
--------------
Why this module exists:
    Keeps every backend call in one place. Locally, with no BACKEND_URL,
    the FastAPI app still runs in-process through TestClient. When
    BACKEND_URL is set, every call is a real HTTP request to the backend.

Error handling:
    Every function raises APIError with a message that's already safe
    and useful to show directly in st.error() — app.py never needs to
    inspect requests exceptions or response bodies itself.
"""

from __future__ import annotations

import os
from functools import lru_cache

import httpx

TIMEOUT_SHORT = 15
TIMEOUT_LONG = 300  # LLM calls can be slow on the first request or on CPU
TIMEOUT_TRANSCRIBE = 600  # first use may download and initialize the Whisper model
TIMEOUT_UPLOAD = 600  # large PDFs: extract + embed hundreds of chunks on CPU


class APIError(Exception):
    """Raised for backend failures with a message safe to show to the user."""


def _streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            return str(value).strip() if value is not None else None
    except Exception:  # noqa: BLE001 - secrets.toml is optional
        return None
    return None


def backend_base_url() -> str:
    """Backend origin for a separate FastAPI process.

    Streamlit secrets take precedence over the environment when you run the
    UI and API as separate local processes.
    """
    secret = _streamlit_secret("BACKEND_URL")
    if secret:
        return secret.rstrip("/")
    return os.environ.get("BACKEND_URL", "").strip().rstrip("/")


def _apply_streamlit_secrets() -> None:
    try:
        import streamlit as st

        from backend.config import settings

        secrets = st.secrets
        setting_names = {
            "LLM_PROVIDER": "llm_provider",
            "HF_MODEL_ID": "hf_model_id",
            "HF_TOKEN": "hf_token",
            "OPENAI_API_KEY": "openai_api_key",
            "EMBEDDING_PROVIDER": "embedding_provider",
            "WHISPER_MODEL_SIZE": "whisper_model_size",
            "WHISPER_DEVICE": "whisper_device",
        }
        for secret_name, setting_name in setting_names.items():
            if secret_name in secrets:
                setattr(settings, setting_name, secrets[secret_name])
    except Exception:  # noqa: BLE001 - in-process mode only
        return


@lru_cache(maxsize=1)
def _inprocess_client():
    _apply_streamlit_secrets()

    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    return client


@lru_cache(maxsize=1)
def _http_client() -> httpx.Client:
    return httpx.Client(base_url=backend_base_url(), timeout=TIMEOUT_LONG)


def _raise_for(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise APIError(f"Backend error ({response.status_code}): {detail}")


def _request(method: str, path: str, **kwargs) -> httpx.Response:
    timeout = kwargs.pop("timeout", TIMEOUT_LONG)
    try:
        if backend_base_url():
            response = _http_client().request(method, path, timeout=timeout, **kwargs)
        else:
            kwargs.pop("timeout", None)
            response = _inprocess_client().request(method, path, **kwargs)
    except httpx.TimeoutException as exc:
        raise APIError("The backend took too long to respond (timed out).") from exc
    except Exception as exc:  # noqa: BLE001 - normalize startup and ASGI errors
        raise APIError(f"Backend request failed: {exc}") from exc
    _raise_for(response)
    return response


def health() -> dict:
    return _request("GET", "/health", timeout=TIMEOUT_SHORT).json()


def list_documents() -> list[dict]:
    return _request("GET", "/documents", timeout=TIMEOUT_SHORT).json()["documents"]


def upload_document(file_bytes: bytes, filename: str) -> dict:
    files = {"file": (filename, file_bytes)}
    return _request("POST", "/documents/upload", files=files, timeout=TIMEOUT_UPLOAD).json()


def delete_document(document_id: str) -> dict:
    return _request("DELETE", f"/documents/{document_id}", timeout=TIMEOUT_SHORT).json()


def chat(query: str, document_ids: list[str] | None = None) -> dict:
    payload = {"query": query, "document_ids": document_ids}
    return _request("POST", "/chat", json=payload, timeout=TIMEOUT_LONG).json()


def summarize(document_id: str) -> dict:
    return _request(
        "POST", f"/documents/{document_id}/summarize", timeout=TIMEOUT_LONG
    ).json()


def compare(document_ids: list[str]) -> dict:
    payload = {"document_ids": document_ids}
    return _request("POST", "/compare", json=payload, timeout=TIMEOUT_LONG).json()


def transcribe(audio_bytes: bytes, filename: str = "recording.wav") -> str:
    files = {"audio": (filename, audio_bytes)}
    return _request(
        "POST", "/voice/transcribe", files=files, timeout=TIMEOUT_TRANSCRIBE
    ).json()["text"]


def speak(text: str) -> bytes:
    """Returns raw WAV bytes."""
    response = _request(
        "POST", "/voice/speak", json={"text": text}, timeout=TIMEOUT_LONG
    )
    return response.content
