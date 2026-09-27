"""
ollama_client.py
----------------
Why this module exists:
    Chat, summarize, and compare share one LLM generation function. Local
    development can use Ollama, Hugging Face, or OpenAI.

Credentials:
    Ollama mode needs no API key. Hugging Face mode reads HF_TOKEN from the
    environment.

Ollama uses /api/generate because the RAG prompt is assembled before the
request; Hugging Face uses chat_completion with the same system and user
prompt content.
"""

from __future__ import annotations

import httpx
from huggingface_hub import InferenceClient

from backend.config import settings


class OllamaError(Exception):
    """Raised when the configured LLM provider fails."""


def generate(prompt: str, system: str | None = None) -> str:
    """Generate text through the selected provider and normalize errors."""
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return _generate_openai(prompt, system)
    if provider == "huggingface":
        return _generate_huggingface(prompt, system)
    if provider != "ollama":
        raise OllamaError(
            f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. "
            "Use 'ollama', 'huggingface', or 'openai'."
        )
    return _generate_ollama(prompt, system)


def _generate_openai(prompt: str, system: str | None) -> str:
    if not settings.openai_api_key:
        raise OllamaError(
            "OPENAI_API_KEY is missing. Set it in your local environment or .env file."
        )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openai_chat_model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": settings.ollama_max_output_tokens,
            },
            timeout=300.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:300]
        raise OllamaError(
            f"OpenAI chat failed (status {exc.response.status_code}): {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaError(f"OpenAI chat request failed: {exc}") from exc

    text = response.json().get("choices", [{}])[0].get("message", {}).get("content")
    if not text:
        raise OllamaError("OpenAI returned an empty chat response.")
    return text


def _generate_huggingface(prompt: str, system: str | None) -> str:
    if not settings.hf_token:
        raise OllamaError("HF_TOKEN is missing. Add it as a Hugging Face Space secret.")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        client = InferenceClient(
            model=settings.hf_model_id,
            provider="auto",
            token=settings.hf_token,
            timeout=300.0,
        )
        response = client.chat_completion(
            messages,
            max_tokens=settings.ollama_max_output_tokens,
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001 - normalize provider errors for API callers
        raise OllamaError(f"Hugging Face Inference failed: {exc}") from exc

    text = response.choices[0].message.content
    if not text:
        raise OllamaError("Hugging Face Inference returned an empty response.")
    return text


def _generate_ollama(prompt: str, system: str | None) -> str:
    payload: dict = {
        "model": settings.ollama_chat_model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": settings.ollama_max_output_tokens},
    }
    if system:
        payload["system"] = system

    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
            timeout=300.0,
        )
        response.raise_for_status()
    except httpx.ConnectError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {settings.ollama_base_url}. "
            "Is it installed and running? (ollama serve)"
        ) from exc
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:300]
        raise OllamaError(
            f"Ollama returned an error (status {exc.response.status_code}): {detail}. "
            f"Confirm the model '{settings.ollama_chat_model}' is pulled "
            f"(ollama pull {settings.ollama_chat_model})."
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc

    data = response.json()
    text = data.get("response", "")
    if not text:
        raise OllamaError("Ollama returned an empty response.")
    return text


def trim_to_budget(text: str, budget_tokens: int) -> str:
    """Approximate-token-trim a block of context text down to
    OLLAMA_CONTEXT_BUDGET_TOKENS, using the same words-as-tokens
    approximation as chunking.py (see that module for why)."""
    words = text.split()
    if len(words) <= budget_tokens:
        return text
    return " ".join(words[:budget_tokens]) + "\n[...context truncated to fit budget...]"
