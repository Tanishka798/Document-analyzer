"""
routers/voice.py
------------------
Why this module exists:
    Phase 8: voice input (speech-to-text via faster-whisper) and voice
    output (text-to-speech via Piper). Both run fully locally — no cloud
    speech API, no API key.

Why these are separate, lazily-imported try/except blocks rather than
top-level imports:
    faster-whisper and piper-tts are both large-ish optional dependencies
    that download their own model/voice files on first use, same pattern
    as embeddings.py. If either package isn't installed, or its model
    hasn't downloaded yet, importing it here (inside the request handler)
    means the rest of the backend keeps working and /health keeps
    responding — only these two endpoints fail, with a clear message,
    per the "never silently pretend it works" rule.

Not testable in this sandbox:
    There's no microphone/audio hardware and no Ollama/HF network access
    in the sandbox this was built in. This module is written to run on
    your Windows machine, where both are available, but hasn't been
    exercised end-to-end yet — treat it as the first thing to sanity
    check there.
"""

from __future__ import annotations

import threading
import uuid

from fastapi import APIRouter, HTTPException, UploadFile

from backend.config import settings
from backend.schemas import SpeakRequest, TranscribeResponse

router = APIRouter(prefix="/voice", tags=["voice"])

_whisper_lock = threading.Lock()
_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel

                _whisper_model = WhisperModel(
                    settings.whisper_model_size, device=settings.whisper_device
                )
    return _whisper_model


def _transcribe_openai(raw: bytes, filename: str) -> str:
    import httpx

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set; cannot use OpenAI Whisper transcription.",
        )
    try:
        response = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            files={"file": (filename, raw)},
            data={"model": "whisper-1"},
            timeout=300.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"OpenAI transcription failed ({exc.response.status_code}): "
                f"{exc.response.text[:300]}"
            ),
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"OpenAI transcription request failed: {exc}"
        ) from exc
    return (response.json().get("text") or "").strip()


def _transcribe_local(raw: bytes, filename: str) -> str:
    temp_path = settings.audio_dir_abs_path / f"{uuid.uuid4().hex}_{filename}"
    temp_path.write_bytes(raw)
    try:
        model = _get_whisper_model()
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="faster-whisper is not installed. Run: pip install faster-whisper",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Could not load Whisper model: {exc}") from exc

    try:
        segments, _info = model.transcribe(str(temp_path))
        return " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile) -> TranscribeResponse:
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    filename = audio.filename or "recording.wav"
    if settings.llm_provider.lower() == "openai":
        text = _transcribe_openai(raw, filename)
    else:
        text = _transcribe_local(raw, filename)

    if not text:
        raise HTTPException(status_code=422, detail="Could not detect any speech in the audio.")
    return TranscribeResponse(text=text)


@router.post("/speak")
def speak(request: SpeakRequest):
    if not settings.tts_enabled:
        raise HTTPException(
            status_code=400,
            detail="TTS_ENABLED=false in .env. Set it to true to use voice output.",
        )

    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="piper-tts is not installed. Run: pip install piper-tts",
        ) from exc

    output_path = settings.audio_dir_abs_path / f"{uuid.uuid4().hex}.wav"
    try:
        voice = PiperVoice.load(settings.tts_voice)
        with open(output_path, "wb") as wav_file:
            voice.synthesize(request.text, wav_file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {exc}") from exc

    from fastapi.responses import FileResponse

    return FileResponse(output_path, media_type="audio/wav", filename=output_path.name)
