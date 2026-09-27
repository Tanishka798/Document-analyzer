"""
config.py
---------
Why this module exists:
    Every other backend module needs settings (paths, model names, limits).
    Instead of each module reading environment variables itself (which makes
    it hard to see all configuration in one place, and easy to typo a
    variable name), we load everything once here into a single, typed
    `Settings` object and import that object everywhere else.

What it receives / returns:
    Reads from a `.env` file (via python-dotenv, wired in by pydantic-settings)
    and from real OS environment variables. Produces a validated `Settings`
    instance. If a value is missing, a sensible default is used instead of
    crashing, since this is meant to work "out of the box" after copying
    .env.example to .env.

Non-obvious design decisions:
    - We use `pydantic-settings` (not plain os.environ.get calls) because it
      gives us type validation for free: e.g. RETRIEVAL_TOP_K will be parsed
      as an int, and a bad value in .env fails fast with a clear error
      instead of causing a confusing bug three modules later.
    - Paths are stored as plain strings from the environment, then resolved
      to absolute Path objects here. This matters for security later
      (Section 14 of the spec): every module that touches the filesystem
      should use these resolved, known-safe paths rather than building its
      own paths from user input.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The project root is the parent of the "backend" folder this file lives in.
# We compute it once here so relative paths in .env (like "./data/chroma_db")
# always resolve the same way, regardless of which directory a script is
# launched from (this matters because FastAPI and Streamlit are started as
# two separate processes, possibly from different working directories).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Typed application settings, populated from environment variables / .env.

    Field names intentionally match the variable names in .env.example so the
    mapping between the two is obvious.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of erroring
    )

    # --- Backend server ---
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    log_level: str = "INFO"
    # Comma-separated origins. Leave empty for Streamlit server-side calls
    # (those are not browser CORS). Set only if a browser will call this API.
    cors_origins: str = ""

    # --- LLM provider ---
    # ollama (local), huggingface, or openai.
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen2.5:3b"
    ollama_max_output_tokens: int = 512
    ollama_context_budget_tokens: int = 3000
    hf_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_token: str = ""
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Embeddings ---
    # "local" uses sentence-transformers.
    # "openai" uses the OpenAI embeddings API (same OPENAI_API_KEY as chat).
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage paths (kept as strings from env, resolved below) ---
    chroma_db_path: str = "./data/chroma_db"
    sqlite_db_path: str = "./data/metadata.db"
    cache_dir: str = "./data/cache"
    audio_dir: str = "./data/audio"

    # --- Chunking / retrieval ---
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 80
    retrieval_top_k: int = 8

    # --- Ingestion limits ---
    max_file_size_mb: int = 50
    max_concurrent_extractions: int = 2

    # --- OCR ---
    ocr_enabled: bool = False
    tesseract_cmd: str = ""

    # --- Voice input ---
    whisper_model_size: str = "small"
    whisper_device: str = "cpu"

    # --- Voice output ---
    tts_enabled: bool = False
    tts_voice: str = "en_US-lessac-medium"

    # -- Resolved absolute paths (computed, not read directly from env) --
    @property
    def chroma_db_abs_path(self) -> Path:
        return self._resolve(self.chroma_db_path)

    @property
    def sqlite_db_abs_path(self) -> Path:
        return self._resolve(self.sqlite_db_path)

    @property
    def cache_dir_abs_path(self) -> Path:
        return self._resolve(self.cache_dir)

    @property
    def audio_dir_abs_path(self) -> Path:
        return self._resolve(self.audio_dir)

    def _resolve(self, relative_or_absolute: str) -> Path:
        """Resolve a configured path against the project root, not the
        current working directory, so behavior is consistent no matter
        where a process is launched from."""
        path = Path(relative_or_absolute)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def ensure_data_dirs_exist(self) -> None:
        """Create the local data directories if they don't exist yet.
        Called once at backend startup. Safe to call repeatedly."""
        for path in (
            self.chroma_db_abs_path,
            self.cache_dir_abs_path,
            self.audio_dir_abs_path,
        ):
            path.mkdir(parents=True, exist_ok=True)
        # SQLite path is a file, not a directory — just ensure its parent exists.
        self.sqlite_db_abs_path.parent.mkdir(parents=True, exist_ok=True)


# Single shared instance imported by the rest of the backend.
# (Loading settings is cheap and side-effect-free, so a module-level
# singleton is simpler than dependency-injecting it everywhere for Phase 1.)
settings = Settings()
