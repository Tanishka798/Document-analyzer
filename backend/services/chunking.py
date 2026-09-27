"""
chunking.py
-----------
Why this module exists:
    Splits extracted document text into overlapping chunks small enough
    to embed and to fit into the LLM's context window, while keeping
    enough overlap that an answer-relevant sentence sitting on a chunk
    boundary still shows up whole in at least one chunk.

Why word count, not a real tokenizer:
    A true tokenizer (tiktoken, the model's own tokenizer) would need to
    download tokenizer files from the internet on first use, and the
    exact tokenizer doesn't matter much for chunk sizing — being off by
    20% on chunk size has no real effect on retrieval quality. Whitespace
    word count is a good-enough, dependency-free approximation (roughly
    0.75 tokens per word for English), so CHUNK_SIZE_TOKENS=500 in .env
    is treated as "500 words" here. This is documented so it's not a
    silent surprise later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    text: str


def chunk_text(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    """
    Split text into overlapping chunks of approximately chunk_size_tokens
    words each, sliding forward by (chunk_size_tokens - overlap_tokens)
    words between chunks.
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < chunk_size_tokens")

    words = text.split()
    if not words:
        return []

    stride = chunk_size_tokens - overlap_tokens
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(start + chunk_size_tokens, len(words))
        chunk_words = words[start:end]
        chunks.append(Chunk(index=index, text=" ".join(chunk_words)))
        index += 1
        if end == len(words):
            break
        start += stride

    return chunks
