"""
rag/chunker.py

Reusable, configurable chunking for the RAG pipeline. Splits cleaned text
into overlapping chunks measured in whole words (a simple, dependency-free
proxy for tokens) so chunk size and overlap are easy to reason about and
tune.

Chunking is done per-page so page-number metadata (available for PDFs) is
preserved on every resulting chunk.

Note on chunk size: the ~220-word default (configured in utils/config.py)
is a reasonable starting point, not a universally optimal value. Retrieval
quality is sensitive to this choice - shorter chunks give more precise
matches but less surrounding context; longer chunks give more context but
can dilute the embedding's focus. Tune CHUNK_SIZE_WORDS and
CHUNK_OVERLAP_WORDS for your own documents and observed retrieval quality.
"""

from dataclasses import dataclass
from typing import List, Optional

from rag.cleaner import clean_text
from rag.loaders import PageText


@dataclass
class Chunk:
    text: str
    page_number: Optional[int]


def _split_words(words: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    """Split a list of words into overlapping windows of `chunk_size` words."""
    if len(words) <= chunk_size:
        return [words] if words else []

    if overlap >= chunk_size:
        overlap = max(chunk_size // 4, 1)  # guard against a misconfigured overlap

    windows: List[List[str]] = []
    start = 0
    total = len(words)
    step = chunk_size - overlap

    while start < total:
        end = min(start + chunk_size, total)
        windows.append(words[start:end])
        if end >= total:
            break
        start += step

    return windows


def chunk_pages(
    pages: List[PageText],
    chunk_size: int = 220,
    overlap: int = 40,
) -> List[Chunk]:
    """Clean and chunk a document's pages into a flat list of Chunk objects."""
    all_chunks: List[Chunk] = []

    for page_number, raw_text in pages:
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        words = cleaned.split()
        for window in _split_words(words, chunk_size=chunk_size, overlap=overlap):
            chunk_text = " ".join(window).strip()
            if chunk_text:
                all_chunks.append(Chunk(text=chunk_text, page_number=page_number))

    return all_chunks
