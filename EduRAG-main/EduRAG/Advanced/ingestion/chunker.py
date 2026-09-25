"""
ingestion/chunker.py

Splits cleaned document text into overlapping chunks suitable for embedding
and retrieval. Chunking is done per-page so page-number metadata (when
available, e.g. from PDFs) is preserved on every chunk.
"""

from dataclasses import dataclass
from typing import List, Optional

from .cleaner import clean_text
from .loaders import PageText


@dataclass
class Chunk:
    text: str
    page_number: Optional[int]


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split a single block of text into overlapping character-based chunks.

    Splitting happens on whitespace boundaries where possible so words are
    not cut in half.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to break on a space near the end of the window, not mid-word.
        if end < text_length:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Move forward, leaving `overlap` characters of context for the next chunk.
        start = max(end - overlap, start + 1)

    return chunks


def chunk_pages(
    pages: List[PageText],
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """Clean and chunk a document's pages into a flat list of Chunk objects."""
    all_chunks: List[Chunk] = []

    for page_number, raw_text in pages:
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        for piece in _split_text(cleaned, chunk_size=chunk_size, overlap=overlap):
            all_chunks.append(Chunk(text=piece, page_number=page_number))

    return all_chunks
