"""
backend/services/ingestion.py

Orchestrates the document ingestion pipeline:

    Upload -> Validate -> Extract -> Clean -> Chunk -> Embed -> Index -> Register

This module ties together the pure ingestion logic (ingestion/) with the
embedding and retrieval-index services, and maintains a simple JSON registry
of uploaded documents (used by GET /documents and DELETE /documents/{id}).
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from backend import config
from backend.services.embeddings import embed_texts
from ingestion.chunker import chunk_pages
from ingestion.loaders import DocumentLoadError, load_document
from retrieval.index import get_store

_registry_lock = threading.Lock()


class DocumentValidationError(Exception):
    """Raised when an uploaded file fails validation before ingestion."""


def validate_upload(filename: str, file_size: int) -> None:
    """Validate file type and size before any processing happens."""
    if file_size == 0:
        raise DocumentValidationError("The uploaded file is empty.")

    if file_size > config.MAX_FILE_SIZE_BYTES:
        max_mb = config.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise DocumentValidationError(
            f"The uploaded file exceeds the maximum allowed size of {max_mb:.0f} MB."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise DocumentValidationError(
            f"Unsupported file type '{suffix}'. Allowed types: {allowed}."
        )


def _load_registry() -> List[Dict]:
    if not config.DOCUMENTS_REGISTRY_PATH.exists():
        return []
    with open(config.DOCUMENTS_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: List[Dict]) -> None:
    with open(config.DOCUMENTS_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def list_documents() -> List[Dict]:
    """Return all registered documents (metadata only, no chunk text)."""
    with _registry_lock:
        return _load_registry()


def get_document(document_id: str) -> Optional[Dict]:
    with _registry_lock:
        registry = _load_registry()
    for doc in registry:
        if doc["document_id"] == document_id:
            return doc
    return None


def ingest_document(file_path: Path, original_filename: str) -> Dict:
    """
    Run the full ingestion pipeline on a saved file and register it.

    Returns the new document's registry entry (document_id, document_name,
    chunk_count, ingested_at).
    """
    try:
        pages = load_document(file_path)
    except DocumentLoadError as exc:
        raise DocumentValidationError(str(exc)) from exc

    chunks = chunk_pages(
        pages,
        chunk_size=config.CHUNK_SIZE_CHARS,
        overlap=config.CHUNK_OVERLAP_CHARS,
    )

    if not chunks:
        raise DocumentValidationError(
            "No usable text could be extracted from this document after cleaning."
        )

    document_id = str(uuid.uuid4())
    ingested_at = datetime.now(timezone.utc).isoformat()

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)

    metadata_entries = [
        {
            "document_id": document_id,
            "document_name": original_filename,
            "chunk_id": f"{document_id}-{i}",
            "chunk_text": chunk.text,
            "page_number": chunk.page_number,
            "ingested_at": ingested_at,
        }
        for i, chunk in enumerate(chunks)
    ]

    store = get_store()
    store.add(vectors, metadata_entries)

    entry = {
        "document_id": document_id,
        "document_name": original_filename,
        "chunk_count": len(chunks),
        "ingested_at": ingested_at,
    }

    with _registry_lock:
        registry = _load_registry()
        registry.append(entry)
        _save_registry(registry)

    return entry


def delete_document(document_id: str) -> bool:
    """
    Remove a document's chunks from the FAISS index and its registry entry.

    Returns True if the document existed and was removed, False otherwise.
    """
    with _registry_lock:
        registry = _load_registry()
        remaining = [d for d in registry if d["document_id"] != document_id]

        if len(remaining) == len(registry):
            return False

        store = get_store()
        store.remove_document(document_id)

        _save_registry(remaining)

    return True
