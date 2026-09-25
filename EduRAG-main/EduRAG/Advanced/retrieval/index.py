"""
retrieval/index.py

Thin wrapper around a FAISS index plus a JSON metadata sidecar. FAISS only
stores vectors and returns integer positions - all human-readable metadata
(document name, page number, chunk text, chunk id) is tracked here and kept
in lockstep with the index so nothing has to be re-derived or guessed later.

Persisted to disk so the index survives an application restart.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class FaissMetadataStore:
    """
    Manages a FAISS IndexFlatIP (cosine similarity via normalized vectors)
    alongside a parallel metadata list, both persisted to disk.

    Thread-safe for the simple read/append patterns used by this application.
    """

    def __init__(self, index_path: Path, metadata_path: Path, dimension: int):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension
        self._lock = threading.Lock()

        import faiss  # local import so importing this module doesn't require faiss at import time

        self._faiss = faiss

        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata: List[Dict] = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(dimension)
            self.metadata = []

    def _persist(self) -> None:
        self._faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    def add(self, vectors: np.ndarray, metadata_entries: List[Dict]) -> None:
        """Add normalized vectors and their matching metadata dicts."""
        if len(vectors) != len(metadata_entries):
            raise ValueError("vectors and metadata_entries must be the same length")

        with self._lock:
            self.index.add(vectors.astype("float32"))
            self.metadata.extend(metadata_entries)
            self._persist()

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        """Return the top_k metadata entries most similar to query_vector."""
        with self._lock:
            if self.index.ntotal == 0:
                return []

            top_k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(
                query_vector.astype("float32").reshape(1, -1), top_k
            )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results

    def remove_document(self, document_id: str) -> int:
        """
        Remove all chunks belonging to a document_id and rebuild the index.

        FAISS's flat index doesn't support efficient in-place deletion, so we
        rebuild from the remaining vectors. Fine for a project of this scale.
        """
        with self._lock:
            keep_positions = [
                i for i, m in enumerate(self.metadata) if m.get("document_id") != document_id
            ]
            removed_count = len(self.metadata) - len(keep_positions)

            if removed_count == 0:
                return 0

            if keep_positions:
                all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
                kept_vectors = all_vectors[keep_positions]
                new_index = self._faiss.IndexFlatIP(self.dimension)
                new_index.add(kept_vectors.astype("float32"))
                self.index = new_index
                self.metadata = [self.metadata[i] for i in keep_positions]
            else:
                self.index = self._faiss.IndexFlatIP(self.dimension)
                self.metadata = []

            self._persist()
            return removed_count


_store_instance: Optional[FaissMetadataStore] = None


def get_store() -> FaissMetadataStore:
    """Return a process-wide singleton FaissMetadataStore instance."""
    global _store_instance
    if _store_instance is None:
        from backend import config

        _store_instance = FaissMetadataStore(
            index_path=config.FAISS_INDEX_PATH,
            metadata_path=config.METADATA_PATH,
            dimension=config.EMBEDDING_DIMENSION,
        )
    return _store_instance
