"""
rag/vector_store.py

An in-memory FAISS vector store paired with a metadata list, scoped to one
Streamlit session (held in st.session_state by app.py). FAISS itself only
stores vectors and returns integer positions - all human-readable metadata
(document_id, document_name, page_number, chunk_id, chunk_text) lives here
in lockstep with the index, so a retrieved result can always be mapped back
to exactly where it came from.

Supports multiple documents in the same index, with document-scoped search
so retrieval never mixes results across documents when a document IS
selected.
"""

from typing import Dict, List, Optional

import numpy as np


class VectorStoreError(Exception):
    """Raised when a vector store operation fails unexpectedly."""


class VectorStore:
    def __init__(self, dimension: int):
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError(
                "FAISS is required for vector search but is not installed."
            ) from exc

        self._faiss = faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict] = []
        self._documents: Dict[str, Dict] = {}  # document_id -> {document_name, chunk_count}

    def add_document(self, document_id: str, document_name: str, chunks, vectors: np.ndarray) -> None:
        """Add a document's chunks and their embeddings to the index."""
        if len(chunks) != len(vectors):
            raise VectorStoreError("Number of chunks and vectors must match.")

        try:
            self.index.add(vectors.astype("float32"))
        except Exception as exc:
            raise VectorStoreError(f"Failed to add vectors to the FAISS index: {exc}") from exc

        for i, chunk in enumerate(chunks):
            self.metadata.append(
                {
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_id": f"{document_id}-{i}",
                    "chunk_text": chunk.text,
                    "page_number": chunk.page_number,
                }
            )

        self._documents[document_id] = {
            "document_id": document_id,
            "document_name": document_name,
            "chunk_count": len(chunks),
        }

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        document_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Return up to `top_k` chunk metadata dicts most similar to
        query_vector. If document_id is given, results are restricted to
        that document by over-fetching candidates and filtering.
        """
        if self.index.ntotal == 0:
            return []

        fetch_k = self.index.ntotal
        if document_id is not None:
            fetch_k = min(self.index.ntotal, max(top_k * 5, 50))
        else:
            fetch_k = min(top_k, self.index.ntotal)

        try:
            scores, indices = self.index.search(
                query_vector.astype("float32").reshape(1, -1), fetch_k
            )
        except Exception as exc:
            raise VectorStoreError(f"FAISS search failed: {exc}") from exc

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            if document_id is None or entry["document_id"] == document_id:
                results.append(entry)

        return results[:top_k]

    def list_documents(self) -> List[Dict]:
        return list(self._documents.values())

    def remove_document(self, document_id: str) -> bool:
        """Remove a document's chunks and rebuild the index."""
        if document_id not in self._documents:
            return False

        keep_positions = [i for i, m in enumerate(self.metadata) if m["document_id"] != document_id]

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

        del self._documents[document_id]
        return True

    @property
    def total_chunks(self) -> int:
        return len(self.metadata)
