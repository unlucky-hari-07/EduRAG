"""
backend/services/embeddings.py

Wraps the Sentence Transformers embedding model. Isolated in its own module
so the embedding model could be swapped without touching ingestion or
retrieval code.
"""

from typing import List

import numpy as np

_model = None


def _get_model():
    """Lazily load the embedding model (heavy import, loaded once per process)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        from backend import config

        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts and return L2-normalized float32 vectors, suitable
    for cosine similarity search via a FAISS IndexFlatIP.
    """
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vectors = vectors.astype("float32")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero for a degenerate zero vector
    normalized = vectors / norms

    return normalized
