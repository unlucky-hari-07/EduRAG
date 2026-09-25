"""
retrieval/rerank.py

Reranking stage: takes the candidate chunks returned by FAISS similarity
search and reorders them using a cross-encoder, which scores the
(question, chunk) pair jointly rather than comparing independent embeddings.

Why rerank at all: a bi-encoder (the embedding model used for FAISS search)
scores the query and each chunk *independently* and compares vectors, which
is fast but loses fine-grained interaction between the two texts. A
cross-encoder reads the query and chunk together and tends to be more
accurate at judging true relevance, at the cost of being too slow to run
over an entire corpus - hence using it only to re-score a small candidate
set retrieved by FAISS. This project has not run a formal benchmark
comparing the two, so no numeric precision improvement is claimed here; the
justification above is architectural, not measured.
"""

from typing import Dict, List, Optional

_cross_encoder = None


def _get_cross_encoder():
    """Lazily load the cross-encoder model (heavy import, loaded once)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        from backend import config

        _cross_encoder = CrossEncoder(config.RERANKER_MODEL_NAME)
    return _cross_encoder


def rerank_chunks(
    question: str,
    candidates: List[Dict],
    top_k: int,
) -> List[Dict]:
    """
    Rerank candidate chunk dicts (each must have a "text" key) against the
    question, and return the top_k, each annotated with a "rerank_score".
    """
    if not candidates:
        return []

    model = _get_cross_encoder()
    pairs = [(question, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
