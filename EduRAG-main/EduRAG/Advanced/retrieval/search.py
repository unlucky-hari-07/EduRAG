"""
retrieval/search.py

Query-time retrieval: embed the query, search the FAISS index for candidate
chunks, and optionally restrict results to a single document.

Since the underlying index is a flat (non-partitioned) FAISS index, scoping
to one document is done by over-fetching candidates and filtering by
document_id in Python. This is simple and correct at the scale this project
targets; a production system with many documents would use per-document
indexes or a filtered vector store instead.
"""

from typing import Dict, List, Optional

from backend.services.embeddings import embed_texts
from retrieval.index import get_store


def search_chunks(
    query: str,
    candidate_k: int,
    document_id: Optional[str] = None,
) -> List[Dict]:
    """
    Return up to `candidate_k` chunk metadata dicts most similar to `query`.

    If `document_id` is provided, over-fetches and filters so the caller
    still gets up to `candidate_k` matches from that document when available.
    """
    store = get_store()
    query_vector = embed_texts([query])[0]

    if document_id is None:
        return store.search(query_vector, top_k=candidate_k)

    # Over-fetch to compensate for post-filtering by document_id.
    fetch_k = min(candidate_k * 5, max(candidate_k, 50))
    raw_results = store.search(query_vector, top_k=fetch_k)
    filtered = [r for r in raw_results if r.get("document_id") == document_id]
    return filtered[:candidate_k]
