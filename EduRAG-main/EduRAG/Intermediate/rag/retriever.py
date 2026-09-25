"""
rag/retriever.py

Query-time retrieval: embed the user's question and search a VectorStore
for the most relevant chunks, optionally scoped to a single document.
"""

from typing import Dict, List, Optional

from rag.embeddings import embed_texts
from rag.vector_store import VectorStore


def retrieve_chunks(
    store: VectorStore,
    question: str,
    top_k: int,
    document_id: Optional[str] = None,
) -> List[Dict]:
    """Return up to `top_k` chunk metadata dicts most relevant to `question`."""
    query_vector = embed_texts([question])[0]
    return store.search(query_vector, top_k=top_k, document_id=document_id)


def build_context(chunks: List[Dict]) -> str:
    """Join retrieved chunk texts into a single context block for the prompt."""
    return "\n\n---\n\n".join(c["chunk_text"] for c in chunks)
