"""
graph/nodes.py

Individual node functions for the EduRAG Pro LangGraph workflow. Each node
takes the current GraphState and returns a dict of the fields it updates -
the standard LangGraph node pattern. Keeping each node small and focused
(rather than one giant function) makes the workflow easy to test and reason
about node-by-node.
"""

from typing import Any, Dict

from backend import config
from backend.services import generation, grounding
from graph.state import GraphState
from retrieval.rerank import rerank_chunks
from retrieval.search import search_chunks


def validate_query(state: GraphState) -> Dict[str, Any]:
    """Ensure required state fields have sane starting values.

    Question length/emptiness is already validated by the Pydantic
    ChatRequest model at the API boundary; this node's job is only to
    initialize workflow control fields so the graph is self-contained even
    if invoked outside the API layer (e.g. in tests).
    """
    return {
        "iteration": 0,
        "max_iterations": state.get("max_iterations") or config.MAX_GRAPH_ITERATIONS,
    }


def rewrite_query(state: GraphState) -> Dict[str, Any]:
    """Rewrite the question into a more retrieval-friendly form."""
    success, result = generation.rewrite_query(state["question"])
    # Fail open: if rewriting fails for any reason, fall back to the
    # original question rather than blocking the whole pipeline.
    rewritten = result if success else state["question"]
    return {"rewritten_question": rewritten}


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieve candidate chunks from FAISS for the current question."""
    query = state.get("rewritten_question") or state["question"]
    candidates = search_chunks(
        query=query,
        candidate_k=config.RETRIEVAL_CANDIDATE_K,
        document_id=state.get("document_scope"),
    )
    return {"retrieved_chunks": candidates}


def rerank(state: GraphState) -> Dict[str, Any]:
    """Rerank retrieved candidates and build the final context string."""
    query = state.get("rewritten_question") or state["question"]
    candidates = state.get("retrieved_chunks", [])

    top_chunks = rerank_chunks(query, list(candidates), top_k=config.RERANK_TOP_K)

    context = "\n\n---\n\n".join(c["chunk_text"] for c in top_chunks)

    sources = [
        {
            "document_id": c["document_id"],
            "document_name": c["document_name"],
            "page_number": c.get("page_number"),
            "chunk_id": c["chunk_id"],
            "text": c["chunk_text"],
        }
        for c in top_chunks
    ]

    return {"reranked_chunks": top_chunks, "context": context, "sources": sources}


def check_context(state: GraphState) -> Dict[str, Any]:
    """Decide whether the reranked context is good enough to generate from."""
    chunks = state.get("reranked_chunks", [])

    if not chunks:
        return {"context_sufficient": False}

    best_score = max(c.get("rerank_score", 0.0) for c in chunks)
    sufficient = best_score >= config.MIN_CONTEXT_SCORE
    return {"context_sufficient": sufficient}


def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """Generate a grounded answer from the current context."""
    success, result = generation.generate_answer(state["question"], state.get("context", ""))

    if not success:
        # Surface the generation error as the answer itself so the API
        # response is still well-formed, and mark it ungrounded so it is
        # not presented as a confident, cited answer.
        return {"answer": result, "grounded": False, "unsupported_claims": []}

    return {"answer": result}


def check_groundedness(state: GraphState) -> Dict[str, Any]:
    """Verify the generated answer is supported by the retrieved context."""
    result = grounding.check_groundedness(
        question=state["question"],
        context=state.get("context", ""),
        answer=state.get("answer", ""),
    )
    return {
        "grounded": result["grounded"],
        "unsupported_claims": result["unsupported_claims"],
        "groundedness_reason": result["reason"],
    }


def refine_query(state: GraphState) -> Dict[str, Any]:
    """Refine the query for another retrieval attempt after insufficient/ungrounded results."""
    iteration = state.get("iteration", 0) + 1

    hint = (
        f"{state['question']}\n\n"
        "(Note: a previous search using this question did not return enough "
        "relevant context. Consider rephrasing with more specific terms or "
        "synonyms that might appear in the source document.)"
    )
    success, result = generation.rewrite_query(hint)
    rewritten = result if success else state.get("rewritten_question", state["question"])

    return {"iteration": iteration, "rewritten_question": rewritten}


def finalize(state: GraphState) -> Dict[str, Any]:
    """Produce the final state, marking insufficient-context cases clearly."""
    has_context = bool(state.get("reranked_chunks"))
    is_grounded = state.get("grounded", False)

    if not has_context or (not is_grounded and not state.get("answer")):
        return {
            "insufficient_context": True,
            "answer": state.get("answer") or (
                "The available documents do not provide enough information "
                "to answer this question confidently."
            ),
            "sources": [],
        }

    return {"insufficient_context": not is_grounded and not has_context}
