"""
graph/state.py

Typed state passed between LangGraph nodes. Using a TypedDict keeps every
node's inputs/outputs explicit and makes the workflow easy to reason about
and test node-by-node.
"""

from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    # Input
    question: str
    document_scope: Optional[str]

    # Query rewriting
    rewritten_question: str

    # Retrieval / reranking
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    context: str
    context_sufficient: bool

    # Generation
    answer: str

    # Groundedness
    grounded: bool
    unsupported_claims: List[str]
    groundedness_reason: str

    # Control flow
    iteration: int
    max_iterations: int
    insufficient_context: bool

    # Output
    sources: List[Dict[str, Any]]
