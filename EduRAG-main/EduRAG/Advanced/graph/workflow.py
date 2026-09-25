"""
graph/workflow.py

Assembles the EduRAG Pro LangGraph workflow:

    validate_query -> rewrite_query -> retrieve -> rerank -> check_context
        check_context:
            sufficient      -> generate_answer
            insufficient
              + retries left -> refine_query -> retrieve (loop)
              no retries     -> finalize
        generate_answer -> check_groundedness
        check_groundedness:
            grounded        -> finalize
            not grounded
              + retries left -> refine_query -> retrieve (loop)
              no retries     -> finalize
        finalize -> END

The iteration counter incremented in `refine_query` and compared against
`max_iterations` in both conditional routers guarantees this graph cannot
loop forever.
"""

from langgraph.graph import END, StateGraph

from graph import nodes
from graph.state import GraphState


def _route_after_context_check(state: GraphState) -> str:
    if state.get("context_sufficient"):
        return "generate_answer"
    if state.get("iteration", 0) < state.get("max_iterations", 2):
        return "refine_query"
    return "finalize"


def _route_after_groundedness(state: GraphState) -> str:
    if state.get("grounded"):
        return "finalize"
    if state.get("iteration", 0) < state.get("max_iterations", 2):
        return "refine_query"
    return "finalize"


def build_workflow():
    """Construct and compile the LangGraph workflow. Call once and reuse."""
    graph = StateGraph(GraphState)

    graph.add_node("validate_query", nodes.validate_query)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("check_context", nodes.check_context)
    graph.add_node("generate_answer", nodes.generate_answer_node)
    graph.add_node("check_groundedness", nodes.check_groundedness)
    graph.add_node("refine_query", nodes.refine_query)
    graph.add_node("finalize", nodes.finalize)

    graph.set_entry_point("validate_query")

    graph.add_edge("validate_query", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check_context")

    graph.add_conditional_edges(
        "check_context",
        _route_after_context_check,
        {
            "generate_answer": "generate_answer",
            "refine_query": "refine_query",
            "finalize": "finalize",
        },
    )

    graph.add_edge("generate_answer", "check_groundedness")

    graph.add_conditional_edges(
        "check_groundedness",
        _route_after_groundedness,
        {
            "finalize": "finalize",
            "refine_query": "refine_query",
        },
    )

    graph.add_edge("refine_query", "retrieve")
    graph.add_edge("finalize", END)

    return graph.compile()


_compiled_workflow = None


def get_workflow():
    """Return a process-wide singleton compiled workflow."""
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = build_workflow()
    return _compiled_workflow
