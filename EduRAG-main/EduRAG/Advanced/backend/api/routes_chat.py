"""
backend/api/routes_chat.py

    POST /chat

Runs the user's question through the compiled LangGraph workflow
(query validation -> rewriting -> retrieval -> reranking -> context check ->
grounded generation -> groundedness check -> refinement loop -> finalize)
and returns a structured, cited response.
"""

from fastapi import APIRouter, HTTPException

from backend import config
from backend.models.requests import ChatRequest
from backend.models.responses import ChatResponse, SourceInfo
from graph.workflow import get_workflow

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    workflow = get_workflow()

    initial_state = {
        "question": request.question,
        "document_scope": request.document_id,
        "max_iterations": config.MAX_GRAPH_ITERATIONS,
    }

    try:
        final_state = workflow.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001 - never leak internal stack traces to the client
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question.",
        ) from exc

    sources = [SourceInfo(**s) for s in final_state.get("sources", [])]

    return ChatResponse(
        answer=final_state.get("answer", ""),
        sources=sources,
        retrieved_chunks=len(final_state.get("retrieved_chunks", [])),
        reranked_chunks=len(final_state.get("reranked_chunks", [])),
        grounded=final_state.get("grounded", False),
        unsupported_claims=final_state.get("unsupported_claims", []),
        iterations=final_state.get("iteration", 0),
        insufficient_context=final_state.get("insufficient_context", False),
    )
