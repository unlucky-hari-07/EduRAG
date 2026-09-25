"""
backend/main.py

FastAPI application entrypoint for EduRAG Pro.

Run directly for local development:
    uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import routes_chat, routes_documents, routes_health

app = FastAPI(
    title="EduRAG Pro API",
    description="Production-style agentic document intelligence system.",
    version="1.0.0",
)

# Permissive CORS for local development between the Streamlit frontend and
# this API. In a real production deployment this should be restricted to
# the frontend's actual origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError) -> JSONResponse:
    """Return a clean 422 with a readable message instead of a raw trace."""
    first_error = exc.errors()[0] if exc.errors() else {}
    message = first_error.get("msg", "Invalid request.")
    return JSONResponse(status_code=422, content={"detail": message})


app.include_router(routes_health.router)
app.include_router(routes_documents.router)
app.include_router(routes_chat.router)
