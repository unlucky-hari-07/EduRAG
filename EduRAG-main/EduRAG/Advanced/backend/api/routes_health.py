"""backend/api/routes_health.py — GET /health"""

from fastapi import APIRouter

from backend import config
from backend.models.responses import HealthResponse
from backend.services.ingestion import list_documents

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ollama_available=config.is_api_key_configured(),
        indexed_documents=len(list_documents()),
    )
