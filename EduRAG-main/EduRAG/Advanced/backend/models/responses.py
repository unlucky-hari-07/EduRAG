"""
backend/models/responses.py

Pydantic models for API responses. Using explicit response models (rather
than raw dicts) means FastAPI validates and documents the exact shape of
every response, and prevents accidental leakage of internal fields.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    """A single cited source backing part of a generated answer.

    All fields here come from application-tracked metadata (set during
    ingestion) - never from the LLM - so a citation can never be invented.
    """

    document_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_id: str
    text: str


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    answer: str
    sources: List[SourceInfo]
    retrieved_chunks: int = Field(..., description="Number of candidate chunks retrieved from FAISS.")
    reranked_chunks: int = Field(..., description="Number of chunks kept after reranking.")
    grounded: bool = Field(..., description="Whether the groundedness check passed.")
    unsupported_claims: List[str] = Field(default_factory=list)
    iterations: int = Field(..., description="Number of retrieve/refine iterations performed.")
    insufficient_context: bool = Field(
        False, description="True if the system could not find enough grounded context to answer."
    )


class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    chunk_count: int
    ingested_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    count: int


class UploadResponse(BaseModel):
    document_id: str
    document_name: str
    chunk_count: int
    ingested_at: str
    message: str = "Document uploaded and indexed successfully."


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    indexed_documents: int


class ErrorResponse(BaseModel):
    detail: str
