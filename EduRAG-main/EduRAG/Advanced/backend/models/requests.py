"""
backend/models/requests.py

Pydantic models for validating incoming API requests.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

MIN_QUESTION_LENGTH = 3
MAX_QUESTION_LENGTH = 1000


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    question: str = Field(..., description="The student's question.")
    document_id: Optional[str] = Field(
        None,
        description="Optional document ID to restrict retrieval to a single document.",
    )

    @field_validator("question")
    @classmethod
    def question_must_be_reasonable_length(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question must not be empty.")
        if len(stripped) < MIN_QUESTION_LENGTH:
            raise ValueError(f"Question must be at least {MIN_QUESTION_LENGTH} characters.")
        if len(stripped) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Question must be at most {MAX_QUESTION_LENGTH} characters.")
        return stripped

    @field_validator("document_id")
    @classmethod
    def document_id_must_not_be_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("document_id must not be blank if provided.")
        return value.strip() if value else value
