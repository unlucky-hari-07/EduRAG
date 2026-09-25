"""
tests/test_validation.py

Tests for request validation (Pydantic) and document upload validation,
independent of the API layer or any external services.

Run with: pytest tests/test_validation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import ValidationError

from backend.models.requests import ChatRequest
from backend.services.ingestion import DocumentValidationError, validate_upload


class TestChatRequestValidation:
    def test_valid_question_passes(self):
        request = ChatRequest(question="What is described in chapter two?")
        assert request.question == "What is described in chapter two?"

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="")

    def test_whitespace_only_question_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="   ")

    def test_too_short_question_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="hi")

    def test_excessively_long_question_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="a" * 5000)

    def test_optional_document_id_defaults_to_none(self):
        request = ChatRequest(question="A normal question here.")
        assert request.document_id is None

    def test_blank_document_id_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="A normal question here.", document_id="   ")


class TestDocumentUploadValidation:
    def test_empty_file_rejected(self):
        with pytest.raises(DocumentValidationError):
            validate_upload("notes.txt", 0)

    def test_oversized_file_rejected(self):
        with pytest.raises(DocumentValidationError):
            validate_upload("notes.pdf", 999_999_999)

    def test_unsupported_extension_rejected(self):
        with pytest.raises(DocumentValidationError):
            validate_upload("notes.docx", 1000)

    def test_valid_pdf_passes(self):
        validate_upload("notes.pdf", 1000)  # should not raise

    def test_valid_txt_passes(self):
        validate_upload("notes.txt", 1000)  # should not raise

    def test_valid_markdown_passes(self):
        validate_upload("notes.md", 1000)  # should not raise
