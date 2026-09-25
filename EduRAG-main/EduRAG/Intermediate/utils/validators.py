"""
utils/validators.py

Input validation, run before any processing (extraction, embedding, or
Ollama calls). Returns friendly (is_valid, message) tuples rather than
raising, so the Streamlit UI can display clear feedback without a stack
trace.
"""

from pathlib import Path
from typing import Tuple

from utils import config


def validate_upload(filename: str, file_size: int) -> Tuple[bool, str]:
    """Validate a file's type and size before any extraction happens."""
    if file_size == 0:
        return False, "The uploaded file is empty. Please choose a different file."

    if file_size > config.MAX_FILE_SIZE_BYTES:
        max_mb = config.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        return False, f"The uploaded file exceeds the maximum allowed size of {max_mb:.0f} MB."

    suffix = Path(filename).suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        return False, f"Unsupported file type '{suffix}'. Allowed types: {allowed}."

    return True, ""


def validate_extracted_text(text: str) -> Tuple[bool, str]:
    """Validate that extraction actually produced usable text."""
    if not text or not text.strip():
        return False, (
            "No readable text could be extracted from this file. It may be "
            "empty, corrupted, or an image-only (scanned) PDF with no "
            "embedded text layer."
        )
    return True, ""


def validate_question(question: str) -> Tuple[bool, str]:
    """Validate a user's question before running retrieval and generation."""
    question = question.strip()

    if not question:
        return False, "Please enter a question before asking."

    if len(question) < config.MIN_QUESTION_LENGTH:
        return False, "Please enter a more complete question."

    if len(question) > config.MAX_QUESTION_LENGTH:
        return False, (
            f"Your question is too long (maximum {config.MAX_QUESTION_LENGTH} characters)."
        )

    return True, ""
