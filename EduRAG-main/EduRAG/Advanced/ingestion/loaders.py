"""
ingestion/loaders.py

Document loading and text extraction for PDF, TXT, and Markdown files.

Each loader returns a list of (page_number, text) tuples so downstream
chunking can retain page-level metadata where it exists. TXT and Markdown
files have no real pagination, so they are returned as a single
"page" (page_number=None).
"""

from pathlib import Path
from typing import List, Optional, Tuple

PageText = Tuple[Optional[int], str]


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or parsed."""


def load_pdf(file_path: Path) -> List[PageText]:
    """Extract text from a PDF file, one entry per page."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise DocumentLoadError(
            "PyMuPDF (fitz) is required to process PDF files but is not installed."
        ) from exc

    pages: List[PageText] = []
    try:
        with fitz.open(file_path) as doc:
            if doc.page_count == 0:
                raise DocumentLoadError("The PDF file contains no pages.")
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text and text.strip():
                    pages.append((page_number, text))
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"Failed to read PDF file: {exc}") from exc

    if not pages:
        raise DocumentLoadError("No extractable text was found in the PDF.")

    return pages


def load_text(file_path: Path) -> List[PageText]:
    """Load a plain TXT or Markdown file as a single page of text."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise DocumentLoadError(f"Failed to read text file: {exc}") from exc

    if not text or not text.strip():
        raise DocumentLoadError("The file is empty or contains no readable text.")

    return [(None, text)]


def load_document(file_path: Path) -> List[PageText]:
    """
    Dispatch to the correct loader based on file extension.

    Returns a list of (page_number, text) tuples.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)
    if suffix in (".txt", ".md"):
        return load_text(file_path)

    raise DocumentLoadError(f"Unsupported file type: {suffix}")
