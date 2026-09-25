"""
rag/loaders.py

Text extraction for PDF (via PyMuPDF) and TXT files. Returns a list of
(page_number, text) tuples so page metadata can be carried through chunking
for citation purposes. TXT files have no real pagination, so they are
returned as a single "page" (page_number=None).
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
        # Covers PyMuPDF's various "cannot open broken document" style errors.
        raise DocumentLoadError(f"Failed to read PDF file - it may be corrupted: {exc}") from exc

    if not pages:
        raise DocumentLoadError(
            "No extractable text was found in the PDF. It may be a scanned, "
            "image-only document with no embedded text layer."
        )

    return pages


def load_txt(file_path: Path) -> List[PageText]:
    """Load a plain TXT file as a single page of text."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise DocumentLoadError(f"Failed to read text file: {exc}") from exc

    if not text or not text.strip():
        raise DocumentLoadError("The file is empty or contains no readable text.")

    return [(None, text)]


def load_document(file_path: Path) -> List[PageText]:
    """Dispatch to the correct loader based on file extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)
    if suffix == ".txt":
        return load_txt(file_path)

    raise DocumentLoadError(f"Unsupported file type: {suffix}")
