"""
ingestion/cleaner.py

Text cleaning utilities applied after extraction and before chunking.
Kept intentionally conservative: normalize whitespace without altering
the actual content, so we never introduce information that wasn't in the
source document.
"""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace and strip noisy artifacts from extracted text."""
    if not text:
        return ""

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ blank lines into a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces/tabs into a single space.
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove trailing whitespace on each line.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()
