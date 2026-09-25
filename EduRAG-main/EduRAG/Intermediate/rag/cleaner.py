"""
rag/cleaner.py

Conservative text cleaning applied after extraction and before chunking.
Only normalizes whitespace - never alters actual content - so grounding
against the original document remains accurate.
"""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace and strip noisy artifacts from extracted text."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse excess blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)            # collapse repeated spaces/tabs

    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()
