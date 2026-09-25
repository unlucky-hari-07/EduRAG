"""
utils/config.py

Centralized, tunable configuration for EduRAG Document Q&A. Values here can
be overridden via environment variables (see .env.example) without editing
code.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

# --- LLM (Ollama - local or Ollama Cloud) ------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()

# --- File validation -----------------------------------------------------
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", str(15 * 1024 * 1024)))  # 15 MB

# --- Chunking --------------------------------------------------------------
# Word-based chunking. These are reasonable starting points, not universally
# optimal values - see the README for guidance on tuning them for your own
# documents and retrieval quality.
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "220"))   # ~roughly 800-1000 tokens
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))

# --- Embeddings / retrieval -------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # matches all-MiniLM-L6-v2

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# --- Question validation -----------------------------------------------------
MIN_QUESTION_LENGTH = 3
MAX_QUESTION_LENGTH = 1000


def ollama_auth_headers() -> dict:
    """Ollama Cloud requires 'Authorization: Bearer <key>'; local Ollama ignores it."""
    if OLLAMA_API_KEY:
        return {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    return {}


def is_api_key_configured() -> bool:
    """
    Check whether the configured Ollama endpoint (local or cloud) is reachable.

    Named to match the app's original "is the LLM ready" check.
    """
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags", headers=ollama_auth_headers(), timeout=5
        )
        return response.status_code == 200
    except Exception:  # noqa: BLE001 - any connectivity issue means "not configured"
        return False
