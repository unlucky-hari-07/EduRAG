"""
config.py

Centralized configuration for EduRAG Pro. All tunable values live here so
they can be adjusted (or overridden via environment variables) without
hunting through the codebase.
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent  # advanced/

# --- LLM (Ollama - local or Ollama Cloud) ------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()

# --- Storage locations ----------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss_index"
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"
METADATA_PATH = FAISS_DIR / "metadata.json"
DOCUMENTS_REGISTRY_PATH = DATA_DIR / "documents.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)

# --- Ingestion ------------------------------------------------------------
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", str(15 * 1024 * 1024)))  # 15 MB
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))

# --- Embeddings / retrieval -------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # matches all-MiniLM-L6-v2

RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "15"))  # candidates from FAISS
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))  # chunks kept after reranking

# --- Context quality / groundedness loop ------------------------------------
MIN_CONTEXT_SCORE = float(os.getenv("MIN_CONTEXT_SCORE", "0.15"))  # min rerank score to trust context
MAX_GRAPH_ITERATIONS = int(os.getenv("MAX_GRAPH_ITERATIONS", "2"))  # refine/retry limit

# --- Question validation ---------------------------------------------------
MIN_QUESTION_LENGTH = 3
MAX_QUESTION_LENGTH = 1000

# --- Frontend ---------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


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
