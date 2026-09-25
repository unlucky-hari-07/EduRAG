"""
tests/test_api.py

API-level tests using FastAPI's TestClient. These exercise the actual HTTP
routes (health, chat, document validation) without hitting a real Ollama
server for most cases - only the "happy path" chat test requires a running
Ollama server with a pulled model, and is skipped automatically if one
isn't available.

Run with: pytest tests/test_api.py

Note: requires the full requirements.txt to be installed (FastAPI, Pydantic,
LangGraph, FAISS, Sentence Transformers) since it imports the real
application. This suite was written and manually reviewed for correctness,
but has not been executed in this project's development sandbox, which has
no internet access to install those dependencies. Run it locally after
`pip install -r requirements.txt` to confirm.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import requests
from fastapi.testclient import TestClient

from backend import config
from backend.main import app

client = TestClient(app)


def _ollama_is_running() -> bool:
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "ollama_available" in body
    assert "indexed_documents" in body


def test_chat_with_empty_question_returns_422():
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_with_too_short_question_returns_422():
    response = client.post("/chat", json={"question": "hi"})
    assert response.status_code == 422


def test_chat_with_missing_question_field_returns_422():
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_upload_unsupported_file_type_returns_422():
    fake_file = io.BytesIO(b"fake content")
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.docx", fake_file, "application/octet-stream")},
    )
    assert response.status_code == 422


def test_upload_empty_file_returns_422():
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", empty_file, "text/plain")},
    )
    assert response.status_code == 422


def test_get_documents_returns_list_shape():
    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert "documents" in body
    assert "count" in body
    assert isinstance(body["documents"], list)


def test_delete_nonexistent_document_returns_404():
    response = client.delete("/documents/nonexistent-id-12345")
    assert response.status_code == 404


@pytest.mark.skipif(
    not _ollama_is_running(),
    reason="Requires a running Ollama server (with a pulled model) and network access.",
)
def test_upload_and_chat_end_to_end():
    """
    End-to-end happy path: upload a small text document, then ask a question
    that should be answerable from it. Requires a running Ollama server, so
    it is skipped in environments without one (e.g. CI without Ollama, or
    this project's offline development sandbox).
    """
    content = (
        b"EduRAG Pro is a document intelligence system. "
        b"It uses FAISS for retrieval and a cross-encoder for reranking. "
        b"Generated answers are checked for groundedness before being returned."
    )
    upload_response = client.post(
        "/documents/upload",
        files={"file": ("test_doc.txt", io.BytesIO(content), "text/plain")},
    )
    assert upload_response.status_code == 200
    document_id = upload_response.json()["document_id"]

    chat_response = client.post(
        "/chat",
        json={"question": "What does EduRAG Pro use for reranking?", "document_id": document_id},
    )
    assert chat_response.status_code == 200
    body = chat_response.json()
    assert "answer" in body
    assert "sources" in body

    client.delete(f"/documents/{document_id}")
