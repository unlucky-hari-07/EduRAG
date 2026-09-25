"""
llm.py

Isolated LLM service layer for EduRAG StudyMate.

All Ollama interaction lives in this file. The rest of the application
(app.py) never talks to the Ollama HTTP API directly - it only calls
`generate_response()`. This means the model provider could be swapped out
later (e.g. for OpenAI or Gemini) by rewriting this file only, with no
changes needed to the UI, prompts, or validation logic.

Works with either:
  - A local Ollama server (OLLAMA_BASE_URL=http://localhost:11434, no key)
  - Ollama Cloud (OLLAMA_BASE_URL=https://ollama.com, OLLAMA_API_KEY set)
The request/response shape is identical either way - only the base URL and
an optional Authorization header differ.
"""

import os
from typing import Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()

_GENERIC_ERROR_MESSAGE = (
    "Unable to generate a response right now. Please check your Ollama "
    "configuration and try again."
)


def _auth_headers() -> dict:
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
            f"{OLLAMA_BASE_URL}/api/tags", headers=_auth_headers(), timeout=5
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def generate_response(prompt: str) -> Tuple[bool, str]:
    """
    Send a prompt to the configured Ollama endpoint (local or cloud) and
    return the result.

    Returns:
        (success, content)
        - success=True, content=<generated text>          on success
        - success=False, content=<user-friendly message>  on any failure

    This function deliberately catches every exception. The caller (the
    Streamlit UI) should never see a raw stack trace - only a clear,
    friendly message it can display with st.error().
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            headers=_auth_headers(),
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        return False, (
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}. If you're "
            "using a local install, make sure it's running (`ollama serve`). "
            "If you're using Ollama Cloud, check your internet connection."
        )
    except requests.exceptions.Timeout:
        return False, (
            "The request to Ollama timed out. The model may still be "
            "loading, or the connection may be slow. Please try again."
        )
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None

        if status_code in (401, 403):
            return False, (
                "Ollama rejected the request as unauthorized. If you're "
                "using Ollama Cloud, check that OLLAMA_API_KEY in your "
                ".env file is correct."
            )

        if status_code == 404:
            return False, (
                f"Model '{MODEL_NAME}' was not found. If you're running "
                f"Ollama locally, pull it first with: ollama pull "
                f"{MODEL_NAME}. If you're using Ollama Cloud, check the "
                "model name against ollama.com/search?c=cloud."
            )

        return False, _GENERIC_ERROR_MESSAGE
    except Exception:  # noqa: BLE001 - intentional broad catch at the boundary
        return False, _GENERIC_ERROR_MESSAGE

    text = data.get("response")

    if not text or not text.strip():
        return False, (
            "The AI did not return a response. This can happen if the "
            "request was unclear or the model produced no output. Please "
            "try rephrasing your input."
        )

    return True, text.strip()
