"""
backend/services/generation.py

All direct calls to the local Ollama server for text generation live here:
query rewriting and grounded answer generation. Isolating these calls means
the LLM provider could be swapped later without touching the LangGraph
nodes that call these functions.

Streaming note: this project uses the standard (non-streaming) generate
call (`stream: false`). Ollama's HTTP API does support a streaming variant,
but given the short, structured answers this system produces and the extra
complexity of threading partial tokens through a FastAPI + LangGraph
pipeline, this project does not implement it and instead relies on
Streamlit's built-in spinner for a responsive loading experience. This is a
deliberate scope decision, not a technical limitation of Ollama or
LangGraph.
"""

from pathlib import Path
from typing import Tuple

import requests

from backend import config

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_GENERIC_ERROR_MESSAGE = (
    "Unable to generate a response right now. Please check that Ollama is "
    "running and try again."
)


def _load_prompt_template(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


_QUERY_REWRITE_TEMPLATE = None
_ANSWER_TEMPLATE = None


def _get_query_rewrite_template() -> str:
    global _QUERY_REWRITE_TEMPLATE
    if _QUERY_REWRITE_TEMPLATE is None:
        _QUERY_REWRITE_TEMPLATE = _load_prompt_template("query_rewrite.txt")
    return _QUERY_REWRITE_TEMPLATE


def _get_answer_template() -> str:
    global _ANSWER_TEMPLATE
    if _ANSWER_TEMPLATE is None:
        _ANSWER_TEMPLATE = _load_prompt_template("answer.txt")
    return _ANSWER_TEMPLATE


def _call_ollama(prompt: str) -> Tuple[bool, str]:
    """
    Shared low-level Ollama call with consistent error handling. Returns
    (success, text_or_error_message). Never raises - all failures are
    converted into a friendly message.
    """
    if not config.is_api_key_configured():
        return False, (
            f"Could not connect to Ollama at {config.OLLAMA_BASE_URL}. Make "
            "sure Ollama is installed and running (`ollama serve`), and "
            "restart the application."
        )

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            headers=config.ollama_auth_headers(),
            json={
                "model": config.OLLAMA_MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to Ollama at {config.OLLAMA_BASE_URL}."
    except requests.exceptions.Timeout:
        return False, "The request to Ollama timed out. Please try again."
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code in (401, 403):
            return False, (
                "Ollama rejected the request as unauthorized. If you're "
                "using Ollama Cloud, check OLLAMA_API_KEY in your .env file."
            )
        if status_code == 404:
            return False, (
                f"Model '{config.OLLAMA_MODEL_NAME}' was not found. If "
                f"running Ollama locally, pull it first with: "
                f"ollama pull {config.OLLAMA_MODEL_NAME}. If using Ollama "
                "Cloud, check the model name against "
                "ollama.com/search?c=cloud."
            )
        return False, _GENERIC_ERROR_MESSAGE
    except Exception:  # noqa: BLE001 - intentional broad catch at the boundary
        return False, _GENERIC_ERROR_MESSAGE

    text = data.get("response")

    if not text or not text.strip():
        return False, (
            "The AI did not return a usable response. This can happen if the "
            "request was unclear. Please try rephrasing your question."
        )

    return True, text.strip()


def rewrite_query(question: str) -> Tuple[bool, str]:
    """Rewrite a user question into a more retrieval-friendly form."""
    prompt = _get_query_rewrite_template().format(question=question)
    return _call_ollama(prompt)


def generate_answer(question: str, context: str) -> Tuple[bool, str]:
    """Generate a grounded answer to `question` using only `context`."""
    prompt = _get_answer_template().format(question=question, context=context)
    return _call_ollama(prompt)
