"""
rag/generator.py

Grounded answer generation. This is the ONLY module that calls the local
Ollama server, isolated so the LLM provider could be swapped without
touching the retrieval or UI code.

Only the retrieved context (never the full document) is sent to the model,
along with an explicit instruction not to use outside knowledge and to say
so plainly when the context doesn't answer the question.
"""

from typing import Tuple

import requests

from utils import config

_GENERIC_ERROR_MESSAGE = (
    "Unable to generate a response right now. Please check that Ollama is "
    "running and try again."
)

_PROMPT_TEMPLATE = """You are a document question-answering assistant.

Answer the user's question using ONLY the supplied document context below.

If the answer cannot be determined from the provided context, explicitly
say that the information is not available in the uploaded document.

Do not invent facts.
Do not use outside knowledge.
Do not fabricate citations, page numbers, or document names - source
information is displayed separately by the application.

Question:
{question}

Document context:
\"\"\"
{context}
\"\"\"

Answer:
"""


def build_prompt(question: str, context: str) -> str:
    return _PROMPT_TEMPLATE.format(question=question, context=context)


def generate_answer(question: str, context: str) -> Tuple[bool, str]:
    """
    Generate a grounded answer to `question` using only `context`.

    Returns (success, text_or_error_message). Never raises - all failures
    are converted into a friendly message so the UI never shows a raw
    stack trace.
    """
    if not config.is_api_key_configured():
        return False, (
            f"Could not connect to Ollama at {config.OLLAMA_BASE_URL}. Make "
            "sure Ollama is installed and running (`ollama serve`), and "
            "restart the application."
        )

    if not context.strip():
        return False, (
            "No relevant context was retrieved for this question, so no "
            "answer can be generated. Try rephrasing, or confirm the right "
            "document is selected."
        )

    prompt = build_prompt(question, context)

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            headers=config.ollama_auth_headers(),
            json={"model": config.OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
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
            "The AI did not return a usable response. This can happen if "
            "the request was unclear. Please try rephrasing your question."
        )

    return True, text.strip()
