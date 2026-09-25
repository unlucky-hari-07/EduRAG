"""
backend/services/grounding.py

Groundedness verification: asks the LLM to check whether a generated answer
is actually supported by the retrieved context, and returns a structured
verdict.

This is a mitigation mechanism, not a guarantee. It reduces the chance that
an ungrounded answer reaches the user by catching many cases where the model
strayed from the retrieved context, but it cannot mathematically guarantee
the absence of hallucination - the check itself is performed by the same
class of LLM that could hallucinate.
"""

import json
import re
from typing import List, TypedDict

from backend.services.generation import _call_ollama, _PROMPTS_DIR


class GroundednessResult(TypedDict):
    grounded: bool
    unsupported_claims: List[str]
    reason: str


_GROUNDEDNESS_TEMPLATE = None


def _get_groundedness_template() -> str:
    global _GROUNDEDNESS_TEMPLATE
    if _GROUNDEDNESS_TEMPLATE is None:
        _GROUNDEDNESS_TEMPLATE = (_PROMPTS_DIR / "groundedness.txt").read_text(encoding="utf-8")
    return _GROUNDEDNESS_TEMPLATE


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model response."""
    # Strip markdown code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to grabbing the first {...} block in the text.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse a JSON groundedness verdict from the model response.")


def check_groundedness(question: str, context: str, answer: str) -> GroundednessResult:
    """
    Verify whether `answer` is supported by `context`.

    On any failure (API error or unparseable response), fails safe by
    returning grounded=False with an explanatory reason, so the calling
    workflow treats the answer as unverified rather than trusting it blindly.
    """
    prompt = _get_groundedness_template().format(question=question, context=context, answer=answer)
    success, content = _call_ollama(prompt)

    if not success:
        return {
            "grounded": False,
            "unsupported_claims": [],
            "reason": f"Groundedness check could not be completed: {content}",
        }

    try:
        parsed = _extract_json(content)
        return {
            "grounded": bool(parsed.get("grounded", False)),
            "unsupported_claims": list(parsed.get("unsupported_claims", [])),
            "reason": str(parsed.get("reason", "")),
        }
    except (ValueError, TypeError):
        return {
            "grounded": False,
            "unsupported_claims": [],
            "reason": "Groundedness check returned an unparseable response.",
        }
