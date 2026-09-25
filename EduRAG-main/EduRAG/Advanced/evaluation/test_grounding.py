"""
evaluation/test_grounding.py

Grounding evaluation: for each question in dataset.json, checks whether the
returned `grounded` flag and `insufficient_context` flag match expectations
- i.e. that answerable questions come back grounded, and the deliberately
unanswerable question correctly triggers an insufficient-context response
rather than a confident but ungrounded guess.

Same honesty note as evaluation/test_retrieval.py: this is a small
illustrative check, not a formal benchmark, and requires the backend
running with your own uploaded documents.

Run with: python evaluation/test_grounding.py
"""

import json
import os
import sys
from pathlib import Path

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


def load_dataset() -> list:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def run_grounding_evaluation() -> None:
    cases = load_dataset()
    results = []

    for case in cases:
        try:
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={"question": case["question"]},
                timeout=60,
            )
        except requests.exceptions.RequestException as exc:
            print(f"[ERROR] Could not reach backend at {BACKEND_URL}: {exc}")
            sys.exit(1)

        if response.status_code != 200:
            results.append({"question": case["question"], "status": "api_error"})
            continue

        body = response.json()
        expects_no_answer = case.get("expected_document_name") is None

        if expects_no_answer:
            passed = body.get("insufficient_context", False)
        else:
            passed = body.get("grounded", False) and not body.get("insufficient_context", False)

        results.append(
            {
                "question": case["question"],
                "passed": passed,
                "grounded": body.get("grounded"),
                "insufficient_context": body.get("insufficient_context"),
                "unsupported_claims": body.get("unsupported_claims", []),
            }
        )

    passed_count = sum(1 for r in results if r.get("passed"))
    total = len(results)

    print(f"\nGrounding evaluation: {passed_count}/{total} cases matched expectations.\n")
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"[{status}] {r['question']}")
        print(f"         grounded={r.get('grounded')} insufficient_context={r.get('insufficient_context')}")
        if r.get("unsupported_claims"):
            print(f"         unsupported_claims={r['unsupported_claims']}")

    print(
        "\nNote: this is a small, illustrative evaluation run, not a formal "
        "benchmark of hallucination rates. No numeric accuracy figure should "
        "be quoted beyond what this script reports on your own data."
    )


if __name__ == "__main__":
    run_grounding_evaluation()
