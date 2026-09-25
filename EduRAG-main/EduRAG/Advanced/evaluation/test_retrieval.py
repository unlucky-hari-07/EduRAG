"""
evaluation/test_retrieval.py

Retrieval evaluation: for each question in dataset.json, checks whether the
expected source document appears among the returned sources.

This is a small, honest evaluation harness - NOT a formal benchmark. No
numeric performance claims should be taken from this project beyond what
running this script actually reports on your own data.

IMPORTANT: This script calls the live backend API and therefore requires:
  1. The backend running (uvicorn backend.main:app), with a running Ollama
     server (and a pulled model) configured.
  2. The documents referenced in dataset.json already uploaded via
     POST /documents/upload.

If these prerequisites are not met, this script will report failures or
errors rather than fabricated results - that is expected and correct
behavior, not a bug.

Run with: python evaluation/test_retrieval.py
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


def run_retrieval_evaluation() -> None:
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
            results.append({"question": case["question"], "status": "api_error", "detail": response.text})
            continue

        body = response.json()
        retrieved_names = {s["document_name"] for s in body.get("sources", [])}

        expected = case.get("expected_document_name")
        if expected is None:
            # This case expects the system to find nothing relevant.
            hit = body.get("insufficient_context", False)
        else:
            hit = expected in retrieved_names

        results.append(
            {
                "question": case["question"],
                "expected_document_name": expected,
                "retrieved_documents": sorted(retrieved_names),
                "hit": hit,
            }
        )

    hits = sum(1 for r in results if r.get("hit"))
    total = len(results)

    print(f"\nRetrieval evaluation: {hits}/{total} cases matched expectations.\n")
    for r in results:
        status = "PASS" if r.get("hit") else "FAIL"
        print(f"[{status}] {r['question']}")
        if "retrieved_documents" in r:
            print(f"         expected: {r['expected_document_name']} | retrieved: {r['retrieved_documents']}")

    print(
        "\nNote: this is a small, illustrative evaluation run against the "
        "dataset in evaluation/dataset.json, not a formal benchmark. Results "
        "depend entirely on which documents you have uploaded."
    )


if __name__ == "__main__":
    run_retrieval_evaluation()
