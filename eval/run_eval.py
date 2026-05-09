"""
Usage:
    python eval/run_eval.py

Requires the API to be running:
    uvicorn main:app --reload
"""

import json
import sys
from pathlib import Path

import requests

API_BASE = "http://localhost:8000"
DATASET = Path(__file__).parent / "dataset.json"


def stream_chat(query: str) -> dict:
    route = None
    tokens = []
    chunks = []

    with requests.post(
        f"{API_BASE}/chat/",
        json={"message": query},
        stream=True,
        timeout=60,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event["type"] == "route":
                route = event["content"]
            elif event["type"] == "token":
                tokens.append(event["content"])
            elif event["type"] == "chunks":
                chunks = event["content"]

    return {"route": route, "answer": "".join(tokens), "chunks": chunks}


def evaluate(case: dict, result: dict) -> list[str]:
    failures = []

    if result["route"] != case["expected_route"]:
        failures.append(
            f"Route mismatch: expected '{case['expected_route']}', got '{result['route']}'"
        )

    if case["expected_route"] == "rag" and not result["chunks"]:
        failures.append("No chunks returned for a RAG query")

    if not result["answer"].strip():
        failures.append("Empty answer")

    return failures


def main():
    cases = json.loads(DATASET.read_text())
    passed = 0
    failed = 0

    for case in cases:
        print(f"\n{'─' * 60}")
        print(f"[{case['id']:02d}] {case['query']}")
        print(f"     Expected route : {case['expected_route']}")

        try:
            result = stream_chat(case["query"])
        except Exception as exc:
            print(f"     ERROR: {exc}")
            failed += 1
            continue

        print(f"     Actual route   : {result['route']}")
        print(f"     Chunks returned: {len(result['chunks'])}")
        print(f"     Answer preview : {result['answer'][:120].strip()}…")

        failures = evaluate(case, result)
        if failures:
            failed += 1
            print(f"     FAIL")
            for f in failures:
                print(f"       • {f}")
        else:
            passed += 1
            print(f"     PASS")

    total = passed + failed
    print(f"\n{'═' * 60}")
    print(f"Results: {passed}/{total} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
