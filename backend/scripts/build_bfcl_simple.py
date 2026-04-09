"""Fetch BFCL v4 Simple (Python) from GitHub gorilla repo and write a normalized JSONL.

Downloads both the question file and the ground-truth answer file, merges
them into a single JSONL with one JSON object per line:

    {
      "id":            str,
      "messages":      [{"role": "user", "content": "..."}],
      "tools":         [{"type": "function", "function": {...}}],
      "expected_call": {"name": "func_name", "arguments": {"arg": ["val1","val2"]}}
    }

The ``tools`` field uses OpenAI-compatible format. BFCL's original
``"type": "dict"`` in parameter schemas is converted to ``"type": "object"``.

Run from the backend dir:
    python scripts/build_bfcl_simple.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data/"
)
Q_URL = BASE + "BFCL_v4_simple_python.json"
A_URL = BASE + "possible_answer/BFCL_v4_simple_python.json"

OUT = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "bfcl_simple.jsonl"

HEADERS = {"User-Agent": "forge-build/0.1"}


def _download_jsonl(url: str) -> list[dict]:
    """Download a newline-delimited JSON file and return parsed objects."""
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode("utf-8")
    objects = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            objects.append(json.loads(line))
    print(f"  -> {len(objects)} entries")
    return objects


def _fix_param_types(schema: dict) -> dict:
    """Recursively replace ``"type": "dict"`` with ``"type": "object"``."""
    if isinstance(schema, dict):
        return {
            k: ("object" if k == "type" and v == "dict" else _fix_param_types(v))
            for k, v in schema.items()
        }
    if isinstance(schema, list):
        return [_fix_param_types(item) for item in schema]
    return schema


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    questions = _download_jsonl(Q_URL)
    answers = _download_jsonl(A_URL)

    # Index answers by id
    answer_map: dict[str, dict] = {}
    for a in answers:
        answer_map[a["id"]] = a

    written = 0
    skipped = 0

    with OUT.open("w", encoding="utf-8") as f:
        for q in questions:
            qid = q["id"]
            ans = answer_map.get(qid)
            if ans is None:
                skipped += 1
                continue

            # question is [[{role, content}]] — unwrap outer list
            messages = q["question"][0]

            # Convert BFCL function defs to OpenAI tool format
            tools = []
            for func_def in q.get("function", []):
                params = _fix_param_types(func_def.get("parameters", {}))
                tools.append({
                    "type": "function",
                    "function": {
                        "name": func_def["name"],
                        "description": func_def.get("description", ""),
                        "parameters": params,
                    },
                })

            # Ground truth: [{func_name: {arg: [acceptable_values]}}]
            gt = ans["ground_truth"][0]
            func_name = list(gt.keys())[0]
            func_args = gt[func_name]

            obj = {
                "id": qid,
                "messages": messages,
                "tools": tools,
                "expected_call": {
                    "name": func_name,
                    "arguments": func_args,
                },
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} examples to {OUT} ({OUT.stat().st_size} bytes)")
    if skipped:
        print(f"Skipped {skipped} entries (no matching answer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
