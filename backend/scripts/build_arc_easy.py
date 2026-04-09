"""Fetch ARC-Easy from HuggingFace and write a normalized JSONL.

We download the parquet file directly via urllib so this script does not
depend on the `datasets` package. Output schema (one JSON object per line):

    {
      "id":       str,
      "question": str,
      "labels":   list[str],   # e.g. ["A","B","C","D"] (some items have 3 or 5)
      "choices":  list[str],   # text in same order as labels
      "answer":   str          # the gold label, e.g. "B"
    }

Run from the backend dir:
    python scripts/build_arc_easy.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

URL = (
    "https://huggingface.co/datasets/allenai/ai2_arc/"
    "resolve/main/ARC-Easy/test-00000-of-00001.parquet"
)
OUT = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "arc_easy.jsonl"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet")

    print(f"Downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "forge-build/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        tmp.write_bytes(r.read())
    print(f"  -> {tmp.name} ({tmp.stat().st_size} bytes)")

    table = pq.read_table(tmp)
    rows = table.to_pylist()
    print(f"Read {len(rows)} rows")

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            choices = row["choices"]
            obj = {
                "id": row["id"],
                "question": row["question"],
                "labels": list(choices["label"]),
                "choices": list(choices["text"]),
                "answer": row["answerKey"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")

    tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
