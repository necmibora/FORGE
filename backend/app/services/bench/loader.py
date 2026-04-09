"""JSONL dataset loader for bundled benchmark files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

# backend/data/benchmarks lives next to backend/app
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "benchmarks"


def dataset_path(name: str) -> Path:
    return DATA_DIR / f"{name}.jsonl"


def iter_jsonl(name: str, limit: int | None = None) -> Iterator[dict]:
    p = dataset_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found: {p}. "
            f"Run scripts/build_{name}.py to generate it."
        )
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                return


def count_jsonl(name: str) -> int:
    p = dataset_path(name)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
