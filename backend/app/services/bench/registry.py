"""Static registry of available benchmarks."""
from __future__ import annotations

from app.schemas import BenchmarkInfo
from app.services.bench import arc

BENCHMARKS: dict[str, BenchmarkInfo] = {
    arc.NAME: BenchmarkInfo(
        id="arc_easy",
        name="ARC-Easy",
        description=(
            "AI2 Reasoning Challenge — Easy split. Grade-school science "
            "multiple-choice questions. Generation-based scoring (0-shot)."
        ),
        total_examples=arc.total_examples(),
        kind="mcq",
    ),
}
