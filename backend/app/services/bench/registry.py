"""Static registry of available benchmarks."""
from __future__ import annotations

from app.schemas import BenchmarkInfo
from app.services.bench import arc, mmlu

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
    mmlu.NAME: BenchmarkInfo(
        id="mmlu",
        name="MMLU",
        description=(
            "Massive Multitask Language Understanding. Broad academic and "
            "professional multiple-choice evaluation."
        ),
        total_examples=mmlu.total_examples(),
        kind="mcq",
        subject_count=mmlu.subject_count() or None,
    ),
}
