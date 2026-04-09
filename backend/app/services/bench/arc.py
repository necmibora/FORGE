from __future__ import annotations

from typing import AsyncIterator

from app.services.bench.loader import count_jsonl
from app.services.bench.mcq import run_mcq_jsonl

NAME = "arc_easy"


def total_examples() -> int:
    return count_jsonl(NAME)


async def run(
    *,
    limit: int | None,
    temperature: float,
    max_tokens: int,
    on_progress,
    should_cancel,
) -> AsyncIterator[None]:
    async for _ in run_mcq_jsonl(
        dataset_name=NAME,
        limit=limit,
        temperature=temperature,
        max_tokens=max_tokens,
        on_progress=on_progress,
        should_cancel=should_cancel,
    ):
        yield None
