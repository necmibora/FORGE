"""Function-calling benchmark runner (BFCL Simple)."""
from __future__ import annotations

from typing import AsyncIterator

from app.services.bench.loader import count_jsonl, iter_jsonl
from app.services.bench.tool_calling import (
    empty_mismatch_counts,
    evaluate_tool_call_output,
)
from app.services.vllm_runner import runner

NAME = "bfcl_simple"


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
    total_in_file = count_jsonl(NAME)
    total = min(total_in_file, limit) if limit is not None else total_in_file

    done = 0
    correct = 0
    total_tokens = 0
    mismatch_counts = empty_mismatch_counts()

    for ex in iter_jsonl(NAME, limit=limit):
        if should_cancel():
            return

        messages = ex["messages"]
        tools = ex["tools"]
        expected = ex["expected_call"]

        prompt = runner.format_prompt_with_tools(messages, tools)

        out, tokens = await runner.generate_one(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
        )
        total_tokens += tokens

        evaluation = evaluate_tool_call_output(out, expected)
        mismatch_counts[evaluation.category] += 1
        if evaluation.correct:
            correct += 1

        done += 1
        await on_progress(done, total, correct, total_tokens, mismatch_counts)
        yield None
