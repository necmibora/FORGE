"""ARC-Easy runner.

Generation-based MCQ scoring: build a 0-shot prompt with labeled choices,
ask the model for a single letter, parse the first matching label.
"""
from __future__ import annotations

import re
from typing import AsyncIterator

from app.services.bench.loader import count_jsonl, iter_jsonl
from app.services.vllm_runner import runner

NAME = "arc_easy"

PROMPT_TEMPLATE = (
    "Answer the multiple-choice question with a single letter only.\n\n"
    "Question: {question}\n"
    "{choices}\n"
    "Answer:"
)


def total_examples() -> int:
    return count_jsonl(NAME)


def _format_choices(labels: list[str], choices: list[str]) -> str:
    return "\n".join(f"{lbl}) {txt}" for lbl, txt in zip(labels, choices))


def _parse_letter(text: str, allowed: list[str]) -> str | None:
    """Find the first allowed label letter that appears in the model output."""
    if not text:
        return None
    # Strict: first non-space char if it's an allowed label.
    stripped = text.lstrip()
    if stripped and stripped[0].upper() in allowed:
        return stripped[0].upper()
    # Loose: any letter A-Z surrounded by non-letters.
    m = re.search(r"\b([A-Z])\b", text.upper())
    if m and m.group(1) in allowed:
        return m.group(1)
    return None


async def run(
    *,
    limit: int | None,
    temperature: float,
    max_tokens: int,
    on_progress,  # async callable: (done, total, correct) -> None
    should_cancel,  # callable: () -> bool
) -> AsyncIterator[None]:
    """Run ARC-Easy and call on_progress after each example.

    This is an async generator that yields None per example so the manager
    can interleave cancellation checks. Returns nothing; results are pushed
    via on_progress.
    """
    total_in_file = count_jsonl(NAME)
    total = min(total_in_file, limit) if limit is not None else total_in_file

    done = 0
    correct = 0
    for ex in iter_jsonl(NAME, limit=limit):
        if should_cancel():
            return
        labels: list[str] = ex["labels"]
        choices: list[str] = ex["choices"]
        gold: str = ex["answer"]

        prompt = PROMPT_TEMPLATE.format(
            question=ex["question"],
            choices=_format_choices(labels, choices),
        )

        out = await runner.generate_one(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=["\n"],
        )
        pred = _parse_letter(out, labels)
        if pred is not None and pred == gold.upper():
            correct += 1
        done += 1
        await on_progress(done, total, correct)
        yield None
