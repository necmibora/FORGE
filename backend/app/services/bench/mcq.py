from __future__ import annotations

import re
from typing import AsyncIterator

from app.services.bench.loader import count_jsonl, iter_jsonl
from app.services.vllm_runner import runner

PROMPT_TEMPLATE = (
    "Answer the multiple-choice question with a single letter only.\n\n"
    "Question: {question}\n"
    "{choices}\n"
    "Answer:"
)


def format_choices(labels: list[str], choices: list[str]) -> str:
    return "\n".join(f"{label}) {choice}" for label, choice in zip(labels, choices))


def parse_letter(text: str, allowed: list[str]) -> str | None:
    if not text:
        return None

    stripped = text.lstrip()
    if stripped and stripped[0].upper() in allowed:
        return stripped[0].upper()

    match = re.search(r"\b([A-Z])\b", text.upper())
    if match and match.group(1) in allowed:
        return match.group(1)
    return None


async def run_mcq_jsonl(
    *,
    dataset_name: str,
    limit: int | None,
    temperature: float,
    max_tokens: int,
    on_progress,
    should_cancel,
) -> AsyncIterator[None]:
    total_in_file = count_jsonl(dataset_name)
    total = min(total_in_file, limit) if limit is not None else total_in_file

    done = 0
    correct = 0

    for ex in iter_jsonl(dataset_name, limit=limit):
        if should_cancel():
            return

        labels: list[str] = ex["labels"]
        choices: list[str] = ex["choices"]
        gold: str = ex["answer"]

        prompt = PROMPT_TEMPLATE.format(
            question=ex["question"],
            choices=format_choices(labels, choices),
        )

        out = await runner.generate_one(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=["\n"],
        )
        pred = parse_letter(out, labels)
        if pred is not None and pred == gold.upper():
            correct += 1

        done += 1
        await on_progress(done, total, correct)
        yield None
