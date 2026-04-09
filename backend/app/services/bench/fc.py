"""Function-calling benchmark runner (BFCL Simple).

Each example provides a user message, a list of tool definitions, and an
expected function call (name + arguments with multiple acceptable values).

Scoring: exact match — the model must produce the correct function name AND
all argument values must match one of the acceptable alternatives.
"""
from __future__ import annotations

import json
import re
from typing import AsyncIterator

from app.services.bench.loader import count_jsonl, iter_jsonl
from app.services.vllm_runner import runner

NAME = "bfcl_simple"


def total_examples() -> int:
    return count_jsonl(NAME)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _extract_function_call(text: str) -> dict | None:
    """Try to extract a function call from model output.

    Models may output tool calls in various formats:
      - Raw JSON: {"name": "func", "arguments": {...}}
      - With tags: <tool_call>{"name": ...}</tool_call>
      - Python-style: func_name(arg1=val1, arg2=val2)
      - OpenAI-style: {"function": {"name": ..., "arguments": ...}}

    Returns {"name": str, "arguments": dict} or None.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Strip common wrapper tags
    for tag in ("tool_call", "function_call", "tool", "functioncall"):
        pattern = rf"<{tag}>(.*?)</{tag}>"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            break

    # Try direct JSON parse
    call = _try_parse_json_call(text)
    if call:
        return call

    # Try to find JSON object in the text
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        call = _try_parse_json_call(m.group())
        if call:
            return call

    # Try Python-style: func_name(arg1=val1, ...)
    call = _try_parse_python_call(text)
    if call:
        return call

    return None


def _try_parse_json_call(text: str) -> dict | None:
    """Parse a JSON string into a normalized function call."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(obj, dict):
        return None

    # Format: {"name": "func", "arguments": {...}}
    if "name" in obj and "arguments" in obj:
        args = obj["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                return None
        if isinstance(args, dict):
            return {"name": obj["name"], "arguments": args}

    # Format: {"function": {"name": ..., "arguments": ...}}
    if "function" in obj and isinstance(obj["function"], dict):
        return _try_parse_json_call(json.dumps(obj["function"]))

    # Format: {"func_name": {"arg1": val1}} (single-key dict = function name)
    if len(obj) == 1:
        name = list(obj.keys())[0]
        args = obj[name]
        if isinstance(args, dict):
            return {"name": name, "arguments": args}

    return None


def _try_parse_python_call(text: str) -> dict | None:
    """Parse Python-style function call: func_name(arg1=val1, arg2=val2)."""
    # Match: optional_module.func_name(...)
    m = re.match(r"([\w.]+)\((.*)\)\s*$", text, re.DOTALL)
    if not m:
        return None

    name = m.group(1)
    args_str = m.group(2).strip()
    if not args_str:
        return {"name": name, "arguments": {}}

    # Try to parse as Python kwargs -> JSON
    # Convert arg=value pairs
    try:
        # Wrap in dict constructor syntax and eval safely
        # First try JSON-like parsing
        args_json = "{" + re.sub(
            r"(\w+)\s*=\s*",
            r'"\1": ',
            args_str,
        ) + "}"
        args = json.loads(args_json)
        if isinstance(args, dict):
            return {"name": name, "arguments": args}
    except (json.JSONDecodeError, TypeError):
        pass

    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalize_value(v):
    """Normalize a value for comparison."""
    if isinstance(v, str):
        return v.strip()
    return v


def _check_argument(predicted_val, acceptable_vals: list) -> bool:
    """Check if a predicted value matches any of the acceptable values."""
    pred = _normalize_value(predicted_val)
    for acc in acceptable_vals:
        if _normalize_value(acc) == pred:
            return True
        # Numeric comparison: "10" == 10
        try:
            if float(pred) == float(acc):
                return True
        except (ValueError, TypeError):
            pass
    return False


def _score_call(predicted: dict | None, expected: dict) -> bool:
    """Check if predicted function call matches expected (exact match).

    Expected format:
        {"name": "func", "arguments": {"arg1": [val1, val2], "arg2": [val3]}}
    where each argument value is a list of acceptable alternatives.
    """
    if predicted is None:
        return False

    # Function name must match
    if predicted["name"] != expected["name"]:
        return False

    expected_args = expected["arguments"]
    predicted_args = predicted.get("arguments", {})

    # Check all required arguments (those in expected) are present and correct
    for arg_name, acceptable_vals in expected_args.items():
        if arg_name not in predicted_args:
            # If acceptable values include empty string, the arg is optional
            if "" in acceptable_vals or None in acceptable_vals:
                continue
            return False
        if not _check_argument(predicted_args[arg_name], acceptable_vals):
            return False

    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

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

    for ex in iter_jsonl(NAME, limit=limit):
        if should_cancel():
            return

        messages = ex["messages"]
        tools = ex["tools"]
        expected = ex["expected_call"]

        # Format prompt using the model's chat template with tools
        prompt = runner.format_prompt_with_tools(messages, tools)

        out = await runner.generate_one(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
        )

        predicted = _extract_function_call(out)
        if _score_call(predicted, expected):
            correct += 1

        done += 1
        await on_progress(done, total, correct)
        yield None
