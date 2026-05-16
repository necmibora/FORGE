"""Tool-call parsing and mismatch categorization for function-calling evals."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

ToolCallMismatchCategory = Literal[
    "correct",
    "no_tool_call",
    "malformed_tool_call",
    "wrong_tool_name",
    "missing_argument",
    "wrong_argument_value",
]

MISMATCH_CATEGORIES: tuple[ToolCallMismatchCategory, ...] = (
    "correct",
    "no_tool_call",
    "malformed_tool_call",
    "wrong_tool_name",
    "missing_argument",
    "wrong_argument_value",
)


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolCallEvaluation:
    correct: bool
    category: ToolCallMismatchCategory
    predicted: ParsedToolCall | None = None
    detail: str | None = None


@dataclass(frozen=True)
class _ParseResult:
    call: ParsedToolCall | None
    malformed: bool = False


def empty_mismatch_counts() -> dict[ToolCallMismatchCategory, int]:
    return {category: 0 for category in MISMATCH_CATEGORIES}


def evaluate_tool_call_output(text: str, expected: dict) -> ToolCallEvaluation:
    """Parse model output and categorize the first exact-match failure reason."""
    parsed = _extract_function_call(text)
    if parsed.call is None:
        category: ToolCallMismatchCategory = (
            "malformed_tool_call" if parsed.malformed else "no_tool_call"
        )
        return ToolCallEvaluation(correct=False, category=category)

    predicted = parsed.call
    if predicted.name != expected["name"]:
        return ToolCallEvaluation(
            correct=False,
            category="wrong_tool_name",
            predicted=predicted,
            detail=f"expected {expected['name']}, got {predicted.name}",
        )

    expected_args = expected["arguments"]
    for arg_name, acceptable_vals in expected_args.items():
        if arg_name not in predicted.arguments:
            if "" in acceptable_vals or None in acceptable_vals:
                continue
            return ToolCallEvaluation(
                correct=False,
                category="missing_argument",
                predicted=predicted,
                detail=arg_name,
            )
        if not _check_argument(predicted.arguments[arg_name], acceptable_vals):
            return ToolCallEvaluation(
                correct=False,
                category="wrong_argument_value",
                predicted=predicted,
                detail=arg_name,
            )

    return ToolCallEvaluation(correct=True, category="correct", predicted=predicted)


def _extract_function_call(text: str) -> _ParseResult:
    """Try to extract a normalized function call from model output."""
    if not text or not text.strip():
        return _ParseResult(call=None)

    text = text.strip()

    for tag in ("tool_call", "function_call", "tool", "functioncall"):
        pattern = rf"<{tag}>(.*?)</{tag}>"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            call = _try_parse_json_call(m.group(1).strip())
            return _ParseResult(call=call, malformed=call is None)

    call = _try_parse_json_call(text)
    if call:
        return _ParseResult(call=call)
    malformed = _looks_like_tool_json(text)

    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        candidate = m.group()
        call = _try_parse_json_call(candidate)
        if call:
            return _ParseResult(call=call)
        malformed = malformed or _looks_like_tool_json(candidate)

    pythonish = _looks_like_python_call(text)
    call = _try_parse_python_call(text)
    if call:
        return _ParseResult(call=call)

    return _ParseResult(call=None, malformed=malformed or pythonish)


def _try_parse_json_call(text: str) -> ParsedToolCall | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(obj, dict):
        return None

    if "name" in obj and "arguments" in obj:
        args = obj["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                return None
        if isinstance(obj["name"], str) and isinstance(args, dict):
            return ParsedToolCall(name=obj["name"], arguments=args)
        return None

    if "function" in obj and isinstance(obj["function"], dict):
        return _try_parse_json_call(json.dumps(obj["function"]))

    if len(obj) == 1:
        name = list(obj.keys())[0]
        args = obj[name]
        if isinstance(name, str) and isinstance(args, dict):
            return ParsedToolCall(name=name, arguments=args)

    return None


def _try_parse_python_call(text: str) -> ParsedToolCall | None:
    m = re.match(r"([\w.]+)\((.*)\)\s*$", text, re.DOTALL)
    if not m:
        return None

    name = m.group(1)
    args_str = m.group(2).strip()
    if not args_str:
        return ParsedToolCall(name=name, arguments={})

    try:
        args_json = "{" + re.sub(r"(\w+)\s*=\s*", r'"\1": ', args_str) + "}"
        args = json.loads(args_json)
        if isinstance(args, dict):
            return ParsedToolCall(name=name, arguments=args)
    except (json.JSONDecodeError, TypeError):
        pass

    return None


def _looks_like_tool_json(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ('"name"', '"arguments"', '"function"'))


def _looks_like_python_call(text: str) -> bool:
    return re.match(r"[\w.]+\(.+\)\s*$", text, re.DOTALL) is not None


def _normalize_value(v):
    if isinstance(v, str):
        return v.strip()
    return v


def _check_argument(predicted_val, acceptable_vals: list) -> bool:
    pred = _normalize_value(predicted_val)
    for acc in acceptable_vals:
        if _normalize_value(acc) == pred:
            return True
        try:
            if float(pred) == float(acc):
                return True
        except (ValueError, TypeError):
            pass
    return False
