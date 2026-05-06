"""Robust JSON parsing helpers for LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from peps.core.exceptions import LLMOutputError

JSON_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def strip_code_fence(text: str) -> str:
    match = JSON_FENCE_RE.search(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_balanced_json(text: str) -> str | None:
    start_positions = [
        index for index, char in enumerate(text) if char in "[{"
    ]
    for start in start_positions:
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        stack: list[str] = []
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == "\"":
                    in_string = False
                continue
            if char == "\"":
                in_string = True
            elif char in "[{":
                stack.append("}" if char == "{" else "]")
            elif char in "]}":
                if not stack or char != stack[-1]:
                    break
                stack.pop()
                if not stack and char == closer:
                    return text[start : index + 1]
    return None


def parse_json_text(text: str) -> Any:
    """Parse a JSON object/array from raw LLM output."""
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = _extract_balanced_json(cleaned)
        if candidate is None:
            raise LLMOutputError("No JSON object or array found in LLM output")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"Failed to parse JSON from LLM output: {exc}") from exc


def parse_json_object(text: str) -> dict[str, Any]:
    parsed = parse_json_text(text)
    if not isinstance(parsed, dict):
        raise LLMOutputError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def parse_json_array(text: str) -> list[Any]:
    parsed = parse_json_text(text)
    if not isinstance(parsed, list):
        raise LLMOutputError(f"Expected JSON array, got {type(parsed).__name__}")
    return parsed


def extract_tag(text: str, tag: str) -> str | None:
    """Extract a simple XML-like tag used by verifier prompts."""
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else None

