"""Helpers for getting validated JSON out of models with uneven structured-output support."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from pydantic import BaseModel

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> str:
    """First balanced JSON object in `text`, ignoring code fences and surrounding prose."""
    fenced = _FENCE.search(text or "")
    if fenced:
        text = fenced.group(1)
    start = (text or "").find("{")
    if start < 0:
        raise ValueError("No JSON object in model output")
    depth, in_str, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            escape = not escape and ch == "\\"
            if ch == '"' and not escape:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unbalanced JSON object in model output")


def strict_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema adjusted for strict modes: closed objects, every property required."""
    out = copy.deepcopy(schema.model_json_schema())

    def close(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)

    close(out)
    return out


def schema_instructions(schema: type[BaseModel]) -> str:
    """Appended to the system prompt when the provider cannot enforce the schema itself."""
    return (
        "\n\nRespond with a single JSON object and nothing else. "
        f"It must validate against this JSON schema:\n{json.dumps(schema.model_json_schema())}"
    )
