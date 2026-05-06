"""Prompt construction for the PEPS Coder Agent."""

from __future__ import annotations

import json
from typing import Any

from peps.core.io import to_plain_data
from peps.core.types import FESMRequirementSet, QueryInput
from peps.tools.workspace import ToolWorkspace

CODER_PROMPT_VERSION = "coder_prompt.v1"


CODER_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["reasoning", "code", "expected_outputs"],
    "properties": {
        "reasoning": {"type": "string"},
        "code": {"type": "string"},
        "expected_outputs": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


def build_coder_system_prompt() -> str:
    return """You are the PEPS Coder Agent.

Your job is deterministic geometric computation over already-acquired workspace variables.

Hard constraints:
1. Do not call perception tools.
2. Do not request new information.
3. Do not revise the FESM requirement set.
4. Do not infer from images or language beyond the provided variables.
5. Use only workspace variables and deterministic formulas.
6. Write a single Python function named execute(workspace).
7. execute(workspace) must return a JSON-serializable dict with keys:
   - answer: string final answer
   - computed_metrics: dict of numeric/intermediate values
   - decision_rule: string explaining the deterministic rule
8. The code runs in a restricted sandbox. Avoid file I/O, networking, subprocesses, eval, exec, and unsafe imports.
9. You may import only math, statistics, itertools, functools, collections, or operator.

Return only valid JSON matching the requested schema."""


def render_workspace_for_coder(workspace: ToolWorkspace) -> str:
    rows: dict[str, Any] = {}
    for name, value in workspace.values().items():
        rows[name] = {
            "source": value.source.value,
            "primitive_instance_id": value.primitive_instance_id,
            "confidence": value.confidence,
            "value": to_plain_data(value.value),
        }
    return json.dumps(rows, ensure_ascii=True, indent=2)


def build_coder_user_prompt(
    requirements: FESMRequirementSet,
    workspace: ToolWorkspace,
    *,
    query_input: QueryInput | None = None,
    previous_error: str | None = None,
) -> str:
    choices = query_input.choices if query_input is not None else []
    return f"""Generate deterministic Python code to answer the query from the workspace.

Original query:
{requirements.query}

Answer choices, if any:
{json.dumps(choices, ensure_ascii=True, indent=2)}

Fixed FESM requirement set:
{json.dumps(to_plain_data(requirements), ensure_ascii=True, indent=2)}

Workspace variables:
{render_workspace_for_coder(workspace)}

Previous sandbox error, if any:
{previous_error or "none"}

Code requirements:
1. Define exactly one callable entrypoint: execute(workspace).
2. Read values from workspace by key, e.g. workspace["target_points"].
3. Handle dataclass-like values as dictionaries because the sandbox receives JSON-compatible data.
4. Compute the metric primitives explicitly before mapping to answer.
5. If answer choices exist, return one of the choices exactly when possible.
6. Do not include markdown fences inside the code string.

Return JSON:
{{
  "reasoning": "...",
  "code": "def execute(workspace):\\n    ...",
  "expected_outputs": ["answer", "computed_metrics", "decision_rule"]
}}"""

