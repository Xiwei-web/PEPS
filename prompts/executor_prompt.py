"""Prompt construction for the PEPS Executor Agent."""

from __future__ import annotations

import json
from typing import Any

from peps.core.io import to_plain_data
from peps.core.types import FESMRequirementSet, QueryInput
from peps.tools.tool_registry import ToolRegistry
from peps.tools.workspace import ToolWorkspace

EXECUTOR_PROMPT_VERSION = "executor_prompt.v1"


EXECUTOR_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["analysis", "missing_requirements", "tool_calls", "workspace_notes"],
    "properties": {
        "analysis": {"type": "string"},
        "missing_requirements": {"type": "array", "items": {"type": "object"}},
        "tool_calls": {"type": "array", "items": {"type": "object"}},
        "workspace_notes": {"type": "string"},
    },
    "additionalProperties": False,
}


def build_executor_system_prompt(*, allow_code_tool: bool = False) -> str:
    code_rule = (
        "The code tool may be used only to derive intermediate primitive values "
        "from already-acquired variables. Do not compute or verbalize the final answer."
        if allow_code_tool
        else "Do not use the code tool in this Executor stage. Final deterministic computation belongs to the Coder Agent."
    )
    return f"""You are the PEPS Executor Agent.

Your job is value acquisition for a fixed FESM requirement set.

Hard constraints:
1. Follow the provided primitive requirement set exactly.
2. Do not add, remove, rename, or reinterpret primitives.
3. Do not answer the query.
4. Do not write final computation code.
5. Do not make spatial decisions such as left/right/front/behind.
6. Choose tool calls only to acquire missing values needed by the primitive set.
7. Tool arguments may use workspace references like "$reconstruction" or "$clock_detection.boxes[0]".
8. Tool calls with the same step_id must be independent and executable in parallel.
9. Prefer a small number of high-value calls that fill multiple primitives.
10. {code_rule}

Return only valid JSON matching the requested schema."""


def render_requirements(requirements: FESMRequirementSet) -> str:
    return json.dumps(to_plain_data(requirements), ensure_ascii=True, indent=2)


def render_tool_specs(
    tool_registry: ToolRegistry,
    *,
    allow_code_tool: bool = False,
) -> str:
    specs = {
        name: to_plain_data(spec)
        for name, spec in tool_registry.specs().items()
        if allow_code_tool or name != "code"
    }
    return json.dumps(specs, ensure_ascii=True, indent=2)


def render_workspace(workspace: ToolWorkspace | None) -> str:
    if workspace is None:
        return "{}"
    summary: dict[str, Any] = {}
    for name, value in workspace.values().items():
        raw = value.value
        if hasattr(raw, "to_message_content"):
            preview = raw.to_message_content()
        else:
            preview = repr(raw)
        summary[name] = {
            "source": value.source.value,
            "primitive_instance_id": value.primitive_instance_id,
            "confidence": value.confidence,
            "preview": preview[:500],
        }
    return json.dumps(summary, ensure_ascii=True, indent=2)


def build_executor_user_prompt(
    requirements: FESMRequirementSet,
    tool_registry: ToolRegistry,
    *,
    query_input: QueryInput | None = None,
    workspace: ToolWorkspace | None = None,
    executor_feedback: str | None = None,
    allow_code_tool: bool = False,
) -> str:
    image_summary = []
    if query_input is not None:
        image_summary = [
            {
                "view_id": image.view_id or f"view_{index}",
                "uri": image.uri,
                "role": image.role,
                "metadata": image.metadata,
            }
            for index, image in enumerate(query_input.images)
        ]

    return f"""Plan immediate tool calls to acquire values for the fixed PEPS requirement set.

Original query:
{requirements.query}

Image/view metadata:
{json.dumps(image_summary, ensure_ascii=True, indent=2)}

Fixed FESM requirement set:
{render_requirements(requirements)}

Current workspace:
{render_workspace(workspace)}

Executor feedback or error from previous attempt, if any:
{executor_feedback or "none"}

Available tools:
{render_tool_specs(tool_registry, allow_code_tool=allow_code_tool)}

Output instructions:
1. missing_requirements should list primitive instances that still need values.
2. tool_calls should contain only calls to available tools.
3. Each tool call must use this shape:
   {{
     "step_id": 1,
     "tool": "detect",
     "output_name": "target_clock_detection",
     "arguments": {{"image_source": "/path/or/url.png", "prompt": "clock"}},
     "fills": ["primitive_instance_id or requirement description"],
     "rationale": "why this call fills those requirements"
   }}
4. Use image URI strings from Image/view metadata when a tool needs an image.
5. Use workspace references only for values already present in Current workspace or produced by lower step_id calls.
6. Do not include natural language answer fields.

Return JSON:
{{
  "analysis": "...",
  "missing_requirements": [],
  "tool_calls": [],
  "workspace_notes": "..."
}}"""

