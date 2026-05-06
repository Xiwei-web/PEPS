"""Prompt construction for the PEPS Verifier Agent."""

from __future__ import annotations

import json
from typing import Any

from peps.core.io import to_plain_data
from peps.core.trace import ExecutionTrace
from peps.core.types import QueryInput

VERIFIER_PROMPT_VERSION = "verifier_prompt.v1"


def build_verifier_system_prompt() -> str:
    return """You are the PEPS Verifier Agent.

Your job is to verify whether a candidate answer is supported by a
primitive-grounded execution trace.

Hard constraints:
1. Do not solve the query from scratch.
2. Do not replace the candidate answer with your own answer.
3. Judge whether the provided answer follows from the FESM requirements, tool calls, workspace variables, and deterministic code result.
4. Use the images only to check grounding plausibility and obvious visual contradictions.
5. Check frame grounding, entity grounding, state coverage, metric coverage, code validity, and answer consistency.
6. If rejecting, distinguish:
   - missing_slot: the persistent schema can express the requirement, but the current primitive set or trace missed it.
   - primitive_gap: the persistent schema cannot express a required dependency.
7. Missing-slot feedback must tell the Parser which existing requirement area is incomplete.
8. Primitive-gap feedback must describe the missing expressive capacity and may propose temporary primitive hypotheses.
9. Quality score must be in [0, 1].

Return exactly these tags:
<verification_decision>accept|reject</verification_decision>
<quality_score>0.0-1.0</quality_score>
<reasoning>...</reasoning>
<feedback_type>none|missing_slot|primitive_gap</feedback_type>
<feedback>...</feedback>
<missing_slots>[...]</missing_slots>
<primitive_gap_hypotheses>[...]</primitive_gap_hypotheses>

For accept, use feedback_type none, feedback empty or 'none', missing_slots [], primitive_gap_hypotheses [].
For primitive_gap_hypotheses, output a JSON array of objects with fields:
family, name, description, arguments_schema, output_schema, metadata."""


def _truncate(value: str, *, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


def compact_trace_for_verifier(trace: ExecutionTrace) -> dict[str, Any]:
    """Render a compact trace focused on verifier-relevant evidence."""
    code_runs = []
    for run in trace.code_runs:
        code_runs.append(
            {
                "run_id": run.run_id,
                "status": run.status.value,
                "code": _truncate(run.code, limit=6000),
                "outputs": run.outputs,
                "computed_metrics": run.computed_metrics,
                "answer": run.answer,
                "error": run.error,
                "metadata": run.metadata,
            }
        )

    return {
        "trace_id": trace.trace_id,
        "schema_version": trace.schema_version,
        "requirements": to_plain_data(trace.requirements),
        "tool_calls": [
            {
                "tool_name": call.tool_name,
                "status": call.status.value,
                "fills": call.fills,
                "result_refs": call.result_refs,
                "result_preview": call.result_preview,
                "error": call.error,
                "metadata": call.metadata,
            }
            for call in trace.tool_calls
        ],
        "code_runs": code_runs,
        "workspace_snapshot": _truncate(
            json.dumps(to_plain_data(trace.workspace_snapshot), ensure_ascii=True, default=str),
            limit=6000,
        ),
        "final_answer": trace.final_answer,
        "computed_metrics": trace.computed_metrics,
        "errors": trace.errors,
    }


def build_verifier_user_prompt(
    query_input: QueryInput,
    trace: ExecutionTrace,
    *,
    candidate_answer: str | None = None,
) -> str:
    image_summary = [
        {
            "view_id": image.view_id or f"view_{index}",
            "uri": image.uri,
            "role": image.role,
            "metadata": image.metadata,
        }
        for index, image in enumerate(query_input.images)
    ]
    answer = candidate_answer if candidate_answer is not None else trace.final_answer
    return f"""Verify the PEPS candidate answer and trace.

Original query:
{query_input.query}

Answer choices, if any:
{json.dumps(query_input.choices, ensure_ascii=True, indent=2)}

Image/view metadata:
{json.dumps(image_summary, ensure_ascii=True, indent=2)}

Candidate answer:
{answer}

Primitive-grounded trace:
{json.dumps(compact_trace_for_verifier(trace), ensure_ascii=True, indent=2, default=str)}

Checklist:
1. Does the requirement set contain the necessary Frame primitives for the query?
2. Does it ground the necessary target, anchor, candidate, filter, view, and count entities?
3. Does it acquire the State variables needed by the Metrics?
4. Are the Metric primitives sufficient for the final answer?
5. Do tool calls fill the required values, or are there ungrounded assumptions?
6. Does the code compute from workspace variables rather than unsupported language reasoning?
7. Does the candidate answer follow from computed_metrics and decision_rule?
8. If rejected, is the issue missing_slot or primitive_gap?

Return the required tags only."""

