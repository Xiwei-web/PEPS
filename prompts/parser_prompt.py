"""Prompt construction for the PEPS Parser Agent."""

from __future__ import annotations

import json
from typing import Any, Iterable

from peps.core.enums import PrimitiveFamily
from peps.core.io import to_plain_data
from peps.core.types import PrimitiveDefinition, PrimitiveHypothesis, QueryInput
from peps.schema.candidate_pool import CandidatePrimitiveRecord
from peps.schema.example_library import ExampleRecord
from peps.schema.primitive_library import PrimitiveLibrary

PARSER_PROMPT_VERSION = "parser_prompt.v1"


PARSER_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "reasoning",
        "requirements",
        "minimality_rationale",
        "uncertainties",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "reasoning": {"type": "string"},
        "requirements": {
            "type": "object",
            "required": ["frame", "entity", "state", "metric"],
            "properties": {
                "frame": {"type": "array", "items": {"type": "object"}},
                "entity": {"type": "array", "items": {"type": "object"}},
                "state": {"type": "array", "items": {"type": "object"}},
                "metric": {"type": "array", "items": {"type": "object"}},
            },
            "additionalProperties": False,
        },
        "minimality_rationale": {"type": "string"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "primitive_gap_hypotheses": {"type": "array", "items": {"type": "object"}},
    },
    "additionalProperties": False,
}


def build_parser_system_prompt(*, allow_primitive_gap: bool = False) -> str:
    """Return the Parser Agent system prompt."""
    primitive_gap_rule = (
        "Primitive-gap mode is enabled only for this refinement round. If the "
        "current persistent schema cannot express a required information "
        "dependency, you may add primitive_gap_hypotheses. These hypotheses are "
        "not part of the main requirements unless explicitly listed as candidate "
        "hypotheses in the user context."
        if allow_primitive_gap
        else "Primitive-gap mode is disabled. You must not invent new primitives."
    )
    return f"""You are the PEPS Parser Agent.

Your job is committed query compilation: convert a spatial reasoning query into
a minimal FESM primitive requirement set.

Hard constraints:
1. Retrieve primitives only from the provided persistent primitive library.
2. Follow dependency order: Frame -> Entity -> State -> Metric.
3. Output what must be known, not how to acquire it.
4. Do not call tools.
5. Do not write code.
6. Do not answer the query.
7. Do not make final spatial decisions.
8. Keep the requirement set minimal but sufficient.
9. Each selected primitive item must include:
   - id: the exact primitive id from the library
   - arguments: concrete query-specific arguments
   - rationale: why this primitive is required
   - dependencies: use [] unless a dependency is unambiguous
   - confidence: a number in [0, 1]

{primitive_gap_rule}

Return only valid JSON matching the requested schema."""


def compact_definition(definition: PrimitiveDefinition) -> dict[str, Any]:
    """Compact a primitive definition for prompt context."""
    return {
        "id": definition.id,
        "family": definition.family.value,
        "name": definition.name,
        "description": definition.description,
        "arguments_schema": definition.arguments_schema,
        "dependencies": definition.dependencies,
        "computable_by": definition.computable_by,
        "tags": definition.tags,
    }


def render_primitive_library(
    library: PrimitiveLibrary,
    *,
    max_per_family: int | None = None,
) -> str:
    """Render primitive definitions grouped by FESM family."""
    rendered: dict[str, list[dict[str, Any]]] = {}
    for family in PrimitiveFamily:
        definitions = library.definitions_by_family(family)
        if max_per_family is not None:
            definitions = definitions[:max_per_family]
        rendered[family.value] = [compact_definition(item) for item in definitions]
    return json.dumps(rendered, ensure_ascii=True, indent=2)


def _unwrap_ranked_item(item: Any) -> Any:
    return getattr(item, "item", item)


def render_examples(examples: Iterable[ExampleRecord | Any] | None) -> str:
    rows: list[dict[str, Any]] = []
    for raw in examples or []:
        example = _unwrap_ranked_item(raw)
        if isinstance(example, ExampleRecord):
            rows.append(
                {
                    "query": example.query,
                    "answer": example.answer,
                    "quality_score": example.quality_score,
                    "primitive_ids": example.primitive_ids,
                    "summary": example.summary,
                    "computed_metrics": example.computed_metrics,
                }
            )
        else:
            rows.append(to_plain_data(example))
    return json.dumps(rows, ensure_ascii=True, indent=2)


def render_candidate_hypotheses(
    candidates: Iterable[CandidatePrimitiveRecord | PrimitiveHypothesis | Any] | None,
) -> str:
    rows: list[dict[str, Any]] = []
    for raw in candidates or []:
        candidate = _unwrap_ranked_item(raw)
        if isinstance(candidate, CandidatePrimitiveRecord):
            rows.append(
                {
                    "hypothesis": to_plain_data(candidate.hypothesis),
                    "reuse_count": candidate.reuse_count,
                    "average_score": candidate.average_score,
                    "average_improvement": candidate.average_improvement,
                    "status": candidate.status,
                }
            )
        elif isinstance(candidate, PrimitiveHypothesis):
            rows.append(to_plain_data(candidate))
        else:
            rows.append(to_plain_data(candidate))
    return json.dumps(rows, ensure_ascii=True, indent=2)


def build_parser_user_prompt(
    query_input: QueryInput,
    library: PrimitiveLibrary,
    *,
    examples: Iterable[ExampleRecord | Any] | None = None,
    verifier_feedback: str | None = None,
    candidate_hypotheses: Iterable[CandidatePrimitiveRecord | PrimitiveHypothesis | Any] | None = None,
    allow_primitive_gap: bool = False,
    max_primitives_per_family: int | None = None,
) -> str:
    """Build the user prompt for one Parser invocation."""
    image_summary = [
        {
            "view_id": image.view_id,
            "uri": image.uri,
            "role": image.role,
            "metadata": image.metadata,
        }
        for image in query_input.images
    ]
    gap_instruction = (
        "If verifier feedback indicates primitive_gap and no library primitive can "
        "express the missing dependency, include primitive_gap_hypotheses."
        if allow_primitive_gap
        else "Do not output primitive_gap_hypotheses unless the array is empty."
    )
    return f"""Compile the following spatial reasoning query into PEPS FESM requirements.

Query:
{query_input.query}

Answer choices, if any:
{json.dumps(query_input.choices, ensure_ascii=True, indent=2)}

Image/view metadata:
{json.dumps(image_summary, ensure_ascii=True, indent=2)}

Verifier feedback from the previous round, if any:
{verifier_feedback or "none"}

Relevant accepted examples:
{render_examples(examples)}

Candidate primitive hypotheses available for reuse:
{render_candidate_hypotheses(candidate_hypotheses)}

Persistent primitive library, grouped by family:
{render_primitive_library(library, max_per_family=max_primitives_per_family)}

Instructions:
1. Decide the reference frame requirements first.
2. Decide the target, anchor, candidate, view, filter, and count requirements next.
3. Decide the entity state variables required for later computation.
4. Decide the final metric primitives required to answer the query.
5. Select only primitives whose ids appear in the persistent library. Candidate hypotheses are context for primitive-gap analysis and must not be inserted into requirements in this version.
6. Use concrete string references in arguments, such as "target.clock", "anchor.door", "frame.viewer_entering_direction".
7. Set dependencies to [] unless you can name a concrete primitive instance id from this same output.
8. {gap_instruction}

Return JSON with this shape:
{{
  "schema_version": "{library.schema_version}",
  "reasoning": "...",
  "requirements": {{
    "frame": [
      {{"id": "frame.camera_based", "arguments": {{}}, "rationale": "...", "dependencies": [], "confidence": 0.0}}
    ],
    "entity": [],
    "state": [],
    "metric": []
  }},
  "minimality_rationale": "...",
  "uncertainties": [],
  "primitive_gap_hypotheses": []
}}"""
