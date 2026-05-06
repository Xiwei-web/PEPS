"""Parser Agent for PEPS.

The Parser compiles a query into a closed FESM primitive requirement set. It
does not call spatial tools, write code, or answer the query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Protocol

from peps.core.enums import FeedbackType, ToolCallStatus, TraceStage
from peps.core.exceptions import LLMOutputError, ValidationError
from peps.core.trace import AgentRunRecord
from peps.core.types import FESMRequirementSet, PrimitiveHypothesis, QueryInput
from peps.llm.image_payload import ImagePayload
from peps.llm.openai_client import OpenAIClient, OpenAIResponse
from peps.prompts.parser_prompt import (
    PARSER_PROMPT_VERSION,
    PARSER_RESPONSE_JSON_SCHEMA,
    build_parser_system_prompt,
    build_parser_user_prompt,
)
from peps.schema.candidate_pool import CandidatePrimitiveRecord
from peps.schema.example_library import ExampleLibrary, ExampleRecord
from peps.schema.primitive_instance import requirement_set_from_parser_output
from peps.schema.primitive_library import PrimitiveLibrary
from peps.schema.validators import validate_requirement_set_against_library


class ParserLLMClient(Protocol):
    """Minimal client interface required by ParserAgent."""

    def create_response(self, **kwargs: Any) -> OpenAIResponse:
        ...


@dataclass(slots=True)
class ParserAgentConfig:
    """Runtime settings for ParserAgent."""

    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 4096
    example_top_k: int = 3
    use_json_schema: bool = True
    strict_json_schema: bool = False
    validate_output: bool = True
    allow_primitive_gap: bool = False
    max_primitives_per_family: int | None = None


@dataclass(slots=True)
class ParserAgentResult:
    """Structured result of one Parser invocation."""

    requirements: FESMRequirementSet
    parsed_output: dict[str, Any]
    raw_output: str
    agent_run: AgentRunRecord
    primitive_gap_hypotheses: list[PrimitiveHypothesis] = field(default_factory=list)


class ParserAgent:
    """Compile query and images into a FESM requirement set."""

    def __init__(
        self,
        *,
        primitive_library: PrimitiveLibrary,
        llm_client: ParserLLMClient | None = None,
        example_library: ExampleLibrary | None = None,
        config: ParserAgentConfig | None = None,
    ) -> None:
        self.primitive_library = primitive_library
        self.llm_client = llm_client or OpenAIClient.from_env()
        self.example_library = example_library
        self.config = config or ParserAgentConfig()

    @classmethod
    def from_library_file(
        cls,
        path: str,
        *,
        llm_client: ParserLLMClient | None = None,
        example_library: ExampleLibrary | None = None,
        config: ParserAgentConfig | None = None,
    ) -> "ParserAgent":
        return cls(
            primitive_library=PrimitiveLibrary.from_file(path),
            llm_client=llm_client,
            example_library=example_library,
            config=config,
        )

    def parse(
        self,
        query_input: QueryInput | str,
        *,
        verifier_feedback: str | None = None,
        feedback_type: FeedbackType | str | None = None,
        examples: list[ExampleRecord | Any] | None = None,
        candidate_hypotheses: list[CandidatePrimitiveRecord | PrimitiveHypothesis | Any] | None = None,
        model: str | None = None,
    ) -> ParserAgentResult:
        """Run ParserAgent once and return a validated requirement set."""
        query_input = self._coerce_query_input(query_input)
        allow_primitive_gap = self._primitive_gap_enabled(feedback_type)
        selected_examples = self._select_examples(query_input, examples)

        system_prompt = build_parser_system_prompt(allow_primitive_gap=allow_primitive_gap)
        user_prompt = build_parser_user_prompt(
            query_input,
            self.primitive_library,
            examples=selected_examples,
            verifier_feedback=verifier_feedback,
            candidate_hypotheses=candidate_hypotheses,
            allow_primitive_gap=allow_primitive_gap,
            max_primitives_per_family=self.config.max_primitives_per_family,
        )
        agent_run = AgentRunRecord(
            agent_name="ParserAgent",
            stage=TraceStage.PARSER,
            prompt_name=PARSER_PROMPT_VERSION,
            status=ToolCallStatus.RUNNING,
            input_summary={
                "query_id": query_input.query_id,
                "query": query_input.query,
                "num_images": len(query_input.images),
                "schema_version": self.primitive_library.schema_version,
                "allow_primitive_gap": allow_primitive_gap,
                "num_examples": len(selected_examples),
            },
        )

        try:
            response = self.llm_client.create_response(
                system_prompt=system_prompt,
                user_text=user_prompt,
                images=[ImagePayload.from_image_ref(image) for image in query_input.images],
                model=model or self.config.model,
                max_output_tokens=self.config.max_output_tokens,
                temperature=self.config.temperature,
                text_format=self._text_format(),
            )
            parsed_output = response.json_object()
            requirements = requirement_set_from_parser_output(
                parsed_output,
                self.primitive_library,
                query=query_input.query,
            )
            if self.config.validate_output:
                validate_requirement_set_against_library(
                    requirements,
                    self.primitive_library.as_mapping(),
                    raise_on_error=True,
                )
            primitive_gap_hypotheses = self._parse_primitive_gap_hypotheses(parsed_output)
        except Exception as exc:
            agent_run.mark_failed(str(exc))
            raise

        agent_run.raw_output = response.text
        agent_run.parsed_output = parsed_output
        agent_run.mark_succeeded()
        return ParserAgentResult(
            requirements=requirements,
            parsed_output=parsed_output,
            raw_output=response.text,
            agent_run=agent_run,
            primitive_gap_hypotheses=primitive_gap_hypotheses,
        )

    def parse_output(
        self,
        parser_output: dict[str, Any] | str,
        *,
        query: str,
    ) -> FESMRequirementSet:
        """Convert already-produced Parser JSON into a validated requirement set."""
        if isinstance(parser_output, str):
            try:
                parser_output = json.loads(parser_output)
            except json.JSONDecodeError as exc:
                raise LLMOutputError(f"Parser output is not valid JSON: {exc}") from exc
        requirements = requirement_set_from_parser_output(
            parser_output,
            self.primitive_library,
            query=query,
        )
        validate_requirement_set_against_library(
            requirements,
            self.primitive_library.as_mapping(),
            raise_on_error=True,
        )
        return requirements

    def _coerce_query_input(self, query_input: QueryInput | str) -> QueryInput:
        if isinstance(query_input, QueryInput):
            return query_input
        if isinstance(query_input, str):
            return QueryInput(query=query_input)
        raise TypeError(f"Unsupported query_input type: {type(query_input).__name__}")

    def _select_examples(
        self,
        query_input: QueryInput,
        examples: list[ExampleRecord | Any] | None,
    ) -> list[ExampleRecord | Any]:
        if examples is not None:
            return examples
        if self.example_library is None or self.config.example_top_k <= 0:
            return []
        return self.example_library.retrieve(query_input.query, top_k=self.config.example_top_k)

    def _primitive_gap_enabled(self, feedback_type: FeedbackType | str | None) -> bool:
        if not self.config.allow_primitive_gap:
            return False
        if feedback_type is None:
            return False
        try:
            return FeedbackType.coerce(feedback_type) is FeedbackType.PRIMITIVE_GAP
        except ValueError:
            return False

    def _text_format(self) -> dict[str, Any] | None:
        if not self.config.use_json_schema:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "name": "peps_parser_output",
            "schema": PARSER_RESPONSE_JSON_SCHEMA,
            "strict": self.config.strict_json_schema,
        }

    def _parse_primitive_gap_hypotheses(
        self,
        parsed_output: dict[str, Any],
    ) -> list[PrimitiveHypothesis]:
        rows = parsed_output.get("primitive_gap_hypotheses", [])
        if not rows:
            return []
        if not self.config.allow_primitive_gap:
            raise ValidationError(
                "Parser returned primitive_gap_hypotheses while primitive-gap mode is disabled"
            )
        hypotheses: list[PrimitiveHypothesis] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValidationError(f"primitive_gap_hypotheses[{index}] must be an object")
            hypotheses.append(
                PrimitiveHypothesis(
                    family=row["family"],
                    name=row["name"],
                    description=row["description"],
                    arguments_schema=row.get("arguments_schema", {}),
                    output_schema=row.get("output_schema", {}),
                    trigger_query=row.get("trigger_query"),
                    metadata=row.get("metadata", {}),
                )
            )
        return hypotheses

