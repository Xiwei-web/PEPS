"""Verifier Agent for PEPS."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

from peps.core.enums import FeedbackType, ToolCallStatus, TraceStage, VerificationDecision
from peps.core.exceptions import LLMOutputError, ValidationError
from peps.core.trace import AgentRunRecord, ExecutionTrace, VerificationResult
from peps.core.types import PrimitiveHypothesis, QueryInput
from peps.llm.image_payload import ImagePayload
from peps.llm.json_parser import extract_tag, parse_json_array
from peps.llm.openai_client import OpenAIClient, OpenAIResponse
from peps.prompts.verifier_prompt import (
    VERIFIER_PROMPT_VERSION,
    build_verifier_system_prompt,
    build_verifier_user_prompt,
)


class VerifierLLMClient(Protocol):
    """Minimal client interface required by VerifierAgent."""

    def create_response(self, **kwargs: Any) -> OpenAIResponse:
        ...


@dataclass(slots=True)
class VerifierAgentConfig:
    """Runtime settings for VerifierAgent."""

    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 2048
    use_images: bool = True
    normalize_legacy_scores: bool = True
    require_feedback_on_reject: bool = True


@dataclass(slots=True)
class VerifierAgentResult:
    """Structured result of one Verifier invocation."""

    verification: VerificationResult
    raw_output: str
    parsed_output: dict[str, Any]
    agent_run: AgentRunRecord


class VerifierAgent:
    """Verify that an answer is supported by a PEPS execution trace."""

    def __init__(
        self,
        *,
        llm_client: VerifierLLMClient | None = None,
        config: VerifierAgentConfig | None = None,
    ) -> None:
        self.llm_client = llm_client or OpenAIClient.from_env()
        self.config = config or VerifierAgentConfig()

    def verify(
        self,
        query_input: QueryInput | str,
        trace: ExecutionTrace,
        *,
        candidate_answer: str | None = None,
        model: str | None = None,
    ) -> VerifierAgentResult:
        query_input = self._coerce_query_input(query_input, trace)
        system_prompt = build_verifier_system_prompt()
        user_prompt = build_verifier_user_prompt(
            query_input,
            trace,
            candidate_answer=candidate_answer,
        )
        agent_run = AgentRunRecord(
            agent_name="VerifierAgent",
            stage=TraceStage.VERIFIER,
            prompt_name=VERIFIER_PROMPT_VERSION,
            status=ToolCallStatus.RUNNING,
            input_summary={
                "query_id": query_input.query_id,
                "trace_id": trace.trace_id,
                "schema_version": trace.schema_version,
                "candidate_answer": candidate_answer or trace.final_answer,
                "num_images": len(query_input.images),
            },
        )

        try:
            response = self.llm_client.create_response(
                system_prompt=system_prompt,
                user_text=user_prompt,
                images=[
                    ImagePayload.from_image_ref(image)
                    for image in query_input.images
                ]
                if self.config.use_images
                else None,
                model=model or self.config.model,
                max_output_tokens=self.config.max_output_tokens,
                temperature=self.config.temperature,
            )
            verification, parsed_output = self.parse_output(response.text)
        except Exception as exc:
            agent_run.mark_failed(str(exc))
            trace.add_agent_run(agent_run)
            raise

        agent_run.raw_output = response.text
        agent_run.parsed_output = parsed_output
        agent_run.mark_succeeded()
        trace.add_agent_run(agent_run)
        trace.set_verification(verification)
        return VerifierAgentResult(
            verification=verification,
            raw_output=response.text,
            parsed_output=parsed_output,
            agent_run=agent_run,
        )

    def parse_output(self, output: str) -> tuple[VerificationResult, dict[str, Any]]:
        """Parse XML/tag-style verifier output into VerificationResult."""
        parsed = {
            "verification_decision": self._require_tag(output, "verification_decision"),
            "quality_score": self._require_tag(output, "quality_score"),
            "reasoning": extract_tag(output, "reasoning") or "",
            "feedback_type": extract_tag(output, "feedback_type") or "none",
            "feedback": extract_tag(output, "feedback") or "",
            "missing_slots": extract_tag(output, "missing_slots") or "[]",
            "primitive_gap_hypotheses": extract_tag(output, "primitive_gap_hypotheses") or "[]",
        }

        decision = VerificationDecision.coerce(parsed["verification_decision"].lower())
        score, score_meta = self._parse_quality_score(parsed["quality_score"])
        feedback_type = FeedbackType.coerce(parsed["feedback_type"].lower())
        feedback = parsed["feedback"].strip()
        if feedback.lower() == "none":
            feedback = ""
        if decision is VerificationDecision.ACCEPT:
            feedback_type = FeedbackType.NONE
            feedback = ""
        if (
            decision is VerificationDecision.REJECT
            and feedback_type is FeedbackType.NONE
            and self.config.require_feedback_on_reject
        ):
            raise ValidationError("Rejected verifier output must include missing_slot or primitive_gap feedback")

        missing_slots = self._parse_string_list(parsed["missing_slots"])
        primitive_gap_hypotheses = self._parse_primitive_gap_hypotheses(
            parsed["primitive_gap_hypotheses"]
        )
        verification = VerificationResult(
            decision=decision,
            quality_score=score,
            feedback_type=feedback_type,
            feedback=feedback,
            reasoning=parsed["reasoning"],
            missing_slots=missing_slots,
            primitive_gap_hypotheses=primitive_gap_hypotheses,
            metadata=score_meta,
        )
        parsed["quality_score_normalized"] = score
        parsed["missing_slots_parsed"] = missing_slots
        parsed["primitive_gap_hypotheses_parsed"] = [
            hypothesis.hypothesis_id for hypothesis in primitive_gap_hypotheses
        ]
        return verification, parsed

    def _coerce_query_input(
        self,
        query_input: QueryInput | str,
        trace: ExecutionTrace,
    ) -> QueryInput:
        if isinstance(query_input, QueryInput):
            return query_input
        if isinstance(query_input, str):
            return QueryInput(query=query_input, query_id=trace.query_id)
        if trace.requirements is not None:
            return QueryInput(query=trace.requirements.query, query_id=trace.query_id)
        raise TypeError(f"Unsupported query_input type: {type(query_input).__name__}")

    def _require_tag(self, text: str, tag: str) -> str:
        value = extract_tag(text, tag)
        if value is None:
            raise LLMOutputError(f"Verifier output missing <{tag}> tag")
        return value.strip()

    def _parse_quality_score(self, raw: str) -> tuple[float, dict[str, Any]]:
        text = raw.strip()
        is_percent = text.endswith("%")
        if is_percent:
            text = text[:-1].strip()
        try:
            score = float(text)
        except ValueError as exc:
            raise LLMOutputError(f"Invalid verifier quality_score: {raw!r}") from exc
        original_score = score
        normalized_from = "0-1"
        if is_percent:
            score = score / 100.0
            normalized_from = "percent"
        elif score > 1.0 and self.config.normalize_legacy_scores:
            if score <= 10.0:
                score = score / 10.0
                normalized_from = "0-10"
            elif score <= 100.0:
                score = score / 100.0
                normalized_from = "0-100"
        if not 0.0 <= score <= 1.0:
            raise LLMOutputError(f"Verifier quality_score must normalize to [0, 1], got {raw!r}")
        return score, {
            "original_quality_score": original_score,
            "score_normalized_from": normalized_from,
        }

    def _parse_string_list(self, raw: str) -> list[str]:
        text = raw.strip()
        if not text or text.lower() == "none":
            return []
        try:
            values = parse_json_array(text)
            return [str(item) for item in values]
        except Exception:
            return [
                line.strip("-* \t")
                for line in text.splitlines()
                if line.strip("-* \t")
            ]

    def _parse_primitive_gap_hypotheses(self, raw: str) -> list[PrimitiveHypothesis]:
        text = raw.strip()
        if not text or text.lower() == "none":
            return []
        rows = parse_json_array(text)
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
                    metadata=row.get("metadata", {}),
                )
            )
        return hypotheses

