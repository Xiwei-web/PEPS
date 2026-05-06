"""Execution trace data structures for PEPS."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import uuid
from typing import Any

from peps.core.enums import (
    FeedbackType,
    ToolCallStatus,
    TraceStage,
    VerificationDecision,
)
from peps.core.types import FESMRequirementSet, PrimitiveHypothesis


def utc_now() -> str:
    """Return a compact ISO timestamp in UTC."""
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(slots=True)
class AgentRunRecord:
    """Prompt/response record for one agent invocation."""

    agent_name: str
    stage: TraceStage | str
    prompt_name: str
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    input_summary: dict[str, Any] = field(default_factory=dict)
    raw_output: str | None = None
    parsed_output: dict[str, Any] | None = None
    status: ToolCallStatus | str = ToolCallStatus.PLANNED
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stage = TraceStage.coerce(self.stage)
        self.status = ToolCallStatus.coerce(self.status)

    def mark_succeeded(self) -> None:
        self.status = ToolCallStatus.SUCCEEDED
        self.ended_at = utc_now()

    def mark_failed(self, error: str) -> None:
        self.status = ToolCallStatus.FAILED
        self.error = error
        self.ended_at = utc_now()


@dataclass(slots=True)
class ToolCallRecord:
    """A concrete tool call made to ground one or more primitive instances."""

    tool_name: str
    arguments: dict[str, Any]
    fills: list[str] = field(default_factory=list)
    call_id: str = field(default_factory=lambda: new_id("tool"))
    status: ToolCallStatus | str = ToolCallStatus.PLANNED
    started_at: str | None = None
    ended_at: str | None = None
    result_refs: dict[str, str] = field(default_factory=dict)
    result_preview: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = ToolCallStatus.coerce(self.status)
        if not self.tool_name:
            raise ValueError("ToolCallRecord.tool_name must be non-empty")

    def mark_running(self) -> None:
        self.status = ToolCallStatus.RUNNING
        self.started_at = utc_now()

    def mark_succeeded(
        self,
        result_refs: dict[str, str] | None = None,
        result_preview: dict[str, Any] | None = None,
    ) -> None:
        self.status = ToolCallStatus.SUCCEEDED
        self.ended_at = utc_now()
        if result_refs:
            self.result_refs.update(result_refs)
        if result_preview:
            self.result_preview.update(result_preview)

    def mark_failed(self, error: str) -> None:
        self.status = ToolCallStatus.FAILED
        self.error = error
        self.ended_at = utc_now()


@dataclass(slots=True)
class CodeExecutionRecord:
    """A deterministic code execution produced by the Coder Agent."""

    code: str
    inputs: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: new_id("code"))
    status: ToolCallStatus | str = ToolCallStatus.PLANNED
    started_at: str | None = None
    ended_at: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    computed_metrics: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = ToolCallStatus.coerce(self.status)
        if not self.code.strip():
            raise ValueError("CodeExecutionRecord.code must be non-empty")

    def mark_running(self) -> None:
        self.status = ToolCallStatus.RUNNING
        self.started_at = utc_now()

    def mark_succeeded(
        self,
        outputs: dict[str, Any],
        answer: str | None = None,
        computed_metrics: dict[str, Any] | None = None,
    ) -> None:
        self.status = ToolCallStatus.SUCCEEDED
        self.ended_at = utc_now()
        self.outputs = outputs
        self.answer = answer
        if computed_metrics:
            self.computed_metrics = computed_metrics

    def mark_failed(self, error: str) -> None:
        self.status = ToolCallStatus.FAILED
        self.error = error
        self.ended_at = utc_now()


@dataclass(slots=True)
class VerificationResult:
    """Verifier output for one candidate answer and trace."""

    decision: VerificationDecision | str
    quality_score: float
    feedback_type: FeedbackType | str = FeedbackType.NONE
    feedback: str = ""
    reasoning: str = ""
    missing_slots: list[str] = field(default_factory=list)
    primitive_gap_hypotheses: list[PrimitiveHypothesis] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.decision = VerificationDecision.coerce(self.decision)
        self.feedback_type = FeedbackType.coerce(self.feedback_type)
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("VerificationResult.quality_score must be in [0, 1]")
        if self.decision is VerificationDecision.ACCEPT and self.feedback_type is not FeedbackType.NONE:
            raise ValueError("Accepted results must use feedback_type='none'")
        if self.feedback_type is FeedbackType.MISSING_SLOT and not self.feedback:
            raise ValueError("Missing-slot feedback requires feedback text")
        if self.feedback_type is FeedbackType.PRIMITIVE_GAP and not self.feedback:
            raise ValueError("Primitive-gap feedback requires feedback text")


@dataclass(slots=True)
class ExecutionTrace:
    """Full trace for a PEPS workflow attempt."""

    query_id: str | None = None
    trace_id: str = field(default_factory=lambda: new_id("trace"))
    created_at: str = field(default_factory=utc_now)
    schema_version: str | None = None
    requirements: FESMRequirementSet | None = None
    agent_runs: list[AgentRunRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    code_runs: list[CodeExecutionRecord] = field(default_factory=list)
    workspace_snapshot: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    computed_metrics: dict[str, Any] = field(default_factory=dict)
    verification: VerificationResult | None = None
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_requirements(self, requirements: FESMRequirementSet) -> None:
        self.requirements = requirements
        self.schema_version = requirements.schema_version

    def add_agent_run(self, record: AgentRunRecord) -> None:
        self.agent_runs.append(record)

    def add_tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls.append(record)

    def add_code_run(self, record: CodeExecutionRecord) -> None:
        self.code_runs.append(record)
        if record.answer is not None:
            self.final_answer = record.answer
        if record.computed_metrics:
            self.computed_metrics.update(record.computed_metrics)

    def set_verification(self, result: VerificationResult) -> None:
        self.verification = result

    def latest_code_run(self) -> CodeExecutionRecord | None:
        return self.code_runs[-1] if self.code_runs else None

    def accepted(self) -> bool:
        return (
            self.verification is not None
            and self.verification.decision is VerificationDecision.ACCEPT
        )

