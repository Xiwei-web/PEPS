"""Serialization helpers for PEPS core objects."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable

from peps.core.enums import FeedbackType, PrimitiveFamily, RequirementStatus, ToolCallStatus, TraceStage, ValueSource, VerificationDecision
from peps.core.exceptions import SerializationError
from peps.core.trace import (
    AgentRunRecord,
    CodeExecutionRecord,
    ExecutionTrace,
    ToolCallRecord,
    VerificationResult,
)
from peps.core.types import (
    FESMRequirementSet,
    ImageRef,
    PrimitiveDefinition,
    PrimitiveHypothesis,
    PrimitiveInstance,
    QueryInput,
    WorkspaceValue,
)


def to_plain_data(value: Any) -> Any:
    """Recursively convert PEPS objects into JSON-compatible Python data."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: to_plain_data(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(to_plain_data(data), ensure_ascii=True, indent=indent) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise SerializationError(f"Failed to write JSON file: {target}") from exc


def read_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SerializationError(f"Failed to read JSON file: {source}") from exc


def append_jsonl(path: str | Path, records: Iterable[Any] | Any) -> None:
    target = Path(path)
    if not isinstance(records, Iterable) or isinstance(records, (str, bytes, dict)):
        records = [records]
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(to_plain_data(record), ensure_ascii=True) + "\n")
    except OSError as exc:
        raise SerializationError(f"Failed to append JSONL file: {target}") from exc


def read_jsonl(path: str | Path) -> list[Any]:
    source = Path(path)
    try:
        rows: list[Any] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    except (OSError, json.JSONDecodeError) as exc:
        raise SerializationError(f"Failed to read JSONL file: {source}") from exc


def image_ref_from_dict(data: dict[str, Any]) -> ImageRef:
    return ImageRef(**data)


def query_input_from_dict(data: dict[str, Any]) -> QueryInput:
    payload = dict(data)
    payload["images"] = [image_ref_from_dict(item) for item in payload.get("images", [])]
    return QueryInput(**payload)


def primitive_definition_from_dict(data: dict[str, Any]) -> PrimitiveDefinition:
    payload = dict(data)
    payload["family"] = PrimitiveFamily.coerce(payload["family"])
    return PrimitiveDefinition(**payload)


def primitive_instance_from_dict(data: dict[str, Any]) -> PrimitiveInstance:
    payload = dict(data)
    payload["family"] = PrimitiveFamily.coerce(payload["family"])
    if "status" in payload:
        payload["status"] = RequirementStatus.coerce(payload["status"])
    return PrimitiveInstance(**payload)


def requirement_set_from_dict(data: dict[str, Any]) -> FESMRequirementSet:
    payload = dict(data)
    for key in ("frame", "entity", "state", "metric"):
        payload[key] = [primitive_instance_from_dict(item) for item in payload.get(key, [])]
    return FESMRequirementSet(**payload)


def workspace_value_from_dict(data: dict[str, Any]) -> WorkspaceValue:
    payload = dict(data)
    payload["source"] = ValueSource.coerce(payload["source"])
    return WorkspaceValue(**payload)


def primitive_hypothesis_from_dict(data: dict[str, Any]) -> PrimitiveHypothesis:
    payload = dict(data)
    payload["family"] = PrimitiveFamily.coerce(payload["family"])
    return PrimitiveHypothesis(**payload)


def verification_result_from_dict(data: dict[str, Any]) -> VerificationResult:
    payload = dict(data)
    payload["decision"] = VerificationDecision.coerce(payload["decision"])
    payload["feedback_type"] = FeedbackType.coerce(payload.get("feedback_type", FeedbackType.NONE))
    payload["primitive_gap_hypotheses"] = [
        primitive_hypothesis_from_dict(item)
        for item in payload.get("primitive_gap_hypotheses", [])
    ]
    return VerificationResult(**payload)


def agent_run_record_from_dict(data: dict[str, Any]) -> AgentRunRecord:
    payload = dict(data)
    payload["stage"] = TraceStage.coerce(payload["stage"])
    payload["status"] = ToolCallStatus.coerce(payload["status"])
    return AgentRunRecord(**payload)


def tool_call_record_from_dict(data: dict[str, Any]) -> ToolCallRecord:
    payload = dict(data)
    payload["status"] = ToolCallStatus.coerce(payload["status"])
    return ToolCallRecord(**payload)


def code_execution_record_from_dict(data: dict[str, Any]) -> CodeExecutionRecord:
    payload = dict(data)
    payload["status"] = ToolCallStatus.coerce(payload["status"])
    return CodeExecutionRecord(**payload)


def execution_trace_from_dict(data: dict[str, Any]) -> ExecutionTrace:
    payload = dict(data)
    if payload.get("requirements") is not None:
        payload["requirements"] = requirement_set_from_dict(payload["requirements"])
    payload["agent_runs"] = [
        agent_run_record_from_dict(item) for item in payload.get("agent_runs", [])
    ]
    payload["tool_calls"] = [
        tool_call_record_from_dict(item) for item in payload.get("tool_calls", [])
    ]
    payload["code_runs"] = [
        code_execution_record_from_dict(item) for item in payload.get("code_runs", [])
    ]
    if payload.get("verification") is not None:
        payload["verification"] = verification_result_from_dict(payload["verification"])
    return ExecutionTrace(**payload)

