"""Enum definitions for PEPS core contracts."""

from __future__ import annotations

from enum import StrEnum


class PepsEnum(StrEnum):
    """Base enum with small convenience helpers."""

    @classmethod
    def coerce(cls, value: str | "PepsEnum") -> "PepsEnum":
        """Convert a string or enum member into this enum class."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            for member in cls:
                if normalized == member.value or normalized.upper() == member.name:
                    return member
        raise ValueError(f"Cannot coerce {value!r} to {cls.__name__}")

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]


class PrimitiveFamily(PepsEnum):
    """The four PEPS primitive families."""

    FRAME = "frame"
    ENTITY = "entity"
    STATE = "state"
    METRIC = "metric"


class RequirementStatus(PepsEnum):
    """Lifecycle state of a primitive instance."""

    PENDING = "pending"
    GROUNDED = "grounded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValueSource(PepsEnum):
    """Where a workspace value came from."""

    PARSER = "parser"
    TOOL = "tool"
    CODE = "code"
    VERIFIER = "verifier"
    CACHE = "cache"
    HUMAN = "human"


class ToolCallStatus(PepsEnum):
    """Lifecycle state of a tool or code call."""

    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class VerificationDecision(PepsEnum):
    """Verifier decision for a candidate answer."""

    ACCEPT = "accept"
    REJECT = "reject"


class FeedbackType(PepsEnum):
    """Typed verifier feedback used by the PEPS refinement loop."""

    NONE = "none"
    MISSING_SLOT = "missing_slot"
    PRIMITIVE_GAP = "primitive_gap"


class TraceStage(PepsEnum):
    """Major stages that may contribute records to a trace."""

    PARSER = "parser"
    EXECUTOR = "executor"
    CODER = "coder"
    VERIFIER = "verifier"
    WORKFLOW = "workflow"

