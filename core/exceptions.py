"""Custom exceptions for PEPS."""

from __future__ import annotations


class PepsError(Exception):
    """Base exception for all PEPS-specific errors."""


class SchemaError(PepsError):
    """Raised when a primitive schema or library is malformed."""


class ValidationError(PepsError):
    """Raised when an object violates a PEPS contract."""


class RegistryError(PepsError):
    """Raised for duplicate, missing, or incompatible registry entries."""


class SerializationError(PepsError):
    """Raised when reading or writing PEPS artifacts fails."""


class TraceError(PepsError):
    """Raised when an execution trace is inconsistent."""


class WorkflowStateError(PepsError):
    """Raised when the workflow state cannot satisfy the next transition."""


class ToolExecutionError(PepsError):
    """Raised when a PEPS tool call fails."""


class LLMOutputError(PepsError):
    """Raised when an LLM response cannot be parsed into the expected contract."""

