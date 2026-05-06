"""Base interfaces for PEPS tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass(slots=True)
class LocalModelRequirement:
    """A model or package dependency that a tool backend may require."""

    name: str
    required_for: str
    local_path_env: str | None = None
    checkpoint_env: str | None = None
    package: str | None = None
    download_hint: str | None = None
    required: bool = False


@dataclass(slots=True)
class ToolSpec:
    """Public contract for one PEPS tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    model_requirements: list[LocalModelRequirement] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    is_placeholder: bool = False


@dataclass(slots=True)
class ToolRequest:
    """A concrete tool request emitted by the Executor."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    output_name: str | None = None
    fills: list[str] = field(default_factory=list)
    call_id: str = field(default_factory=lambda: f"tool_{uuid.uuid4().hex[:12]}")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolContext:
    """Runtime context passed into each tool call."""

    workspace: Any | None = None
    query: str | None = None
    images: list[Any] = field(default_factory=list)
    cache_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    """Structured result returned by every PEPS tool."""

    tool_name: str
    result: Any = None
    error: str | None = None
    error_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def raise_for_error(self) -> None:
        if self.error:
            source = f" from {self.error_source}" if self.error_source else ""
            raise RuntimeError(f"{self.tool_name} failed{source}: {self.error}")


class BaseTool:
    """Base class for all PEPS tools."""

    spec: ToolSpec

    def __init__(self, spec: ToolSpec | None = None) -> None:
        if spec is not None:
            self.spec = spec
        if not hasattr(self, "spec"):
            raise ValueError(f"{type(self).__name__} must define a ToolSpec")

    @property
    def name(self) -> str:
        return self.spec.name

    async def arun(self, *, context: ToolContext | None = None, **kwargs: Any) -> ToolResult:
        try:
            return await self._arun(context=context or ToolContext(), **kwargs)
        except Exception as exc:
            return self.error(str(exc), source=f"{type(self).__name__}.arun")

    async def _arun(self, *, context: ToolContext, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def success(self, result: Any, **metadata: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, result=result, metadata=metadata)

    def error(self, message: str, *, source: str | None = None, **metadata: Any) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            error=message,
            error_source=source or type(self).__name__,
            metadata=metadata,
        )


class UnavailableTool(BaseTool):
    """A registered tool slot whose backend will be implemented later."""

    async def _arun(self, *, context: ToolContext, **kwargs: Any) -> ToolResult:
        return self.error(
            "Tool backend is not implemented yet.",
            source=self.name,
            arguments=kwargs,
        )

