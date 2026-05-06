"""Workspace storage and reference resolution for PEPS tools."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import re
from typing import Any

from peps.core.enums import ValueSource
from peps.core.exceptions import WorkflowStateError
from peps.core.io import to_plain_data
from peps.core.types import WorkspaceValue


OPS_RE = re.compile(r"(\.[A-Za-z0-9_]+|\[.*?\])")


class ToolWorkspace:
    """Tool workspace with GCA-style `$variable.attr[0]` reference resolution."""

    def __init__(self, values: dict[str, WorkspaceValue] | None = None) -> None:
        self._values: dict[str, WorkspaceValue] = values or {}

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any], *, source: ValueSource | str = ValueSource.TOOL) -> "ToolWorkspace":
        workspace = cls()
        for key, value in mapping.items():
            workspace.set(key, value, source=source)
        return workspace

    def set(
        self,
        name: str,
        value: Any,
        *,
        source: ValueSource | str = ValueSource.TOOL,
        primitive_instance_id: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceValue:
        workspace_value = WorkspaceValue(
            name=name,
            value=value,
            source=source,
            primitive_instance_id=primitive_instance_id,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._values[name] = workspace_value
        return workspace_value

    def get(self, name: str) -> WorkspaceValue:
        try:
            return self._values[name]
        except KeyError as exc:
            raise WorkflowStateError(
                f"Workspace value {name!r} not found. Available: {sorted(self._values)}"
            ) from exc

    def get_raw(self, name: str) -> Any:
        return self.get(name).value

    def contains(self, name: str) -> bool:
        return name in self._values

    def names(self) -> list[str]:
        return list(self._values.keys())

    def values(self) -> dict[str, WorkspaceValue]:
        return dict(self._values)

    def raw_values(self) -> dict[str, Any]:
        return {key: value.value for key, value in self._values.items()}

    def to_dict(self) -> dict[str, Any]:
        return to_plain_data(self._values)

    def resolve(self, value: Any) -> Any:
        """Recursively resolve `$workspace` references in tool arguments."""
        if isinstance(value, dict):
            return {key: self.resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.resolve(item) for item in value)
        if isinstance(value, str) and value.strip().startswith("$"):
            return self.resolve_reference(value)
        return value

    def resolve_reference(self, reference: str) -> Any:
        clean = reference.strip()[1:]
        match = re.match(r"^([A-Za-z0-9_]+)(.*)$", clean)
        if not match:
            raise WorkflowStateError(f"Invalid workspace reference: {reference}")
        name, ops = match.groups()
        value = self.get_raw(name)
        if not ops:
            return value
        return self._apply_ops(value, ops, reference)

    def _apply_ops(self, value: Any, ops: str, original_reference: str) -> Any:
        for op in OPS_RE.findall(ops):
            try:
                if op.startswith("."):
                    value = self._get_attr(value, op[1:])
                else:
                    value = value[self._parse_index(op[1:-1])]
            except Exception as exc:
                raise WorkflowStateError(
                    f"Could not resolve reference {original_reference!r} at op {op!r}: {exc}"
                ) from exc
        return value

    def _get_attr(self, value: Any, attr: str) -> Any:
        if isinstance(value, dict):
            return value[attr]
        if is_dataclass(value):
            valid = {field.name for field in fields(value)}
            if attr in valid:
                return getattr(value, attr)
        return getattr(value, attr)

    def _parse_index(self, text: str) -> Any:
        if "," in text:
            return tuple(self._parse_index(part.strip()) for part in text.split(","))
        if ":" in text:
            parts = text.split(":")
            if len(parts) > 3:
                raise ValueError(f"Invalid slice: {text}")
            values = [int(part) if part else None for part in parts]
            return slice(*values)
        return int(text)

