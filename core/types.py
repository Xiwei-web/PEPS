"""Core PEPS data types.

These classes define the stable contracts between Parser, Executor, Coder,
Verifier, and the workflow orchestrator. They intentionally contain no LLM,
tool, or graph execution logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, TypeAlias

from peps.core.enums import PrimitiveFamily, RequirementStatus, ValueSource

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}:{digest}"


@dataclass(slots=True)
class ImageRef:
    """Reference to an input image or view used by a PEPS query."""

    uri: str
    view_id: str | None = None
    role: str = "input"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("ImageRef.uri must be non-empty")


@dataclass(slots=True)
class QueryInput:
    """A single PEPS query with optional answer choices and image references."""

    query: str
    images: list[ImageRef] = field(default_factory=list)
    query_id: str | None = None
    choices: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("QueryInput.query must be non-empty")


@dataclass(slots=True)
class PrimitiveDefinition:
    """A persistent schema entry in the PEPS primitive library."""

    id: str
    family: PrimitiveFamily | str
    name: str
    description: str
    arguments_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    computable_by: list[str] = field(default_factory=list)
    version: str = "0.1.0"
    tags: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.family = PrimitiveFamily.coerce(self.family)
        if not self.id:
            raise ValueError("PrimitiveDefinition.id must be non-empty")
        if not self.name:
            self.name = self.id
        if not self.description:
            raise ValueError("PrimitiveDefinition.description must be non-empty")


@dataclass(slots=True)
class PrimitiveInstance:
    """A query-specific primitive selected by the Parser."""

    definition_id: str
    family: PrimitiveFamily | str
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    instance_id: str | None = None
    dependencies: list[str] = field(default_factory=list)
    status: RequirementStatus | str = RequirementStatus.PENDING
    value_ref: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.family = PrimitiveFamily.coerce(self.family)
        self.status = RequirementStatus.coerce(self.status)
        if not self.definition_id:
            raise ValueError("PrimitiveInstance.definition_id must be non-empty")
        if self.instance_id is None:
            self.instance_id = _stable_id(
                self.definition_id,
                {
                    "definition_id": self.definition_id,
                    "family": self.family.value,
                    "arguments": self.arguments,
                },
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("PrimitiveInstance.confidence must be in [0, 1]")


@dataclass(slots=True)
class FESMRequirementSet:
    """Parser output: the required Frame, Entity, State, and Metric primitives."""

    query: str
    schema_version: str
    frame: list[PrimitiveInstance] = field(default_factory=list)
    entity: list[PrimitiveInstance] = field(default_factory=list)
    state: list[PrimitiveInstance] = field(default_factory=list)
    metric: list[PrimitiveInstance] = field(default_factory=list)
    reasoning: str = ""
    minimality_rationale: str = ""
    uncertainties: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("FESMRequirementSet.query must be non-empty")
        if not self.schema_version:
            raise ValueError("FESMRequirementSet.schema_version must be non-empty")

    def all_instances(self) -> list[PrimitiveInstance]:
        """Return primitive instances in executable dependency order."""
        return [*self.frame, *self.entity, *self.state, *self.metric]

    def by_family(self, family: PrimitiveFamily | str) -> list[PrimitiveInstance]:
        family = PrimitiveFamily.coerce(family)
        if family is PrimitiveFamily.FRAME:
            return self.frame
        if family is PrimitiveFamily.ENTITY:
            return self.entity
        if family is PrimitiveFamily.STATE:
            return self.state
        if family is PrimitiveFamily.METRIC:
            return self.metric
        raise ValueError(f"Unknown primitive family: {family}")

    def definition_ids(self) -> set[str]:
        return {instance.definition_id for instance in self.all_instances()}

    def instance_ids(self) -> set[str]:
        return {instance.instance_id for instance in self.all_instances() if instance.instance_id}


@dataclass(slots=True)
class WorkspaceValue:
    """A grounded value stored in the Executor/Coder workspace."""

    name: str
    value: Any
    source: ValueSource | str
    primitive_instance_id: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source = ValueSource.coerce(self.source)
        if not self.name:
            raise ValueError("WorkspaceValue.name must be non-empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("WorkspaceValue.confidence must be in [0, 1]")


@dataclass(slots=True)
class PrimitiveHypothesis:
    """A temporary primitive proposed after primitive-gap feedback."""

    family: PrimitiveFamily | str
    name: str
    description: str
    arguments_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    proposed_by_trace_id: str | None = None
    trigger_query: str | None = None
    hypothesis_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.family = PrimitiveFamily.coerce(self.family)
        if not self.name:
            raise ValueError("PrimitiveHypothesis.name must be non-empty")
        if not self.description:
            raise ValueError("PrimitiveHypothesis.description must be non-empty")
        if self.hypothesis_id is None:
            self.hypothesis_id = _stable_id(
                f"hypothesis.{self.family.value}.{self.name}",
                {
                    "family": self.family.value,
                    "name": self.name,
                    "description": self.description,
                    "arguments_schema": self.arguments_schema,
                    "output_schema": self.output_schema,
                },
            )

