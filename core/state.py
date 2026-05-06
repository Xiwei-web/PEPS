"""Mutable workflow state for one PEPS query."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from peps.core.enums import RequirementStatus, ValueSource
from peps.core.exceptions import WorkflowStateError
from peps.core.trace import ExecutionTrace, VerificationResult
from peps.core.types import (
    FESMRequirementSet,
    PrimitiveInstance,
    QueryInput,
    WorkspaceValue,
)


@dataclass(slots=True)
class WorkflowState:
    """State passed through Parser, Executor, Coder, and Verifier."""

    query_input: QueryInput
    max_rounds: int = 3
    round_index: int = 0
    requirements: FESMRequirementSet | None = None
    workspace: dict[str, WorkspaceValue] = field(default_factory=dict)
    trace: ExecutionTrace = field(default_factory=ExecutionTrace)
    last_verification: VerificationResult | None = None
    candidate_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("WorkflowState.max_rounds must be >= 1")
        self.trace.query_id = self.query_input.query_id

    def set_requirements(self, requirements: FESMRequirementSet) -> None:
        self.requirements = requirements
        self.trace.set_requirements(requirements)

    def register_value(
        self,
        name: str,
        value: Any,
        source: ValueSource | str,
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
        self.workspace[name] = workspace_value
        self.trace.workspace_snapshot[name] = value
        return workspace_value

    def get_value(self, name: str) -> WorkspaceValue:
        try:
            return self.workspace[name]
        except KeyError as exc:
            raise WorkflowStateError(f"Workspace value not found: {name}") from exc

    def unresolved_instances(self) -> list[PrimitiveInstance]:
        if self.requirements is None:
            return []
        unresolved: list[PrimitiveInstance] = []
        for instance in self.requirements.all_instances():
            if instance.status is RequirementStatus.GROUNDED:
                continue
            if instance.value_ref and instance.value_ref in self.workspace:
                continue
            unresolved.append(instance)
        return unresolved

    def grounded_instances(self) -> list[PrimitiveInstance]:
        if self.requirements is None:
            return []
        return [
            instance
            for instance in self.requirements.all_instances()
            if instance.status is RequirementStatus.GROUNDED
            or (instance.value_ref is not None and instance.value_ref in self.workspace)
        ]

    def can_refine(self) -> bool:
        return self.round_index + 1 < self.max_rounds

    def advance_round(self) -> None:
        if not self.can_refine():
            raise WorkflowStateError("Maximum PEPS refinement rounds reached")
        self.round_index += 1

    def set_verification(self, result: VerificationResult) -> None:
        self.last_verification = result
        self.trace.set_verification(result)

