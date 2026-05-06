"""Validation helpers for PEPS core contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

from peps.core.enums import FeedbackType, PrimitiveFamily, VerificationDecision
from peps.core.exceptions import ValidationError
from peps.core.trace import ExecutionTrace, VerificationResult
from peps.core.types import (
    FESMRequirementSet,
    PrimitiveDefinition,
    PrimitiveInstance,
)


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(slots=True)
class ValidationIssue:
    severity: ValidationSeverity
    message: str
    path: str = ""


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    def add_error(self, message: str, path: str = "") -> None:
        self.issues.append(ValidationIssue(ValidationSeverity.ERROR, message, path))

    def add_warning(self, message: str, path: str = "") -> None:
        self.issues.append(ValidationIssue(ValidationSeverity.WARNING, message, path))

    def raise_if_errors(self) -> None:
        if self.ok:
            return
        lines = [
            f"{issue.severity.value}: {issue.path}: {issue.message}"
            if issue.path
            else f"{issue.severity.value}: {issue.message}"
            for issue in self.issues
            if issue.severity is ValidationSeverity.ERROR
        ]
        raise ValidationError("\n".join(lines))


def validate_primitive_definition(
    definition: PrimitiveDefinition,
    report: ValidationReport | None = None,
    path: str = "",
) -> ValidationReport:
    report = report or ValidationReport()
    if not definition.id:
        report.add_error("Primitive definition id is required", path)
    if not definition.description:
        report.add_error("Primitive definition description is required", path)
    if not isinstance(definition.arguments_schema, dict):
        report.add_error("arguments_schema must be a dict", f"{path}.arguments_schema")
    if not isinstance(definition.output_schema, dict):
        report.add_error("output_schema must be a dict", f"{path}.output_schema")
    return report


def validate_primitive_instance(
    instance: PrimitiveInstance,
    definitions: Mapping[str, PrimitiveDefinition] | None = None,
    report: ValidationReport | None = None,
    path: str = "",
) -> ValidationReport:
    report = report or ValidationReport()
    if not instance.definition_id:
        report.add_error("Primitive instance definition_id is required", path)
    if instance.confidence is not None and not 0.0 <= instance.confidence <= 1.0:
        report.add_error("Primitive confidence must be in [0, 1]", f"{path}.confidence")
    if definitions is not None:
        definition = definitions.get(instance.definition_id)
        if definition is None:
            report.add_error(
                f"Unknown primitive definition_id: {instance.definition_id}",
                f"{path}.definition_id",
            )
        elif definition.family is not instance.family:
            report.add_error(
                (
                    f"Instance family {instance.family.value!r} does not match "
                    f"definition family {definition.family.value!r}"
                ),
                f"{path}.family",
            )
    return report


def validate_requirement_set(
    requirements: FESMRequirementSet,
    definitions: Mapping[str, PrimitiveDefinition] | None = None,
    *,
    raise_on_error: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    family_lists: list[tuple[PrimitiveFamily, list[PrimitiveInstance], str]] = [
        (PrimitiveFamily.FRAME, requirements.frame, "frame"),
        (PrimitiveFamily.ENTITY, requirements.entity, "entity"),
        (PrimitiveFamily.STATE, requirements.state, "state"),
        (PrimitiveFamily.METRIC, requirements.metric, "metric"),
    ]
    seen_instance_ids: set[str] = set()
    all_instance_ids = requirements.instance_ids()

    for expected_family, instances, family_path in family_lists:
        for index, instance in enumerate(instances):
            path = f"requirements.{family_path}[{index}]"
            if instance.family is not expected_family:
                report.add_error(
                    (
                        f"Primitive listed under {family_path!r} has family "
                        f"{instance.family.value!r}"
                    ),
                    f"{path}.family",
                )
            validate_primitive_instance(instance, definitions, report, path)
            if instance.instance_id in seen_instance_ids:
                report.add_error(
                    f"Duplicate primitive instance_id: {instance.instance_id}",
                    f"{path}.instance_id",
                )
            seen_instance_ids.add(instance.instance_id)
            for dep in instance.dependencies:
                if dep not in all_instance_ids:
                    report.add_error(
                        f"Dependency does not refer to a known instance_id: {dep}",
                        f"{path}.dependencies",
                    )

    if not requirements.metric:
        report.add_warning("Requirement set has no metric primitives", "requirements.metric")
    if raise_on_error:
        report.raise_if_errors()
    return report


def validate_verification_result(
    result: VerificationResult,
    *,
    raise_on_error: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    if not 0.0 <= result.quality_score <= 1.0:
        report.add_error("quality_score must be in [0, 1]", "verification.quality_score")
    if result.decision is VerificationDecision.ACCEPT and result.feedback_type is not FeedbackType.NONE:
        report.add_error(
            "Accepted verification result must use feedback_type='none'",
            "verification.feedback_type",
        )
    if result.decision is VerificationDecision.REJECT and result.feedback_type is FeedbackType.NONE:
        report.add_warning(
            "Rejected verification result should include missing_slot or primitive_gap feedback",
            "verification.feedback_type",
        )
    if raise_on_error:
        report.raise_if_errors()
    return report


def validate_trace(
    trace: ExecutionTrace,
    *,
    raise_on_error: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    if trace.requirements is not None:
        requirement_report = validate_requirement_set(trace.requirements)
        report.issues.extend(requirement_report.issues)
    if trace.verification is not None:
        verification_report = validate_verification_result(trace.verification)
        report.issues.extend(verification_report.issues)
    if trace.verification is not None and trace.final_answer is None:
        report.add_warning("Trace has verification but no final answer", "trace.final_answer")
    if raise_on_error:
        report.raise_if_errors()
    return report

