"""Schema-level validation for primitive libraries and instances."""

from __future__ import annotations

from typing import Any, Mapping

from peps.core.enums import PrimitiveFamily
from peps.core.exceptions import ValidationError
from peps.core.types import (
    FESMRequirementSet,
    PrimitiveDefinition,
    PrimitiveInstance,
)
from peps.core.validation import ValidationReport, validate_requirement_set


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def validate_arguments(
    arguments: Mapping[str, Any],
    arguments_schema: Mapping[str, Any],
    report: ValidationReport,
    path: str,
) -> None:
    """Validate a small, JSON-Schema-like arguments schema."""
    required = arguments_schema.get("required", [])
    properties = arguments_schema.get("properties", {})

    for key in required:
        if key not in arguments:
            report.add_error(f"Missing required argument: {key}", f"{path}.{key}")

    for key, value in arguments.items():
        spec = properties.get(key)
        if spec is None:
            continue
        expected_type = spec.get("type")
        if expected_type and not _type_matches(value, expected_type):
            report.add_error(
                f"Argument {key!r} must have type {expected_type!r}",
                f"{path}.{key}",
            )
        if "enum" in spec and value not in spec["enum"]:
            report.add_error(
                f"Argument {key!r} must be one of {spec['enum']!r}",
                f"{path}.{key}",
            )


def validate_library_definitions(
    definitions: Mapping[str, PrimitiveDefinition],
    *,
    raise_on_error: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    for definition_id, definition in definitions.items():
        path = f"library.{definition_id}"
        if definition_id != definition.id:
            report.add_error("Definition key does not match definition.id", path)
        if definition.family not in set(PrimitiveFamily):
            report.add_error("Invalid primitive family", f"{path}.family")
        if not definition.computable_by:
            report.add_warning("Primitive has no computable_by tools", f"{path}.computable_by")
    if raise_on_error:
        report.raise_if_errors()
    return report


def validate_instance_against_definition(
    instance: PrimitiveInstance,
    definition: PrimitiveDefinition,
    report: ValidationReport,
    path: str,
) -> None:
    if instance.family is not definition.family:
        report.add_error(
            (
                f"Instance family {instance.family.value!r} does not match "
                f"definition family {definition.family.value!r}"
            ),
            f"{path}.family",
        )
    validate_arguments(instance.arguments, definition.arguments_schema, report, f"{path}.arguments")


def validate_requirement_set_against_library(
    requirements: FESMRequirementSet,
    definitions: Mapping[str, PrimitiveDefinition],
    *,
    raise_on_error: bool = False,
) -> ValidationReport:
    report = validate_requirement_set(requirements, definitions)
    for index, instance in enumerate(requirements.all_instances()):
        definition = definitions.get(instance.definition_id)
        if definition is None:
            continue
        validate_instance_against_definition(
            instance,
            definition,
            report,
            f"requirements.instances[{index}]",
        )
    if raise_on_error:
        report.raise_if_errors()
    return report


def ensure_valid_requirement_set(
    requirements: FESMRequirementSet,
    definitions: Mapping[str, PrimitiveDefinition],
) -> None:
    report = validate_requirement_set_against_library(
        requirements,
        definitions,
        raise_on_error=False,
    )
    if not report.ok:
        raise ValidationError(
            "\n".join(f"{issue.path}: {issue.message}" for issue in report.issues)
        )

