"""Helpers for creating Parser-selected primitive instances."""

from __future__ import annotations

from typing import Any

from peps.core.enums import PrimitiveFamily
from peps.core.exceptions import ValidationError
from peps.core.types import FESMRequirementSet, PrimitiveDefinition, PrimitiveInstance
from peps.schema.primitive_library import PrimitiveLibrary
from peps.schema.validators import validate_instance_against_definition
from peps.core.validation import ValidationReport


def instantiate_primitive(
    definition: PrimitiveDefinition,
    arguments: dict[str, Any] | None = None,
    *,
    rationale: str = "",
    dependencies: list[str] | None = None,
    confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> PrimitiveInstance:
    instance = PrimitiveInstance(
        definition_id=definition.id,
        family=definition.family,
        arguments=arguments or {},
        rationale=rationale,
        dependencies=dependencies or [],
        confidence=confidence,
        metadata=metadata or {},
    )
    report = ValidationReport()
    validate_instance_against_definition(instance, definition, report, "primitive")
    report.raise_if_errors()
    return instance


def primitive_instance_from_parser_item(
    item: dict[str, Any],
    library: PrimitiveLibrary,
    expected_family: PrimitiveFamily | str,
) -> PrimitiveInstance:
    expected_family = PrimitiveFamily.coerce(expected_family)
    definition_id = (
        item.get("definition_id")
        or item.get("primitive_id")
        or item.get("id")
        or item.get("name")
    )
    if not definition_id:
        raise ValidationError(f"Parser item under {expected_family.value} lacks primitive id")
    definition = library.get(definition_id)
    if definition.family is not expected_family:
        raise ValidationError(
            (
                f"Parser selected {definition_id!r} under {expected_family.value!r}, "
                f"but the library defines it as {definition.family.value!r}"
            )
        )
    return instantiate_primitive(
        definition,
        item.get("arguments", {}),
        rationale=item.get("rationale", ""),
        dependencies=item.get("dependencies", []),
        confidence=item.get("confidence"),
        metadata=item.get("metadata", {}),
    )


def requirement_set_from_parser_output(
    parser_output: dict[str, Any],
    library: PrimitiveLibrary,
    *,
    query: str,
) -> FESMRequirementSet:
    """Build a typed requirement set from Parser JSON output."""
    requirements = parser_output.get("requirements", parser_output)
    if not isinstance(requirements, dict):
        raise ValidationError("Parser output must contain a 'requirements' mapping")

    payload: dict[str, list[PrimitiveInstance]] = {}
    for family in PrimitiveFamily:
        rows = requirements.get(family.value, [])
        if not isinstance(rows, list):
            raise ValidationError(f"Parser requirements.{family.value} must be a list")
        payload[family.value] = [
            primitive_instance_from_parser_item(item, library, family)
            for item in rows
        ]

    return FESMRequirementSet(
        query=query,
        schema_version=parser_output.get("schema_version", library.schema_version),
        frame=payload["frame"],
        entity=payload["entity"],
        state=payload["state"],
        metric=payload["metric"],
        reasoning=parser_output.get("reasoning", ""),
        minimality_rationale=parser_output.get("minimality_rationale", ""),
        uncertainties=parser_output.get("uncertainties", []),
        metadata=parser_output.get("metadata", {}),
    )

