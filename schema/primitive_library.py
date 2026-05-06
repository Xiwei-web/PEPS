"""Persistent primitive library for PEPS."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

from peps.core.enums import PrimitiveFamily
from peps.core.exceptions import SchemaError, SerializationError
from peps.core.io import primitive_definition_from_dict, to_plain_data, write_json
from peps.core.registry import PrimitiveRegistry
from peps.core.types import PrimitiveDefinition
from peps.schema.primitive_types import DEFAULT_SCHEMA_VERSION, FESM_ORDER
from peps.schema.retrieval import RankedItem, rank_primitives
from peps.schema.validators import validate_library_definitions


def _load_json_compatible_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise SerializationError(f"Failed to read primitive library: {source}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise SerializationError(
            (
                f"{source} is not JSON-compatible YAML and PyYAML is not installed. "
                "Use JSON-compatible YAML or install pyyaml."
            )
        ) from exc

    try:
        loaded = yaml.safe_load(text)
    except Exception as exc:  # pragma: no cover - depends on optional PyYAML
        raise SerializationError(f"Failed to parse YAML file: {source}") from exc
    if not isinstance(loaded, dict):
        raise SerializationError(f"Primitive library must be a mapping: {source}")
    return loaded


@dataclass(slots=True)
class PrimitiveLibrary:
    """Versioned collection of persistent PEPS primitive definitions."""

    schema_version: str = DEFAULT_SCHEMA_VERSION
    description: str = ""
    registry: PrimitiveRegistry = field(default_factory=PrimitiveRegistry)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "PrimitiveLibrary":
        raw = _load_json_compatible_yaml(path)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrimitiveLibrary":
        library = cls(
            schema_version=data.get("schema_version", DEFAULT_SCHEMA_VERSION),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )
        primitives = data.get("primitives", {})
        if isinstance(primitives, dict):
            for family in FESM_ORDER:
                for row in primitives.get(family, []):
                    payload = dict(row)
                    payload.setdefault("family", family)
                    library.add(primitive_definition_from_dict(payload))
        elif isinstance(primitives, list):
            for row in primitives:
                library.add(primitive_definition_from_dict(row))
        else:
            raise SchemaError("Primitive library 'primitives' must be a dict or list")
        library.validate(raise_on_error=True)
        return library

    def to_dict(self) -> dict[str, Any]:
        primitives: dict[str, list[dict[str, Any]]] = {family: [] for family in FESM_ORDER}
        for definition in self.registry.values():
            primitives[definition.family.value].append(to_plain_data(definition))
        return {
            "schema_version": self.schema_version,
            "description": self.description,
            "metadata": to_plain_data(self.metadata),
            "primitives": primitives,
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    def add(self, definition: PrimitiveDefinition, *, overwrite: bool = False) -> PrimitiveDefinition:
        return self.registry.register(definition, overwrite=overwrite)

    def add_many(
        self,
        definitions: Iterable[PrimitiveDefinition],
        *,
        overwrite: bool = False,
    ) -> None:
        self.registry.register_many(definitions, overwrite=overwrite)

    def get(self, definition_id: str) -> PrimitiveDefinition:
        return self.registry.get(definition_id)

    def maybe_get(self, definition_id: str) -> PrimitiveDefinition | None:
        return self.registry.maybe_get(definition_id)

    def contains(self, definition_id: str) -> bool:
        return self.registry.contains(definition_id)

    def definitions(self) -> list[PrimitiveDefinition]:
        return self.registry.values()

    def definitions_by_family(self, family: PrimitiveFamily | str) -> list[PrimitiveDefinition]:
        return self.registry.by_family(family)

    def as_mapping(self) -> dict[str, PrimitiveDefinition]:
        return dict(self.registry.items())

    def search(
        self,
        query: str,
        *,
        family: PrimitiveFamily | str | None = None,
        top_k: int = 8,
        min_score: float = 0.0,
    ) -> list[RankedItem[PrimitiveDefinition]]:
        definitions = (
            self.definitions_by_family(family)
            if family is not None
            else self.definitions()
        )
        return rank_primitives(query, definitions, top_k=top_k, min_score=min_score)

    def validate(self, *, raise_on_error: bool = False):
        return validate_library_definitions(self.as_mapping(), raise_on_error=raise_on_error)

