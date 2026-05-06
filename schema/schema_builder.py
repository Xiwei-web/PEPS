"""Utilities for constructing and updating primitive libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from peps.core.types import PrimitiveDefinition
from peps.schema.candidate_pool import CandidatePrimitivePool
from peps.schema.primitive_library import PrimitiveLibrary
from peps.schema.schema_admission import (
    AdmissionCriteria,
    AdmissionDecision,
    admit_candidate,
)


@dataclass(slots=True)
class SchemaBuildReport:
    added_ids: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    admissions: list[AdmissionDecision] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PrimitiveSchemaBuilder:
    """Build or update a persistent PEPS primitive library."""

    def __init__(self, library: PrimitiveLibrary | None = None) -> None:
        self.library = library or PrimitiveLibrary()

    @classmethod
    def from_file(cls, path: str | Path) -> "PrimitiveSchemaBuilder":
        return cls(PrimitiveLibrary.from_file(path))

    def add_definition(
        self,
        definition: PrimitiveDefinition,
        *,
        overwrite: bool = False,
    ) -> SchemaBuildReport:
        report = SchemaBuildReport()
        if self.library.contains(definition.id) and not overwrite:
            report.skipped.append(definition.id)
            return report
        self.library.add(definition, overwrite=overwrite)
        report.added_ids.append(definition.id)
        return report

    def admit_candidates(
        self,
        pool: CandidatePrimitivePool,
        *,
        criteria: AdmissionCriteria | None = None,
        overwrite: bool = False,
    ) -> SchemaBuildReport:
        report = SchemaBuildReport()
        for record in pool.active():
            decision = admit_candidate(
                record,
                self.library,
                criteria,
                overwrite=overwrite,
            )
            report.admissions.append(decision)
            if not decision.admit or decision.promoted_definition is None:
                report.skipped.append(record.hypothesis_id)
                continue
            report.added_ids.append(decision.promoted_definition.id)
        return report

    def save(self, path: str | Path) -> None:
        self.library.save(path)
