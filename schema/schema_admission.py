"""Admission rules for promoting primitive hypotheses into the main library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from peps.core.enums import PrimitiveFamily
from peps.core.types import PrimitiveDefinition
from peps.schema.candidate_pool import CandidatePrimitiveRecord
from peps.schema.primitive_library import PrimitiveLibrary
from peps.schema.primitive_types import CORE_TOOL_NAMES


@dataclass(slots=True)
class AdmissionCriteria:
    """Conservative thresholds for candidate primitive promotion."""

    min_reuse_count: int = 5
    min_accepted_count: int = 1
    min_average_score: float = 0.7
    min_average_improvement: float = 0.0
    require_non_conflict: bool = True
    require_computable_by: bool = True
    allowed_families: set[str] = field(
        default_factory=lambda: {family.value for family in PrimitiveFamily}
    )


@dataclass(slots=True)
class AdmissionDecision:
    admit: bool
    reasons: list[str] = field(default_factory=list)
    promoted_definition: PrimitiveDefinition | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_candidate_for_admission(
    record: CandidatePrimitiveRecord,
    library: PrimitiveLibrary,
    criteria: AdmissionCriteria | None = None,
) -> AdmissionDecision:
    criteria = criteria or AdmissionCriteria()
    reasons: list[str] = []
    warnings: list[str] = []

    if record.status != "candidate":
        reasons.append(f"candidate status is {record.status!r}, expected 'candidate'")

    if record.hypothesis.family.value not in criteria.allowed_families:
        reasons.append(f"family {record.hypothesis.family.value!r} is not allowed")

    if not record.hypothesis.name.replace("_", "").replace("-", "").isalnum():
        reasons.append(f"hypothesis name is not a safe primitive suffix: {record.hypothesis.name!r}")

    if record.reuse_count < criteria.min_reuse_count:
        reasons.append(
            f"reuse_count {record.reuse_count} < {criteria.min_reuse_count}"
        )
    if record.accepted_count < criteria.min_accepted_count:
        reasons.append(
            f"accepted_count {record.accepted_count} < {criteria.min_accepted_count}"
        )
    if record.average_score < criteria.min_average_score:
        reasons.append(
            f"average_score {record.average_score:.3f} < {criteria.min_average_score:.3f}"
        )
    if record.average_improvement < criteria.min_average_improvement:
        reasons.append(
            (
                f"average_improvement {record.average_improvement:.3f} < "
                f"{criteria.min_average_improvement:.3f}"
            )
        )

    proposed_id = f"{record.hypothesis.family.value}.{record.hypothesis.name}"
    if criteria.require_non_conflict and library.contains(proposed_id):
        reasons.append(f"primitive id already exists: {proposed_id}")

    computable_by = record.hypothesis.metadata.get("computable_by", [])
    if criteria.require_computable_by and not computable_by:
        reasons.append("hypothesis metadata must include non-empty computable_by")
    unknown_tools = [tool for tool in computable_by if tool not in CORE_TOOL_NAMES]
    if unknown_tools:
        warnings.append(f"unknown computable_by tools: {unknown_tools}")

    if reasons:
        return AdmissionDecision(admit=False, reasons=reasons, metadata={"warnings": warnings})

    promoted = PrimitiveDefinition(
        id=proposed_id,
        family=record.hypothesis.family,
        name=record.hypothesis.name,
        description=record.hypothesis.description,
        arguments_schema=record.hypothesis.arguments_schema,
        output_schema=record.hypothesis.output_schema,
        computable_by=record.hypothesis.metadata.get("computable_by", []),
        tags=record.hypothesis.metadata.get("tags", ["promoted"]),
        examples=[
            {
                "query": query,
                "source": "candidate_pool",
            }
            for query in record.trigger_queries[:5]
        ],
        metadata={
            "promoted_from": record.hypothesis_id,
            "average_score": record.average_score,
            "average_improvement": record.average_improvement,
            "reuse_count": record.reuse_count,
            "accepted_count": record.accepted_count,
            "rejected_count": record.rejected_count,
        },
    )
    return AdmissionDecision(
        admit=True,
        reasons=["candidate satisfies admission criteria", *warnings],
        promoted_definition=promoted,
    )


def admit_candidate(
    record: CandidatePrimitiveRecord,
    library: PrimitiveLibrary,
    criteria: AdmissionCriteria | None = None,
    *,
    overwrite: bool = False,
) -> AdmissionDecision:
    """Evaluate one candidate and add it to the library if admitted."""
    decision = evaluate_candidate_for_admission(record, library, criteria)
    if decision.admit and decision.promoted_definition is not None:
        library.add(decision.promoted_definition, overwrite=overwrite)
        record.status = "promoted"
    return decision


def admit_candidates(
    records: list[CandidatePrimitiveRecord],
    library: PrimitiveLibrary,
    criteria: AdmissionCriteria | None = None,
    *,
    overwrite: bool = False,
) -> list[AdmissionDecision]:
    return [
        admit_candidate(record, library, criteria, overwrite=overwrite)
        for record in records
    ]
