"""Candidate primitive pool for primitive-gap feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from peps.core.io import (
    append_jsonl,
    primitive_hypothesis_from_dict,
    read_json,
    read_jsonl,
    to_plain_data,
    write_json,
)
from peps.core.types import PrimitiveHypothesis
from peps.schema.retrieval import RankedItem, lexical_score


@dataclass(slots=True)
class CandidatePrimitiveRecord:
    """Usage and quality history for one temporary primitive hypothesis."""

    hypothesis: PrimitiveHypothesis
    status: str = "candidate"
    trigger_queries: list[str] = field(default_factory=list)
    verified_scores: list[float] = field(default_factory=list)
    score_improvements: list[float] = field(default_factory=list)
    accepted_trace_ids: list[str] = field(default_factory=list)
    rejected_trace_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def hypothesis_id(self) -> str:
        return self.hypothesis.hypothesis_id or ""

    @property
    def reuse_count(self) -> int:
        return len(set(self.trigger_queries))

    @property
    def average_score(self) -> float:
        if not self.verified_scores:
            return 0.0
        return sum(self.verified_scores) / len(self.verified_scores)

    @property
    def average_improvement(self) -> float:
        if not self.score_improvements:
            return 0.0
        return sum(self.score_improvements) / len(self.score_improvements)

    @property
    def accepted_count(self) -> int:
        return len(set(self.accepted_trace_ids))

    @property
    def rejected_count(self) -> int:
        return len(set(self.rejected_trace_ids))

    @property
    def latest_score(self) -> float | None:
        return self.verified_scores[-1] if self.verified_scores else None

    def search_text(self) -> str:
        return " ".join(
            [
                self.hypothesis.family.value,
                self.hypothesis.name,
                self.hypothesis.description,
                " ".join(self.trigger_queries),
                " ".join(str(tag) for tag in self.hypothesis.metadata.get("tags", [])),
            ]
        )

    def record_use(
        self,
        *,
        query: str,
        quality_score: float,
        score_improvement: float = 0.0,
        trace_id: str | None = None,
        accepted: bool = False,
    ) -> None:
        if not 0.0 <= quality_score <= 1.0:
            raise ValueError("quality_score must be in [0, 1]")
        self.trigger_queries.append(query)
        self.verified_scores.append(quality_score)
        self.score_improvements.append(score_improvement)
        if trace_id and accepted:
            self.accepted_trace_ids.append(trace_id)
        elif trace_id:
            self.rejected_trace_ids.append(trace_id)

    def to_dict(self) -> dict[str, Any]:
        return to_plain_data(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidatePrimitiveRecord":
        payload = dict(data)
        payload["hypothesis"] = primitive_hypothesis_from_dict(payload["hypothesis"])
        return cls(**payload)


class CandidatePrimitivePool:
    """Persistent collection of temporary primitive hypotheses."""

    def __init__(self, records: list[CandidatePrimitiveRecord] | None = None) -> None:
        self._records: dict[str, CandidatePrimitiveRecord] = {}
        for record in records or []:
            self.upsert(record)

    @classmethod
    def from_json(cls, path: str | Path) -> "CandidatePrimitivePool":
        source = Path(path)
        if not source.exists():
            return cls()
        return cls(CandidatePrimitiveRecord.from_dict(row) for row in read_json(source))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CandidatePrimitivePool":
        source = Path(path)
        if not source.exists():
            return cls()
        return cls(CandidatePrimitiveRecord.from_dict(row) for row in read_jsonl(source))

    @classmethod
    def load(cls, path: str | Path) -> "CandidatePrimitivePool":
        source = Path(path)
        if source.suffix == ".jsonl":
            return cls.from_jsonl(source)
        return cls.from_json(source)

    def save_json(self, path: str | Path) -> None:
        write_json(path, [record.to_dict() for record in self.records()])

    def save_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        if target.exists():
            target.unlink()
        append_jsonl(target, [record.to_dict() for record in self.records()])

    def append_record_jsonl(self, path: str | Path, record: CandidatePrimitiveRecord) -> None:
        append_jsonl(path, record.to_dict())

    def upsert(self, record: CandidatePrimitiveRecord) -> CandidatePrimitiveRecord:
        existing = self._records.get(record.hypothesis_id)
        if existing is None:
            self._records[record.hypothesis_id] = record
            return record
        existing.trigger_queries.extend(record.trigger_queries)
        existing.verified_scores.extend(record.verified_scores)
        existing.score_improvements.extend(record.score_improvements)
        existing.accepted_trace_ids.extend(record.accepted_trace_ids)
        existing.rejected_trace_ids.extend(record.rejected_trace_ids)
        if record.status != "candidate":
            existing.status = record.status
        existing.metadata.update(record.metadata)
        return existing

    def add_hypothesis(
        self,
        hypothesis: PrimitiveHypothesis,
        *,
        query: str | None = None,
        trace_id: str | None = None,
    ) -> CandidatePrimitiveRecord:
        record = CandidatePrimitiveRecord(hypothesis=hypothesis)
        if query:
            record.trigger_queries.append(query)
        if trace_id:
            record.rejected_trace_ids.append(trace_id)
        return self.upsert(record)

    def record_use(
        self,
        hypothesis_id: str,
        *,
        query: str,
        quality_score: float,
        score_improvement: float = 0.0,
        trace_id: str | None = None,
        accepted: bool = False,
    ) -> CandidatePrimitiveRecord:
        record = self.get(hypothesis_id)
        if record is None:
            raise KeyError(f"Candidate hypothesis not found: {hypothesis_id}")
        record.record_use(
            query=query,
            quality_score=quality_score,
            score_improvement=score_improvement,
            trace_id=trace_id,
            accepted=accepted,
        )
        return record

    def get(self, hypothesis_id: str) -> CandidatePrimitiveRecord | None:
        return self._records.get(hypothesis_id)

    def remove(self, hypothesis_id: str) -> CandidatePrimitiveRecord | None:
        return self._records.pop(hypothesis_id, None)

    def records(self) -> list[CandidatePrimitiveRecord]:
        return list(self._records.values())

    def active(self) -> list[CandidatePrimitiveRecord]:
        return [record for record in self.records() if record.status == "candidate"]

    def promoted(self) -> list[CandidatePrimitiveRecord]:
        return [record for record in self.records() if record.status == "promoted"]

    def rejected(self) -> list[CandidatePrimitiveRecord]:
        return [record for record in self.records() if record.status == "rejected"]

    def promotable(self, admission_fn) -> list[CandidatePrimitiveRecord]:
        return [record for record in self.active() if admission_fn(record).admit]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        active_only: bool = True,
    ) -> list[RankedItem[CandidatePrimitiveRecord]]:
        rows = self.active() if active_only else self.records()
        ranked: list[RankedItem[CandidatePrimitiveRecord]] = []
        for record in rows:
            score, matched = lexical_score(query, record.search_text())
            if score >= min_score:
                ranked.append(RankedItem(record, score, matched))
        ranked.sort(
            key=lambda item: (
                item.score,
                item.item.average_score,
                item.item.average_improvement,
            ),
            reverse=True,
        )
        return ranked[:top_k]
