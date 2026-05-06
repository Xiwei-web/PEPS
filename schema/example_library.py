"""Example Library for accepted high-quality PEPS traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from peps.core.io import append_jsonl, read_jsonl, to_plain_data, write_json
from peps.core.trace import ExecutionTrace
from peps.schema.retrieval import RankedItem, lexical_score


@dataclass(slots=True)
class ExampleRecord:
    """Compact retained example used by Parser and Executor prompts."""

    query: str
    answer: str
    trace_id: str
    quality_score: float
    primitive_ids: list[str] = field(default_factory=list)
    summary: str = ""
    example_id: str | None = None
    computed_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.example_id is None:
            self.example_id = self.trace_id
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("ExampleRecord.quality_score must be in [0, 1]")

    @classmethod
    def from_trace(cls, trace: ExecutionTrace, *, summary: str = "") -> "ExampleRecord":
        if trace.final_answer is None:
            raise ValueError("Cannot build ExampleRecord from trace without final_answer")
        if trace.verification is None:
            raise ValueError("Cannot build ExampleRecord from trace without verification")
        primitive_ids = (
            sorted(trace.requirements.definition_ids())
            if trace.requirements is not None
            else []
        )
        return cls(
            query=trace.requirements.query if trace.requirements else "",
            answer=trace.final_answer,
            trace_id=trace.trace_id,
            quality_score=trace.verification.quality_score,
            primitive_ids=primitive_ids,
            summary=summary,
            computed_metrics=trace.computed_metrics,
            metadata={"schema_version": trace.schema_version},
        )

    def search_text(self) -> str:
        return " ".join(
            [
                self.query,
                self.answer,
                self.summary,
                " ".join(self.primitive_ids),
                " ".join(str(value) for value in self.computed_metrics.values()),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return to_plain_data(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExampleRecord":
        return cls(**data)


class ExampleLibrary:
    """Persistent retrieval store for accepted high-score programs."""

    def __init__(self, examples: list[ExampleRecord] | None = None) -> None:
        self._examples: dict[str, ExampleRecord] = {}
        for example in examples or []:
            self.add(example)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ExampleLibrary":
        source = Path(path)
        if not source.exists():
            return cls()
        return cls([ExampleRecord.from_dict(row) for row in read_jsonl(source)])

    def add(self, example: ExampleRecord, *, overwrite: bool = True) -> ExampleRecord:
        if example.example_id in self._examples and not overwrite:
            raise ValueError(f"Duplicate example_id: {example.example_id}")
        self._examples[example.example_id or example.trace_id] = example
        return example

    def maybe_add_trace(
        self,
        trace: ExecutionTrace,
        *,
        threshold: float,
        summary: str = "",
    ) -> ExampleRecord | None:
        if trace.verification is None or trace.final_answer is None:
            return None
        if not trace.accepted() or trace.verification.quality_score < threshold:
            return None
        return self.add(ExampleRecord.from_trace(trace, summary=summary))

    def save_json(self, path: str | Path) -> None:
        write_json(path, [example.to_dict() for example in self.examples()])

    def append_jsonl(self, path: str | Path, example: ExampleRecord) -> None:
        append_jsonl(path, example.to_dict())

    def examples(self) -> list[ExampleRecord]:
        return list(self._examples.values())

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[RankedItem[ExampleRecord]]:
        ranked: list[RankedItem[ExampleRecord]] = []
        for example in self.examples():
            score, matched = lexical_score(query, example.search_text())
            if score >= min_score:
                ranked.append(RankedItem(example, score, matched))
        ranked.sort(key=lambda row: (row.score, row.item.quality_score), reverse=True)
        return ranked[:top_k]

