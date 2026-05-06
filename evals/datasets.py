"""Dataset loading utilities for PEPS evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from peps.core.io import read_json, read_jsonl
from peps.core.types import ImageRef, QueryInput


@dataclass(slots=True)
class EvalExample:
    """One spatial reasoning evaluation example."""

    query: str
    answer: str | None = None
    images: list[ImageRef] = field(default_factory=list)
    choices: list[str] = field(default_factory=list)
    example_id: str | None = None
    dataset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_query_input(self) -> QueryInput:
        return QueryInput(
            query=self.query,
            images=self.images,
            query_id=self.example_id,
            choices=self.choices,
            metadata={
                "dataset": self.dataset,
                **self.metadata,
            },
        )


@dataclass(slots=True)
class EvalDataset:
    """A loaded PEPS evaluation dataset."""

    name: str
    examples: list[EvalExample]
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    def subset(self, limit: int | None = None, offset: int = 0) -> "EvalDataset":
        rows = self.examples[offset:]
        if limit is not None:
            rows = rows[:limit]
        return EvalDataset(
            name=self.name,
            examples=rows,
            source=self.source,
            metadata={**self.metadata, "offset": offset, "limit": limit},
        )


def load_eval_dataset(
    path: str | Path,
    *,
    name: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> EvalDataset:
    source = Path(path)
    raw = read_jsonl(source) if source.suffix == ".jsonl" else read_json(source)
    dataset_name = name or source.stem
    metadata: dict[str, Any] = {}
    if isinstance(raw, dict):
        dataset_name = name or raw.get("name", dataset_name)
        metadata = raw.get("metadata", {})
        rows = raw.get("examples") or raw.get("queries") or raw.get("data") or []
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError("Eval dataset must be a list or mapping with examples/queries/data")
    examples = [
        eval_example_from_record(row, index=index, dataset=dataset_name)
        for index, row in enumerate(rows)
    ]
    return EvalDataset(
        name=dataset_name,
        examples=examples,
        source=str(source),
        metadata=metadata,
    ).subset(limit=limit, offset=offset)


def eval_example_from_record(
    record: dict[str, Any],
    *,
    index: int,
    dataset: str | None = None,
) -> EvalExample:
    query = record.get("query") or record.get("question") or record.get("instruction")
    if not query:
        raise ValueError(f"Eval record {index} is missing query/question/instruction")
    answer = (
        record.get("answer")
        or record.get("label")
        or record.get("target")
        or record.get("gold")
        or record.get("gt_answer")
    )
    images = _parse_images(record.get("images", record.get("image", [])))
    choices = record.get("choices", record.get("options", [])) or []
    if isinstance(choices, dict):
        choices = [str(value) for value in choices.values()]
    if isinstance(choices, str):
        choices = [choices]
    return EvalExample(
        query=str(query),
        answer=str(answer) if answer is not None else None,
        images=images,
        choices=[str(choice) for choice in choices],
        example_id=str(record.get("id", record.get("query_id", f"{dataset or 'example'}_{index}"))),
        dataset=dataset,
        metadata={key: value for key, value in record.items() if key not in _KNOWN_FIELDS},
    )


def _parse_images(raw: Any) -> list[ImageRef]:
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        raw = [raw]
    images: list[ImageRef] = []
    for index, item in enumerate(raw):
        if isinstance(item, ImageRef):
            images.append(item)
        elif isinstance(item, dict):
            payload = dict(item)
            payload.setdefault("view_id", f"view_{index}")
            images.append(ImageRef(**payload))
        else:
            images.append(ImageRef(uri=str(item), view_id=f"view_{index}"))
    return images


_KNOWN_FIELDS = {
    "query",
    "question",
    "instruction",
    "answer",
    "label",
    "target",
    "gold",
    "gt_answer",
    "images",
    "image",
    "choices",
    "options",
    "id",
    "query_id",
}

