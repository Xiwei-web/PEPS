"""Simple retrieval utilities for primitive and example libraries."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Generic, Iterable, TypeVar

from peps.core.types import PrimitiveDefinition

T = TypeVar("T")

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class RankedItem(Generic[T]):
    item: T
    score: float
    matched_terms: list[str]


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def lexical_score(query: str, document: str) -> tuple[float, list[str]]:
    query_terms = tokenize(query)
    doc_terms = tokenize(document)
    if not query_terms or not doc_terms:
        return 0.0, []
    overlap = query_terms & doc_terms
    union = query_terms | doc_terms
    return len(overlap) / len(union), sorted(overlap)


def rank_by_text(
    query: str,
    items: Iterable[T],
    text_fn,
    *,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[RankedItem[T]]:
    ranked: list[RankedItem[T]] = []
    for item in items:
        score, matched_terms = lexical_score(query, text_fn(item))
        if score >= min_score:
            ranked.append(RankedItem(item=item, score=score, matched_terms=matched_terms))
    ranked.sort(key=lambda row: row.score, reverse=True)
    return ranked[:top_k]


def primitive_search_text(definition: PrimitiveDefinition) -> str:
    tags = " ".join(definition.tags)
    examples = " ".join(str(example) for example in definition.examples)
    return " ".join(
        [
            definition.id,
            definition.name,
            definition.description,
            tags,
            examples,
        ]
    )


def rank_primitives(
    query: str,
    definitions: Iterable[PrimitiveDefinition],
    *,
    top_k: int = 8,
    min_score: float = 0.0,
) -> list[RankedItem[PrimitiveDefinition]]:
    return rank_by_text(
        query,
        definitions,
        primitive_search_text,
        top_k=top_k,
        min_score=min_score,
    )

