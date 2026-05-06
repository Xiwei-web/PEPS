"""Evaluation metrics for PEPS outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any

from peps.core.enums import FeedbackType


@dataclass(slots=True)
class EvalPrediction:
    """One evaluated prediction row."""

    example_id: str | None
    query: str
    gold_answer: str | None
    predicted_answer: str | None
    accepted: bool
    verifier_score: float | None = None
    feedback_type: str | None = None
    correct: bool | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalMetrics:
    """Aggregated evaluation metrics."""

    total: int
    evaluated: int
    correct: int
    accuracy: float | None
    accepted: int
    accept_rate: float
    mean_verifier_score: float | None
    feedback_type_counts: dict[str, int]
    error_count: int


def normalize_answer(answer: Any) -> str:
    if answer is None:
        return ""
    text = str(answer).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;!?\"'")
    aliases = {
        "yes.": "yes",
        "no.": "no",
        "left side": "left",
        "right side": "right",
        "in front": "front",
        "front of": "front",
        "behind of": "behind",
        "back": "behind",
    }
    return aliases.get(text, text)


def answers_match(predicted: Any, gold: Any) -> bool:
    pred = normalize_answer(predicted)
    target = normalize_answer(gold)
    if not pred or not target:
        return False
    if pred == target:
        return True
    pred_tokens = set(pred.split())
    target_tokens = set(target.split())
    if len(target_tokens) == 1 and target in pred_tokens:
        return True
    if len(pred_tokens) == 1 and pred in target_tokens:
        return True
    return False


def compute_eval_metrics(predictions: list[EvalPrediction]) -> EvalMetrics:
    total = len(predictions)
    evaluable = [row for row in predictions if row.gold_answer is not None and row.correct is not None]
    correct = sum(1 for row in evaluable if row.correct)
    accepted = sum(1 for row in predictions if row.accepted)
    scores = [row.verifier_score for row in predictions if row.verifier_score is not None]
    feedback_counts = Counter(row.feedback_type or "none" for row in predictions)
    error_count = sum(1 for row in predictions if row.error)
    return EvalMetrics(
        total=total,
        evaluated=len(evaluable),
        correct=correct,
        accuracy=correct / len(evaluable) if evaluable else None,
        accepted=accepted,
        accept_rate=accepted / total if total else 0.0,
        mean_verifier_score=sum(scores) / len(scores) if scores else None,
        feedback_type_counts=dict(feedback_counts),
        error_count=error_count,
    )


def prediction_from_workflow_result(example: Any, result: Any, *, error: str | None = None) -> EvalPrediction:
    final_trace = result.final_trace if result is not None else None
    verification = final_trace.verification if final_trace and final_trace.verification else None
    predicted = result.final_answer if result is not None else None
    gold = example.answer
    correct = answers_match(predicted, gold) if gold is not None and predicted is not None else None
    return EvalPrediction(
        example_id=example.example_id,
        query=example.query,
        gold_answer=gold,
        predicted_answer=predicted,
        accepted=bool(result.accepted) if result is not None else False,
        verifier_score=verification.quality_score if verification else None,
        feedback_type=verification.feedback_type.value if verification else None,
        correct=correct,
        error=error,
        metadata={
            "trace_id": final_trace.trace_id if final_trace else None,
            "num_attempts": len(result.attempts) if result is not None else 0,
        },
    )

