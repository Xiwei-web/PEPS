"""Evaluation report generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from peps.core.io import to_plain_data, write_json
from peps.evals.metrics import EvalMetrics, EvalPrediction, compute_eval_metrics


@dataclass(slots=True)
class EvalReport:
    """Complete PEPS evaluation report."""

    dataset_name: str
    metrics: EvalMetrics
    predictions: list[EvalPrediction]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return to_plain_data(self)


def build_eval_report(
    *,
    dataset_name: str,
    predictions: list[EvalPrediction],
    metadata: dict[str, Any] | None = None,
) -> EvalReport:
    return EvalReport(
        dataset_name=dataset_name,
        metrics=compute_eval_metrics(predictions),
        predictions=predictions,
        metadata=metadata or {},
    )


def save_eval_report(report: EvalReport, path: str | Path) -> None:
    write_json(path, report.to_dict())


def save_predictions_jsonl(predictions: list[EvalPrediction], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(to_plain_data(row), ensure_ascii=True) + "\n")


def print_report_summary(report: EvalReport) -> None:
    metrics = report.metrics
    payload = {
        "dataset": report.dataset_name,
        "total": metrics.total,
        "accuracy": metrics.accuracy,
        "accept_rate": metrics.accept_rate,
        "mean_verifier_score": metrics.mean_verifier_score,
        "feedback_type_counts": metrics.feedback_type_counts,
        "error_count": metrics.error_count,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))

