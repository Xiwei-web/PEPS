"""Evaluation runner for PEPS workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from peps.evals.datasets import EvalDataset, load_eval_dataset
from peps.evals.metrics import EvalPrediction, prediction_from_workflow_result
from peps.evals.report import EvalReport, build_eval_report, save_eval_report, save_predictions_jsonl
from peps.workflow.orchestrator import PEPSWorkflowOrchestrator


@dataclass(slots=True)
class EvalRunnerConfig:
    """Runtime settings for evaluation."""

    continue_on_error: bool = True
    save_predictions_path: str | Path | None = None
    save_report_path: str | Path | None = None


class EvalRunner:
    """Run a PEPS workflow over an EvalDataset."""

    def __init__(
        self,
        workflow: PEPSWorkflowOrchestrator,
        *,
        config: EvalRunnerConfig | None = None,
    ) -> None:
        self.workflow = workflow
        self.config = config or EvalRunnerConfig()

    def run_dataset(self, dataset: EvalDataset) -> EvalReport:
        predictions: list[EvalPrediction] = []
        for example in dataset.examples:
            try:
                result = self.workflow.run(example.to_query_input())
                predictions.append(prediction_from_workflow_result(example, result))
            except Exception as exc:
                if not self.config.continue_on_error:
                    raise
                predictions.append(
                    EvalPrediction(
                        example_id=example.example_id,
                        query=example.query,
                        gold_answer=example.answer,
                        predicted_answer=None,
                        accepted=False,
                        correct=False if example.answer is not None else None,
                        error=str(exc),
                    )
                )

        report = build_eval_report(
            dataset_name=dataset.name,
            predictions=predictions,
            metadata={
                "source": dataset.source,
                "dataset_metadata": dataset.metadata,
            },
        )
        if self.config.save_predictions_path:
            save_predictions_jsonl(predictions, self.config.save_predictions_path)
        if self.config.save_report_path:
            save_eval_report(report, self.config.save_report_path)
        return report

    def run_file(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> EvalReport:
        dataset = load_eval_dataset(path, name=name, limit=limit, offset=offset)
        return self.run_dataset(dataset)

