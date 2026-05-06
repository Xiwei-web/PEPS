"""Run PEPS evaluation from the command line."""

from __future__ import annotations

import argparse

from peps.entrypoints.common import (
    build_workflow_from_args,
    load_candidate_pool,
    load_example_library,
    save_candidate_pool,
    save_example_library,
)
from peps.evals.datasets import load_eval_dataset
from peps.evals.report import print_report_summary
from peps.evals.runner import EvalRunner, EvalRunnerConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PEPS on a JSON/JSONL dataset.")
    parser.add_argument("--input", required=True, help="Evaluation dataset JSON/JSONL.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--predictions-output", default="peps/data/reports/predictions.jsonl")
    parser.add_argument("--report-output", default="peps/data/reports/report.json")
    parser.add_argument("--primitive-library", default="peps/data/primitive_library.yaml")
    parser.add_argument("--candidate-pool", default="peps/data/candidate_primitive_pool.json")
    parser.add_argument("--example-library", default="peps/data/example_library.json")
    parser.add_argument("--trace-dir", default="peps/data/traces")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--min-accept-score", type=float, default=0.0)
    parser.add_argument("--example-threshold", type=float, default=0.7)
    parser.add_argument("--model", default=None)
    parser.add_argument("--parser-model", default=None)
    parser.add_argument("--executor-model", default=None)
    parser.add_argument("--coder-model", default=None)
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--no-save-traces", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset = load_eval_dataset(
        args.input,
        name=args.dataset_name,
        limit=args.limit,
        offset=args.offset,
    )
    examples = load_example_library(args.example_library)
    candidates = load_candidate_pool(args.candidate_pool)
    workflow = build_workflow_from_args(
        args,
        example_library=examples,
        candidate_pool=candidates,
    )
    report = EvalRunner(
        workflow,
        config=EvalRunnerConfig(
            continue_on_error=not args.fail_fast,
            save_predictions_path=args.predictions_output,
            save_report_path=args.report_output,
        ),
    ).run_dataset(dataset)
    save_example_library(args.example_library, examples)
    save_candidate_pool(args.candidate_pool, candidates)
    print_report_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

