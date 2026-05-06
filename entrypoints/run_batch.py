"""Run PEPS on a JSON/JSONL batch of queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peps.core.io import to_plain_data
from peps.entrypoints.common import (
    build_workflow_from_args,
    load_candidate_pool,
    load_example_library,
    load_query_records,
    save_candidate_pool,
    save_example_library,
    workflow_result_summary,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PEPS on a batch of queries.")
    parser.add_argument("--input", required=True, help="JSON or JSONL batch file.")
    parser.add_argument("--output", required=True, help="JSONL output path.")
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
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    queries = load_query_records(args.input)
    examples = load_example_library(args.example_library)
    candidates = load_candidate_pool(args.candidate_pool)
    workflow = build_workflow_from_args(
        args,
        example_library=examples,
        candidate_pool=candidates,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, query_input in enumerate(queries):
            try:
                result = workflow.run(query_input)
                row = {
                    "index": index,
                    "query_id": query_input.query_id,
                    **workflow_result_summary(result),
                }
                accepted += int(result.accepted)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                row = {
                    "index": index,
                    "query_id": query_input.query_id,
                    "accepted": False,
                    "error": str(exc),
                }
            handle.write(json.dumps(to_plain_data(row), ensure_ascii=True) + "\n")

    save_example_library(args.example_library, examples)
    save_candidate_pool(args.candidate_pool, candidates)
    print(json.dumps({"total": len(queries), "accepted": accepted, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

