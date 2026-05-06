"""Run PEPS on a single query."""

from __future__ import annotations

import argparse

from peps.entrypoints.common import (
    build_workflow_from_args,
    load_candidate_pool,
    load_example_library,
    load_query_input_from_args,
    print_or_write_json,
    save_candidate_pool,
    save_example_library,
    workflow_result_summary,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PEPS on one spatial reasoning query.")
    parser.add_argument("--query", required=True, help="Spatial reasoning query.")
    parser.add_argument("--image", action="append", default=[], help="Input image path/URL. Repeat for multiple views.")
    parser.add_argument("--choice", action="append", default=[], help="Answer choice. Repeat for multiple choices.")
    parser.add_argument("--query-id", default=None)
    parser.add_argument("--primitive-library", default="peps/data/primitive_library.yaml")
    parser.add_argument("--candidate-pool", default="peps/data/candidate_primitive_pool.json")
    parser.add_argument("--example-library", default="peps/data/example_library.json")
    parser.add_argument("--trace-dir", default="peps/data/traces")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--min-accept-score", type=float, default=0.0)
    parser.add_argument("--example-threshold", type=float, default=0.7)
    parser.add_argument("--model", default=None, help="Model override for all agents.")
    parser.add_argument("--parser-model", default=None)
    parser.add_argument("--executor-model", default=None)
    parser.add_argument("--coder-model", default=None)
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--no-save-traces", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    examples = load_example_library(args.example_library)
    candidates = load_candidate_pool(args.candidate_pool)
    workflow = build_workflow_from_args(
        args,
        example_library=examples,
        candidate_pool=candidates,
    )
    result = workflow.run(load_query_input_from_args(args))
    save_example_library(args.example_library, examples)
    save_candidate_pool(args.candidate_pool, candidates)
    print_or_write_json(workflow_result_summary(result), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
