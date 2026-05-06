"""Validate or rebuild the PEPS primitive schema."""

from __future__ import annotations

import argparse

from peps.core.io import to_plain_data
from peps.entrypoints.common import print_or_write_json
from peps.schema import CandidatePrimitivePool
from peps.schema.schema_admission import AdmissionCriteria
from peps.schema.schema_builder import PrimitiveSchemaBuilder


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate/build a PEPS primitive library.")
    parser.add_argument("--library", default="peps/data/primitive_library.yaml")
    parser.add_argument("--output", default=None, help="Where to write the resulting primitive library.")
    parser.add_argument("--candidate-pool", default=None)
    parser.add_argument("--admit-candidates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-reuse-count", type=int, default=5)
    parser.add_argument("--min-accepted-count", type=int, default=1)
    parser.add_argument("--min-average-score", type=float, default=0.7)
    parser.add_argument("--min-average-improvement", type=float, default=0.0)
    parser.add_argument("--summary-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    builder = PrimitiveSchemaBuilder.from_file(args.library)
    validation = builder.library.validate(raise_on_error=True)
    report = None
    if args.admit_candidates:
        if not args.candidate_pool:
            raise ValueError("--candidate-pool is required with --admit-candidates")
        pool = CandidatePrimitivePool.load(args.candidate_pool)
        criteria = AdmissionCriteria(
            min_reuse_count=args.min_reuse_count,
            min_accepted_count=args.min_accepted_count,
            min_average_score=args.min_average_score,
            min_average_improvement=args.min_average_improvement,
        )
        report = builder.admit_candidates(pool, criteria=criteria, overwrite=args.overwrite)
        if str(args.candidate_pool).endswith(".jsonl"):
            pool.save_jsonl(args.candidate_pool)
        else:
            pool.save_json(args.candidate_pool)

    output = args.output or args.library
    builder.save(output)
    summary = {
        "library": output,
        "schema_version": builder.library.schema_version,
        "num_primitives": len(builder.library.definitions()),
        "validation_ok": validation.ok,
        "admission_report": to_plain_data(report) if report else None,
    }
    print_or_write_json(summary, args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
