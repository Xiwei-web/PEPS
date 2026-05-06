"""Promote qualified candidate primitives into the persistent library."""

from __future__ import annotations

import argparse

from peps.core.io import to_plain_data
from peps.entrypoints.common import print_or_write_json
from peps.schema import CandidatePrimitivePool, PrimitiveLibrary
from peps.schema.schema_admission import AdmissionCriteria, admit_candidate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote candidate primitives after admission checks.")
    parser.add_argument("--library", default="peps/data/primitive_library.yaml")
    parser.add_argument("--candidate-pool", default="peps/data/candidate_primitive_pool.json")
    parser.add_argument("--output-library", default=None)
    parser.add_argument("--output-pool", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-reuse-count", type=int, default=5)
    parser.add_argument("--min-accepted-count", type=int, default=1)
    parser.add_argument("--min-average-score", type=float, default=0.7)
    parser.add_argument("--min-average-improvement", type=float, default=0.0)
    parser.add_argument("--summary-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    library = PrimitiveLibrary.from_file(args.library)
    pool = CandidatePrimitivePool.load(args.candidate_pool)
    criteria = AdmissionCriteria(
        min_reuse_count=args.min_reuse_count,
        min_accepted_count=args.min_accepted_count,
        min_average_score=args.min_average_score,
        min_average_improvement=args.min_average_improvement,
    )

    decisions = []
    for record in pool.active():
        if args.dry_run:
            from peps.schema.schema_admission import evaluate_candidate_for_admission

            decision = evaluate_candidate_for_admission(record, library, criteria)
        else:
            decision = admit_candidate(record, library, criteria, overwrite=args.overwrite)
        decisions.append(decision)

    admitted = [
        decision.promoted_definition.id
        for decision in decisions
        if decision.admit and decision.promoted_definition is not None
    ]
    if not args.dry_run:
        library.save(args.output_library or args.library)
        pool_path = args.output_pool or args.candidate_pool
        if str(pool_path).endswith(".jsonl"):
            pool.save_jsonl(pool_path)
        else:
            pool.save_json(pool_path)

    summary = {
        "dry_run": args.dry_run,
        "admitted": admitted,
        "num_admitted": len(admitted),
        "num_evaluated": len(decisions),
        "decisions": to_plain_data(decisions),
    }
    print_or_write_json(summary, args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

