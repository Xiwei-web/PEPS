"""Inspect a saved PEPS trace."""

from __future__ import annotations

import argparse
from pathlib import Path

from peps.core.io import execution_trace_from_dict, read_json
from peps.entrypoints.common import print_or_write_json, trace_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a saved PEPS trace JSON file.")
    parser.add_argument("trace", help="Path to a saved trace JSON file.")
    parser.add_argument("--show-code", action="store_true")
    parser.add_argument("--show-workspace", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON summary output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path = Path(args.trace)
    trace = execution_trace_from_dict(read_json(path))
    print_or_write_json(
        trace_summary(trace, show_code=args.show_code, show_workspace=args.show_workspace),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

