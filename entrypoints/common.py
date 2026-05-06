"""Shared helpers for PEPS command-line entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from peps.agents.coder_agent import CoderAgent
from peps.agents.executor_agent import ExecutorAgent
from peps.agents.parser_agent import ParserAgent, ParserAgentConfig
from peps.agents.verifier_agent import VerifierAgent
from peps.core.io import read_json, read_jsonl, to_plain_data, write_json
from peps.core.trace import ExecutionTrace
from peps.core.types import ImageRef, QueryInput
from peps.schema.candidate_pool import CandidatePrimitivePool
from peps.schema.example_library import ExampleLibrary, ExampleRecord
from peps.schema.primitive_library import PrimitiveLibrary
from peps.tools.tool_registry import build_default_tool_registry
from peps.workflow.cache import WorkflowCache
from peps.workflow.orchestrator import PEPSWorkflowConfig, PEPSWorkflowOrchestrator


def load_query_input_from_args(args: Any) -> QueryInput:
    images = [
        ImageRef(uri=uri, view_id=f"view_{index}")
        for index, uri in enumerate(args.image or [])
    ]
    return QueryInput(
        query=args.query,
        images=images,
        query_id=args.query_id,
        choices=args.choice or [],
    )


def query_input_from_record(record: dict[str, Any], *, index: int) -> QueryInput:
    image_rows = record.get("images", record.get("image", []))
    if isinstance(image_rows, str):
        image_rows = [image_rows]
    images: list[ImageRef] = []
    for image_index, row in enumerate(image_rows or []):
        if isinstance(row, dict):
            images.append(ImageRef(**row))
        else:
            images.append(ImageRef(uri=str(row), view_id=f"view_{image_index}"))
    choices = record.get("choices", record.get("choice", []))
    if isinstance(choices, str):
        choices = [choices]
    return QueryInput(
        query=record["query"],
        images=images,
        query_id=record.get("query_id", record.get("id", f"query_{index}")),
        choices=choices or [],
        metadata=record.get("metadata", {}),
    )


def load_query_records(path: str | Path) -> list[QueryInput]:
    source = Path(path)
    rows = read_jsonl(source) if source.suffix == ".jsonl" else read_json(source)
    if isinstance(rows, dict):
        rows = rows.get("queries", [rows])
    if not isinstance(rows, list):
        raise ValueError("Batch input must be a JSON list, JSONL file, or {'queries': [...]} mapping")
    return [query_input_from_record(row, index=index) for index, row in enumerate(rows)]


def load_example_library(path: str | Path | None) -> ExampleLibrary:
    if path is None:
        return ExampleLibrary()
    source = Path(path)
    if not source.exists():
        return ExampleLibrary()
    if source.suffix == ".jsonl":
        return ExampleLibrary.from_jsonl(source)
    rows = read_json(source)
    if isinstance(rows, dict):
        rows = rows.get("examples", [])
    return ExampleLibrary([ExampleRecord.from_dict(row) for row in rows])


def save_example_library(path: str | Path | None, library: ExampleLibrary) -> None:
    if path is None:
        return
    target = Path(path)
    if target.suffix == ".jsonl":
        if target.exists():
            target.unlink()
        for example in library.examples():
            library.append_jsonl(target, example)
    else:
        library.save_json(target)


def load_candidate_pool(path: str | Path | None) -> CandidatePrimitivePool:
    if path is None:
        return CandidatePrimitivePool()
    return CandidatePrimitivePool.load(path)


def save_candidate_pool(path: str | Path | None, pool: CandidatePrimitivePool) -> None:
    if path is None:
        return
    target = Path(path)
    if target.suffix == ".jsonl":
        pool.save_jsonl(target)
    else:
        pool.save_json(target)


def build_workflow_from_args(
    args: Any,
    *,
    example_library: ExampleLibrary | None = None,
    candidate_pool: CandidatePrimitivePool | None = None,
) -> PEPSWorkflowOrchestrator:
    primitive_library = PrimitiveLibrary.from_file(args.primitive_library)
    tool_registry = build_default_tool_registry()
    examples = example_library or ExampleLibrary()
    candidates = candidate_pool or CandidatePrimitivePool()
    config = PEPSWorkflowConfig(
        max_rounds=args.max_rounds,
        min_accept_score=args.min_accept_score,
        example_retention_threshold=args.example_threshold,
        parser_model=args.parser_model or args.model,
        executor_model=args.executor_model or args.model,
        coder_model=args.coder_model or args.model,
        verifier_model=args.verifier_model or args.model,
        save_traces=not args.no_save_traces,
        trace_cache_dir=args.trace_dir,
    )
    return PEPSWorkflowOrchestrator(
        parser=ParserAgent(
            primitive_library=primitive_library,
            example_library=examples,
            config=ParserAgentConfig(allow_primitive_gap=True),
        ),
        executor=ExecutorAgent(tool_registry=tool_registry),
        coder=CoderAgent(),
        verifier=VerifierAgent(),
        config=config,
        example_library=examples,
        candidate_pool=candidates,
        cache=WorkflowCache(args.trace_dir),
    )


def workflow_result_summary(result: Any) -> dict[str, Any]:
    final_verification = (
        to_plain_data(result.final_trace.verification)
        if result.final_trace is not None and result.final_trace.verification is not None
        else None
    )
    return {
        "accepted": result.accepted,
        "final_answer": result.final_answer,
        "num_attempts": len(result.attempts),
        "final_trace_id": result.final_trace.trace_id if result.final_trace else None,
        "verification": final_verification,
        "last_feedback": result.last_feedback,
        "metadata": result.metadata,
    }


def print_or_write_json(data: Any, output: str | Path | None = None) -> None:
    if output:
        write_json(output, data)
    else:
        print(json.dumps(to_plain_data(data), ensure_ascii=True, indent=2))


def trace_summary(trace: ExecutionTrace, *, show_code: bool = False, show_workspace: bool = False) -> dict[str, Any]:
    requirements = trace.requirements
    summary = {
        "trace_id": trace.trace_id,
        "query_id": trace.query_id,
        "schema_version": trace.schema_version,
        "query": requirements.query if requirements else None,
        "final_answer": trace.final_answer,
        "verification": to_plain_data(trace.verification),
        "num_tool_calls": len(trace.tool_calls),
        "num_code_runs": len(trace.code_runs),
        "errors": trace.errors,
        "requirements": {
            "frame": [item.definition_id for item in requirements.frame] if requirements else [],
            "entity": [item.definition_id for item in requirements.entity] if requirements else [],
            "state": [item.definition_id for item in requirements.state] if requirements else [],
            "metric": [item.definition_id for item in requirements.metric] if requirements else [],
        },
        "tool_calls": [
            {
                "tool_name": call.tool_name,
                "status": call.status.value,
                "fills": call.fills,
                "result_refs": call.result_refs,
                "error": call.error,
            }
            for call in trace.tool_calls
        ],
        "code_runs": [
            {
                "run_id": run.run_id,
                "status": run.status.value,
                "answer": run.answer,
                "computed_metrics": run.computed_metrics,
                "error": run.error,
                **({"code": run.code} if show_code else {}),
            }
            for run in trace.code_runs
        ],
    }
    if show_workspace:
        summary["workspace_snapshot"] = trace.workspace_snapshot
    return summary

