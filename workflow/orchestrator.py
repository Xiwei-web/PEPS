"""End-to-end PEPS workflow orchestrator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from peps.agents.coder_agent import CoderAgent
from peps.agents.executor_agent import ExecutorAgent
from peps.agents.parser_agent import ParserAgent, ParserAgentConfig
from peps.agents.verifier_agent import VerifierAgent
from peps.core.enums import FeedbackType
from peps.core.trace import ExecutionTrace, VerificationResult
from peps.core.types import QueryInput
from peps.schema.candidate_pool import CandidatePrimitivePool
from peps.schema.example_library import ExampleLibrary
from peps.schema.primitive_library import PrimitiveLibrary
from peps.tools.tool_registry import ToolRegistry, build_default_tool_registry
from peps.tools.workspace import ToolWorkspace
from peps.workflow.cache import WorkflowCache
from peps.workflow.loop_controller import LoopController, WorkflowLoopConfig
from peps.workflow.refinement import ParserRefinementContext, RefinementManager
from peps.workflow.router import WorkflowRoute


@dataclass(slots=True)
class PEPSWorkflowConfig:
    """End-to-end workflow settings."""

    max_rounds: int = 3
    min_accept_score: float = 0.0
    example_retention_threshold: float = 0.7
    parser_model: str | None = None
    executor_model: str | None = None
    coder_model: str | None = None
    verifier_model: str | None = None
    save_traces: bool = True
    trace_cache_dir: str | Path = "peps/data/traces"


@dataclass(slots=True)
class PEPSWorkflowAttempt:
    """One Parser -> Executor -> Coder -> Verifier attempt."""

    round_index: int
    trace: ExecutionTrace
    verification: VerificationResult | None = None
    route: WorkflowRoute | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.route is WorkflowRoute.ACCEPT


@dataclass(slots=True)
class PEPSWorkflowResult:
    """Final PEPS workflow result."""

    accepted: bool
    attempts: list[PEPSWorkflowAttempt]
    final_trace: ExecutionTrace | None = None
    final_answer: str | None = None
    last_feedback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PEPSWorkflowOrchestrator:
    """Run PEPS end to end with Parser, Executor, Coder, and Verifier agents."""

    def __init__(
        self,
        *,
        parser: ParserAgent,
        executor: ExecutorAgent,
        coder: CoderAgent,
        verifier: VerifierAgent,
        config: PEPSWorkflowConfig | None = None,
        example_library: ExampleLibrary | None = None,
        candidate_pool: CandidatePrimitivePool | None = None,
        loop_controller: LoopController | None = None,
        refinement_manager: RefinementManager | None = None,
        cache: WorkflowCache | None = None,
    ) -> None:
        self.parser = parser
        self.executor = executor
        self.coder = coder
        self.verifier = verifier
        self.config = config or PEPSWorkflowConfig()
        self.example_library = example_library
        self.candidate_pool = candidate_pool
        self.loop_controller = loop_controller or LoopController(
            WorkflowLoopConfig(
                max_rounds=self.config.max_rounds,
                min_accept_score=self.config.min_accept_score,
            )
        )
        self.refinement_manager = refinement_manager or RefinementManager(candidate_pool)
        self.cache = cache or WorkflowCache(self.config.trace_cache_dir)

    @classmethod
    def from_defaults(
        cls,
        *,
        primitive_library_path: str | Path = "peps/data/primitive_library.yaml",
        tool_registry: ToolRegistry | None = None,
        config: PEPSWorkflowConfig | None = None,
        example_library: ExampleLibrary | None = None,
        candidate_pool: CandidatePrimitivePool | None = None,
    ) -> "PEPSWorkflowOrchestrator":
        primitive_library = PrimitiveLibrary.from_file(primitive_library_path)
        parser = ParserAgent(
            primitive_library=primitive_library,
            example_library=example_library,
            config=ParserAgentConfig(allow_primitive_gap=True),
        )
        return cls(
            parser=parser,
            executor=ExecutorAgent(tool_registry=tool_registry or build_default_tool_registry()),
            coder=CoderAgent(),
            verifier=VerifierAgent(),
            config=config,
            example_library=example_library,
            candidate_pool=candidate_pool,
        )

    def run(self, query_input: QueryInput | str) -> PEPSWorkflowResult:
        """Synchronous entrypoint."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(query_input))
        raise RuntimeError("PEPSWorkflowOrchestrator.run cannot be called inside a running event loop")

    async def arun(self, query_input: QueryInput | str) -> PEPSWorkflowResult:
        query_input = self._coerce_query_input(query_input)
        attempts: list[PEPSWorkflowAttempt] = []
        refinement = ParserRefinementContext()

        for round_index in range(self.config.max_rounds):
            trace = ExecutionTrace(query_id=query_input.query_id)
            trace.metadata["round_index"] = round_index
            attempt = PEPSWorkflowAttempt(round_index=round_index, trace=trace)
            attempts.append(attempt)

            try:
                parser_result = self.parser.parse(
                    query_input,
                    verifier_feedback=refinement.feedback,
                    feedback_type=refinement.feedback_type,
                    candidate_hypotheses=refinement.candidate_hypotheses,
                    model=self.config.parser_model,
                )
                trace.add_agent_run(parser_result.agent_run)
                trace.set_requirements(parser_result.requirements)

                workspace = ToolWorkspace()
                executor_result = await self.executor.arun(
                    parser_result.requirements,
                    query_input=query_input,
                    workspace=workspace,
                    trace=trace,
                    model=self.config.executor_model,
                )

                coder_result = self.coder.run(
                    parser_result.requirements,
                    executor_result.workspace,
                    query_input=query_input,
                    trace=trace,
                    model=self.config.coder_model,
                )

                verifier_result = self.verifier.verify(
                    query_input,
                    trace,
                    candidate_answer=coder_result.answer,
                    model=self.config.verifier_model,
                )
                attempt.verification = verifier_result.verification

                decision = self.loop_controller.evaluate(
                    verifier_result.verification,
                    round_index=round_index,
                )
                attempt.route = decision.route
                attempt.metadata["route_reason"] = decision.reason

                if self.config.save_traces:
                    self.cache.save_trace(
                        trace,
                        name=self._trace_name(query_input, round_index, trace.trace_id),
                    )

                if decision.route is WorkflowRoute.ACCEPT:
                    self._maybe_store_example(trace)
                    return PEPSWorkflowResult(
                        accepted=True,
                        attempts=attempts,
                        final_trace=trace,
                        final_answer=trace.final_answer,
                        metadata={"route_reason": decision.reason},
                    )

                if decision.route is not WorkflowRoute.REFINE:
                    return PEPSWorkflowResult(
                        accepted=False,
                        attempts=attempts,
                        final_trace=trace,
                        final_answer=trace.final_answer,
                        last_feedback=verifier_result.verification.feedback,
                        metadata={"route_reason": decision.reason},
                    )

                refinement = self.refinement_manager.prepare_next_parser_context(
                    verifier_result.verification,
                    query=query_input.query,
                    trace=trace,
                )

            except Exception as exc:
                attempt.error = str(exc)
                trace.errors.append(str(exc))
                if self.config.save_traces:
                    self.cache.save_trace(
                        trace,
                        name=self._trace_name(query_input, round_index, trace.trace_id),
                    )
                return PEPSWorkflowResult(
                    accepted=False,
                    attempts=attempts,
                    final_trace=trace,
                    final_answer=trace.final_answer,
                    last_feedback=refinement.feedback,
                    metadata={"error": str(exc)},
                )

        final_attempt = attempts[-1] if attempts else None
        final_trace = final_attempt.trace if final_attempt else None
        return PEPSWorkflowResult(
            accepted=False,
            attempts=attempts,
            final_trace=final_trace,
            final_answer=final_trace.final_answer if final_trace else None,
            last_feedback=(
                final_trace.verification.feedback
                if final_trace and final_trace.verification
                else refinement.feedback
            ),
            metadata={"route_reason": "Maximum PEPS workflow rounds reached."},
        )

    def _maybe_store_example(self, trace: ExecutionTrace) -> None:
        if self.example_library is None:
            return
        self.example_library.maybe_add_trace(
            trace,
            threshold=self.config.example_retention_threshold,
            summary="Accepted PEPS workflow trace.",
        )

    def _coerce_query_input(self, query_input: QueryInput | str) -> QueryInput:
        if isinstance(query_input, QueryInput):
            return query_input
        if isinstance(query_input, str):
            return QueryInput(query=query_input)
        raise TypeError(f"Unsupported query_input type: {type(query_input).__name__}")

    def _trace_name(self, query_input: QueryInput, round_index: int, trace_id: str) -> str:
        query_key = query_input.query_id or self.cache.key_for_text(query_input.query)
        return f"{query_key}_round_{round_index}_{trace_id}"

